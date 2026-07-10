# SSL AudioNoise dataloader — analysis & benchmark results

Environment: 4-CPU Linux container, no GPU (pure dataloader throughput), warm page cache.
Data: dummy 16 kHz wavs, durations uniform in [15, 20] s — silent main wavs, zero-mean
random-noise wavs for the noise manifest (256 main files / 4096 manifest entries / 64 noise files).
Dataloader: `AudioNoiseDataset` + `MultiSpeakerNoiseAugmentation` (NEST `train_ds` defaults,
`prob=0.5`, 1 segment / 1 speaker), built via `get_audio_noise_dataset_from_config`.

## TL;DR

The dataloader is CPU-bound at **~2 samples/s per worker** (~0.45 s per item). Throughput is
**flat with batch size** — a bigger batch just takes proportionally longer to assemble — and
scales only linearly with workers, so at moderate batch sizes the GPU starves no matter what.
Two Python-level hotspots in the noise-sampling path cause ~99% of the cost; fixing both makes
the same dataloader **~40–60× faster** (390 samples/s at bs=16/nw=4) and no longer the bottleneck.

## Root causes (in order of impact)

### 1. Broken "empty segment" retry loop in `load_noise_audio` (`ssl_dataset.py:189-218`)

When the sampled noise clip is longer than the target length, the loader picks a random offset
and retries while `sum(audio_segment.samples) > 0` is false, up to `max_trial=100` times:

- **`sum()` is the Python builtin over a numpy array** — ~20 ms per call for a 15–20 s clip
  (vs 0.15 ms for the numpy equivalent), i.e. 100× slower than the decode itself (~1.4 ms).
- **The predicate tests the signed sum, not emptiness.** For zero-mean audio the sign of the
  sum is a coin flip, and all 100 trial windows of the same file overlap heavily (durations
  15–20 s vs targets 15–20 s), so their sums are almost perfectly correlated: if the first
  window sums negative, *all 100 trials fail* and each pays a full decode + 20 ms Python sum.

Measured per-item latency is bimodal: **~20 ms normally vs ~1.9 s** (101 `AudioSegment.from_file`
calls) when a noise file draws a negative sum — roughly half the noise files, whenever
`noise_duration > main_duration`. This is not an artifact of dummy data: any zero-mean real
noise corpus behaves the same. The same slow `sum(...) == 0` check also runs once per
successful load (line 214), putting a ~20 ms floor under every item.

### 2. `Counter(random.choices(range(n), k=mix_len))` in the batch augmentor (`augmentation.py:201`)

`mix_len` is in **samples** (~160 k for a 20 s clip at `mix_rate=0.5`), so segment lengths are
chosen by drawing 160 k Python-level random numbers and counting them: ~16–50 ms per augmented
item, even with `num_segments=1` where the answer is trivially `[mix_len]`. An
`np.random.multinomial(mix_len, ...)` draw is distribution-identical and O(num_segments).

For reference, actually decoding a 15–20 s wav costs ~1.4 ms — the real I/O is negligible
next to these two Python loops.

## Benchmark: batch-size scaling at `num_workers=4`

Baseline (noise manifest + augmentor, as in NEST config) vs the same dataloader with the two
fixes monkeypatched in (`--patched`), and a no-noise-manifest contrast:

| batch size | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s | no-noise samples/s |
|-----------:|-----------------:|-------------------:|----------------:|------------------:|-------------------:|
| 4          | 0.42             | 9.6                | 0.011           | 352               | 214                |
| 8          | 0.81             | 9.9                | 0.021           | 391               | —                  |
| 16         | 1.87             | 8.6                | 0.041           | 388               | 239                |
| 32         | 5.13             | 6.2                | 0.094           | 341               | —                  |
| 64         | 7.76             | 8.3                | 0.181           | 354               | 166                |

Baseline throughput is **flat at ~6–10 samples/s** regardless of batch size: batch assembly time
grows linearly (0.42 s → 7.8 s per batch), which is exactly the "gradient accumulation instead of
bigger batches" symptom — the loader can't fill a big batch any faster than N small ones.
Time-to-first-batch also balloons (2 s → 32 s). Patched, throughput is ~355–390 samples/s and
still flat — but at that level a batch of 64 takes 0.18 s to build, comfortably ahead of a
typical training step.

## Benchmark: worker scaling at `batch_size=16`

| num_workers | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s |
|------------:|-----------------:|-------------------:|----------------:|------------------:|
| 0           | 7.47             | 2.1                | 0.067           | 238               |
| 1           | 7.96             | 2.0                | 0.101           | 158               |
| 2           | 3.28             | 4.9                | 0.048           | 333               |
| 4           | 2.08             | 7.7                | 0.041           | 393               |
| 8           | 1.40             | 11.4               | 0.042           | 385               |

Baseline scales ~linearly with workers (2.1 → 11.4 samples/s from 1× to 8× on a 4-core box —
oversubscription helps a little because item cost is bimodal and stragglers dominate). Linear
scaling from a 2 samples/s-per-worker base is hopeless: feeding a GPU that consumes a bs=32
batch every ~0.3 s would need ~50 workers. Patched, even `num_workers=0` (238 samples/s) beats
the baseline with 8 workers by 20×, and scaling saturates at 4 workers because the loader is no
longer CPU-bound.

## The literal "empty wav" case (silent noise files)

With truly silent noise wavs, every item where `noise_duration > main_duration` exhausts all
100 trials deterministically and then falls back to `WhiteNoisePerturbation` (plus a warning
log per item): 1.51 s/batch at bs=8/nw=4 (5.3 samples/s). Silent stretches in a real noise
corpus hit the same path.

## Recommended fixes

1. `ssl_dataset.py` — replace both emptiness checks with a numpy energy test, e.g.
   `np.abs(audio_segment.samples).max() > 0` (and `== 0` for the fallback check). This removes
   the 20 ms/item floor *and* the coin-flip retry storm, since any window with nonzero content
   passes on the first trial.
2. `augmentation.py:201` — replace `Counter(random.choices(range(num_segments), k=mix_len))`
   with `np.random.multinomial(mix_len, np.ones(n)/n)` (drop zero counts to keep semantics).
3. Optional hardening: skip the offset-retry decode loop entirely when the noise file is known
   non-silent, and rate-limit the "empty noise" warning which does console I/O per item.

`LhotseAudioNoiseDataset` shares `sample_noise`/`load_noise_audio` and therefore hotspot #1;
it additionally loads the *whole batch* inside a single `__getitem__`, so its batch-size
scaling is structurally worse (workers pipeline whole batches instead of items).

Raw numbers: `results.jsonl` (one JSON object per run, produced by `run_sweeps.sh`).
