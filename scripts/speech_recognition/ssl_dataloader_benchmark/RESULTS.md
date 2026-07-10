# SSL AudioNoise dataloader — analysis & benchmark results

Environment: 4-CPU Linux container, no GPU (pure dataloader throughput), warm page cache.
Data: dummy 16 kHz wavs, durations uniform in [15, 20] s — silent main wavs, zero-mean
random-noise wavs for the noise manifest (256 main files / 16384 manifest entries / 64 noise files).
Both variants are built through the exact model code paths
(`EncDecDenoiseMaskedTokenPredModel._setup_dataloader_from_config`):

- **map-style**: `AudioNoiseDataset` + `MultiSpeakerNoiseAugmentation` (NEST `train_ds` defaults)
- **lhotse**: `LhotseAudioNoiseDataset` via `get_lhotse_dataloader_from_config` (`use_lhotse: true`)

Measurement methodology: after warmup, batches are consumed until BOTH a minimum batch count
and a minimum wall-time window (40 s baseline / 15 s patched) elapse, so the steady-state
production rate dominates over draining the `num_workers x prefetch_factor` batch queue.

## TL;DR

Both variants are CPU-bound in the noise-sampling path, not in audio I/O (decoding a 15–20 s
wav takes ~1.4 ms). The map-style loader saturates at **~2 samples/s per worker**; throughput is
flat with batch size, so a bigger batch just takes proportionally longer to assemble and the GPU
starves. The Lhotse variant looks 2–10× faster *on this dummy data* but only because a padding
artifact masks the main pathology (see below) — on realistic data (noise recordings longer than
utterances) it degrades to the same behavior. Fixing two Python-level hotspots makes **both**
variants **~40–60× faster** (~330–420 samples/s), flat across batch sizes, and no longer the
bottleneck.

## Root causes (in order of impact)

### 1. Broken "empty segment" retry loop in `load_noise_audio` (`ssl_dataset.py:189-218`)

When the sampled noise clip is longer than the target length, the loader picks a random offset
and retries while `sum(audio_segment.samples) > 0` is false, up to `max_trial=100` times:

- **`sum()` is the Python builtin over a numpy array** — ~20 ms per call for a 15–20 s clip
  (vs 0.15 ms for the numpy equivalent), i.e. 100× slower than the decode itself (~1.4 ms).
- **The predicate tests the signed sum, not emptiness.** For zero-mean audio the sign of the
  sum is a coin flip, and all 100 trial windows of the same file overlap heavily, so their sums
  are almost perfectly correlated: if the first window sums negative, *all 100 trials fail* and
  each pays a full decode + 20 ms Python sum.

Measured per-item latency is bimodal: **~20 ms normally vs ~1.9 s** (101 `AudioSegment.from_file`
calls) when a noise file draws a negative sum — roughly half the noise files, whenever
`noise_duration > target_length`. This is not an artifact of dummy data: any zero-mean real
noise corpus behaves the same. The same slow `sum(...) == 0` check also runs once per
successful load (line 214), putting a ~20 ms floor under every item.

### 2. `Counter(random.choices(range(n), k=mix_len))` in the batch augmentor (`augmentation.py:201`)

`mix_len` is in **samples** (~160 k for a 20 s clip at `mix_rate=0.5`), so segment lengths are
chosen by drawing 160 k Python-level random numbers and counting them: ~16–50 ms per augmented
item, even with `num_segments=1` where the answer is trivially `[mix_len]`. An
`np.random.multinomial(mix_len, ...)` draw is distribution-identical and O(num_segments).

## Map-style variant

### Batch-size scaling at `num_workers=4`

| batch size | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s |
|-----------:|-----------------:|-------------------:|----------------:|------------------:|
| 4          | 0.61             | 6.5                | 0.011           | 365               |
| 8          | 1.19             | 6.8                | 0.020           | 400               |
| 16         | 1.89             | 8.5                | 0.048           | 331               |
| 32         | 3.75             | 8.5                | 0.089           | 358               |
| 64         | 7.15             | 9.0                | 0.178           | 359               |

No-noise-manifest contrast (zeros noise, augmentor on): 239–277 samples/s — the noise path is
~30× the cost of everything else combined.

### Worker scaling at `batch_size=16`

| num_workers | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s |
|------------:|-----------------:|-------------------:|----------------:|------------------:|
| 0           | 7.51             | 2.1                | 0.061           | 264               |
| 1           | 6.82             | 2.4                | 0.090           | 177               |
| 2           | 4.06             | 3.9                | 0.049           | 326               |
| 4           | 1.87             | 8.6                | 0.039           | 407               |
| 8           | 1.68             | 9.5                | 0.040           | 398               |

Baseline throughput is flat with batch size and scales only linearly with workers from a
~2 samples/s-per-worker base (saturating at the 4 physical cores) — feeding a GPU that consumes
a bs=32 batch every ~0.3 s would need ~50 workers. Patched, even `num_workers=0` (264 samples/s)
beats the baseline with 8 workers by ~28×, and worker scaling saturates at 4 because the loader
is no longer CPU-bound.

With truly silent noise wavs (the literal "empty wav" case) every item where
`noise_duration > target` exhausts all 100 trials deterministically, then falls back to
`WhiteNoisePerturbation`: 3.5 samples/s at bs=8/nw=4.

## Lhotse variant

### Batch-size scaling at `num_workers=4`

| batch size | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s |
|-----------:|-----------------:|-------------------:|----------------:|------------------:|
| 4          | 0.38             | 10.6               | 0.011           | 362               |
| 8          | 0.48             | 16.6               | 0.020           | 406               |
| 16         | 0.84             | 19.1               | 0.039           | 406               |
| 32         | 0.81             | 39.7               | 0.104           | 308               |
| 64         | 0.73             | 88.0               | 0.209           | 306               |

### Worker scaling at `batch_size=16`

| num_workers | baseline s/batch | baseline samples/s | patched s/batch | patched samples/s |
|------------:|-----------------:|-------------------:|----------------:|------------------:|
| 0           | 3.65             | 4.4                | 0.068           | 237               |
| 1           | 1.54             | 10.4               | 0.090           | 177               |
| 2           | 1.29             | 12.4               | 0.047           | 340               |
| 4           | 0.64             | 25.0               | 0.038           | 418               |
| 8           | 0.67             | 23.9               | 0.040           | 398               |

No-noise contrast: 230–271 samples/s (same as map-style). Silent-noise: 10.0 samples/s at bs=8.

### Why the Lhotse baseline looks faster here — and why it won't be on real data

`LhotseAudioNoiseDataset.__getitem__` calls `AudioSamples` (lhotse `collate_audio`), which
**pads every cut to the batch max length and returns the padded cuts**. The subsequent
`sample_noise(..., cut.num_samples)` therefore uses the *batch-max* length as the noise target
for every item (verified by instrumentation: all noise loads in a batch share the same
`max_dur`). Hotspot #1's retry loop only triggers when `noise_duration > target`, so with this
dummy data (noise 15–20 s vs batch max ≈ 19–20 s) only the longest noise files can trigger it —
and the bigger the batch, the higher the batch max, the rarer the pathology. That is the entire
reason baseline samples/s *rises* with batch size (10.6 → 88).

On realistic data, noise recordings (often minutes long) exceed any utterance batch max, so
**every noise draw enters the retry loop** with the ~50% coin-flip failure mode — the Lhotse
variant then behaves like the map-style one. Its floor is also higher per load (~40–80 ms vs
~20 ms) because it decodes the *full* noise file before trimming, plus the Python `sum()`.
Structurally it also assembles the whole batch inside a single `__getitem__` (workers pipeline
whole batches), which is why patched throughput at bs=32/64 (306–308 samples/s) trails the
map-style variant (358) — coarser parallelism granularity.

The worker-scaling anomaly at `num_workers=1` (177 samples/s in both variants, patched) is the
usual IPC/serialization overhead of a single worker vs in-process loading; it disappears at ≥2.

## GPU-visible stall vs batch size (simulated consumer)

Throughput alone hides what the GPU experiences per step, so `--sim-gpu-samples-per-s RATE`
adds a consumer that sleeps `batch_size/RATE` seconds after each batch (like a training step,
leaving the CPUs to the workers) and records how long each `next(batch)` blocks — the actual
GPU idle time, including tails. Setup: **lhotse, no noise manifest, augmentor on** (a common
NEST configuration), `num_workers=4`.

| GPU rate | bs | baseline stall mean / p95 / max (ms) | baseline GPU util | patched stall mean / p95 / max (ms) | patched GPU util |
|---------:|---:|-------------------------------------:|------------------:|------------------------------------:|-----------------:|
| 200/s    | 8  | 14 / 30 / 75                         | 74%               | 12 / 14 / 18                        | 77%              |
| 200/s    | 16 | 24 / 39 / 157                        | 77%               | 22 / 25 / 33                        | 79%              |
| 200/s    | 32 | 46 / 72 / 413                        | 78%               | 40 / 46 / 104                       | 80%              |
| 200/s    | 64 | 124 / 170 / **2454**                 | 72%               | 72 / 89 / 192                       | 82%              |
| 400/s    | 8  | 21 / 50 / 91                         | 48%               | 13 / 16 / 46                        | 61%              |
| 400/s    | 16 | 39 / 68 / 239                        | 51%               | 22 / 28 / 39                        | 65%              |
| 400/s    | 32 | 83 / 174 / 496                       | 49%               | 41 / 52 / 106                       | 66%              |
| 400/s    | 64 | 156 / 209 / **969**                  | 51%               | 76 / 84 / 252                       | 68%              |

Observations:

1. **Mean stall per step grows ~linearly with batch size** in both versions (~1–2 ms/sample).
   This is main-process work that sits in the GPU's critical path — chiefly receiving the
   batch tensors from the worker over IPC (2–3 float32 `[B, T]` tensors, hundreds of MB at
   bs=64 with 20 s audio). Prefetching cannot hide it because it happens in the consumer
   process. `return_noise=False` (PR 15589) cuts a third of that payload.
2. **The stall *fraction* (GPU utilization) is roughly flat with batch size** — larger batches
   don't reduce average overlap; both the stall and the compute grow together. Gradient
   accumulation with small micro-batches doesn't change the average rates; what it changes is
   the *granularity*.
3. **The tail is where large batches hurt.** Baseline max stall grows superlinearly (75 ms at
   bs=8 → 2.45 s at bs=64 even with a slow GPU): one worker assembles the whole batch, so a
   batch containing straggler items (the bimodal noise path, a slow read) becomes a single
   unhidable multi-second GPU gap, and the fixed-size prefetch queue (num_workers x
   prefetch_factor batches) drains. The patched loader eliminates the stragglers and with them
   the tail (max 192 ms at bs=64). This tail behavior — fine at small batch, multi-second
   freezes at large batch — is exactly the symptom that pushes users toward gradient
   accumulation.

Caveats: 4-core container, sleep-based consumer, no pin_memory/CUDA H2D overlap; absolute
numbers shift on real hardware but the structure (linear IPC term + superlinear straggler
tail) carries over.

## Recommended fixes

1. `ssl_dataset.py` — replace both emptiness checks with a numpy energy test, e.g.
   `np.abs(audio_segment.samples).max() > 0` (and `== 0` for the fallback check). This removes
   the 20 ms/item floor *and* the coin-flip retry storm in both dataset variants, since any
   window with nonzero content passes on the first trial.
2. `augmentation.py:201` — replace `Counter(random.choices(range(num_segments), k=mix_len))`
   with `np.random.multinomial(mix_len, np.ones(n)/n)` (drop zero counts to keep semantics).
3. Optional hardening: skip the offset-retry decode loop when the noise file is known
   non-silent; rate-limit the "empty noise" warning (console I/O per item); in the Lhotse
   variant, consider decoding only the needed noise window instead of the full file.

Raw numbers: `results.jsonl` (map-style) and `results_lhotse.jsonl` (lhotse), one JSON object
per run, produced by `run_sweeps.sh` / `run_sweeps_lhotse.sh`.
