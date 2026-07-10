# SSL AudioNoise dataloader benchmark

Benchmarks the dataloader used by `EncDecDenoiseMaskedTokenPredModel` (NEST) — i.e.
`nemo.collections.asr.data.ssl_dataset.AudioNoiseDataset` with the
`MultiSpeakerNoiseAugmentation` batch augmentor, constructed through the exact same
code path as the model (`get_audio_noise_dataset_from_config`).

## Usage

```bash
# 1. generate dummy data (silent 15-20s main wavs + zero-mean noise wavs @16kHz)
python make_dummy_data.py --out ./data

# 2. single measurement
python bench_dataloader.py \
    --manifest data/train_manifest.json \
    --noise-manifest data/noise_manifest.json \
    --batch-size 16 --num-workers 4

# 3. full sweep matrices (batch-size sweep, worker sweep, no-noise, silent-noise, patched)
./run_sweeps.sh           # map-style AudioNoiseDataset -> results.jsonl
./run_sweeps_lhotse.sh    # LhotseAudioNoiseDataset    -> results_lhotse.jsonl
```

`--patched` applies two candidate fixes via monkeypatching so the headroom can be
measured without modifying NeMo; `--lhotse` benchmarks `LhotseAudioNoiseDataset`
(`use_lhotse: true` path) instead of the map-style dataset.

## Findings

See `RESULTS.md` for full numbers and analysis.
