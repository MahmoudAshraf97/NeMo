#!/bin/bash
# Benchmark matrix for the Lhotse variant (LhotseAudioNoiseDataset).
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M=data/train_manifest.json
N=data/noise_manifest.json
SN=data/silent_noise_manifest.json
OUT=results_lhotse.jsonl
[ -f "$M" ] || $PY make_dummy_data.py --out ./data
: > "$OUT"

run() {
  echo "=== $* ===" >&2
  $PY bench_dataloader.py --manifest $M --lhotse "$@" 2>/dev/null | grep '^RESULT:' | sed 's/^RESULT://' >> "$OUT"
}

# 1) baseline: batch-size sweep at num_workers=4
for bs in 4 8 16 32 64; do
  run --noise-manifest $N --batch-size $bs --num-workers 4 --warmup-batches 3 --measure-batches 10 --time-cap 150 --min-time 40 --tag lhotse-bs-sweep
done

# 2) baseline: worker sweep at fixed batch_size=16
for nw in 0 1 2 4 8; do
  run --noise-manifest $N --batch-size 16 --num-workers $nw --warmup-batches 2 --measure-batches 10 --time-cap 150 --min-time 40 --tag lhotse-nw-sweep
done

# 3) contrast: no noise manifest (zeros noise), augmentor on
for bs in 4 16 64; do
  run --batch-size $bs --num-workers 4 --warmup-batches 3 --measure-batches 10 --time-cap 120 --min-time 15 --tag lhotse-no-noise
done

# 4) literal "empty wav" noise manifest
run --noise-manifest $SN --batch-size 8 --num-workers 4 --warmup-batches 1 --measure-batches 5 --time-cap 240 --min-time 40 --tag lhotse-silent-noise

# 5) patched hotspots: batch-size sweep at num_workers=4
for bs in 4 8 16 32 64; do
  run --noise-manifest $N --batch-size $bs --num-workers 4 --warmup-batches 3 --measure-batches 20 --time-cap 120 --min-time 15 --patched --tag lhotse-bs-sweep-patched
done

# 6) patched hotspots: worker sweep at batch_size=16
for nw in 0 1 2 4 8; do
  run --noise-manifest $N --batch-size 16 --num-workers $nw --warmup-batches 2 --measure-batches 20 --time-cap 120 --min-time 15 --patched --tag lhotse-nw-sweep-patched
done

echo DONE >&2
