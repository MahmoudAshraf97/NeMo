#!/bin/bash
# GPU-stall measurement for the user's case: lhotse dataloader, no noise manifest, augmentor on.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
M=data/train_manifest.json
OUT=results_stall.jsonl
: > "$OUT"

run() {
  echo "=== $* ===" >&2
  $PY bench_dataloader.py --manifest $M --lhotse "$@" 2>/dev/null | grep '^RESULT:' | sed 's/^RESULT://' >> "$OUT"
}

for rate in 200 400; do
  for bs in 8 16 32 64; do
    run --batch-size $bs --num-workers 4 --warmup-batches 3 --measure-batches 20 --time-cap 90 --min-time 25 \
        --sim-gpu-samples-per-s $rate --tag stall-baseline
    run --batch-size $bs --num-workers 4 --warmup-batches 3 --measure-batches 20 --time-cap 90 --min-time 25 \
        --sim-gpu-samples-per-s $rate --patched --tag stall-patched
  done
done
echo DONE >&2
