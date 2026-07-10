"""Generate a dummy dataset of wav files (15-20s @ 16kHz) plus NeMo manifests.

Creates:
  - <out>/wavs/main_XXXX.wav   : silent (all-zero) wavs, durations uniform in [15, 20] s
  - <out>/wavs/noise_XXXX.wav  : low-amplitude zero-mean random noise wavs, same duration range
  - <out>/train_manifest.json  : N_ENTRIES manifest lines cycling over the main wavs
  - <out>/noise_manifest.json  : one line per noise wav
  - <out>/silent_noise_manifest.json : noise manifest pointing at the *silent* main wavs
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-main", type=int, default=256)
    parser.add_argument("--num-noise", type=int, default=64)
    parser.add_argument("--num-entries", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    main_files = []
    for i in range(args.num_main):
        dur = rng.uniform(15.0, 20.0)
        n = int(dur * SR)
        path = os.path.join(wav_dir, f"main_{i:04d}.wav")
        if not os.path.exists(path):
            sf.write(path, np.zeros(n, dtype=np.int16), SR, subtype="PCM_16")
        main_files.append((path, n / SR))

    noise_files = []
    for i in range(args.num_noise):
        dur = rng.uniform(15.0, 20.0)
        n = int(dur * SR)
        path = os.path.join(wav_dir, f"noise_{i:04d}.wav")
        if not os.path.exists(path):
            data = (rng.standard_normal(n) * 300).astype(
                np.int16
            )  # ~-40 dBFS zero-mean noise
            sf.write(path, data, SR, subtype="PCM_16")
        noise_files.append((path, n / SR))

    with open(os.path.join(args.out, "train_manifest.json"), "w") as f:
        for i in range(args.num_entries):
            path, dur = main_files[i % len(main_files)]
            f.write(
                json.dumps({"audio_filepath": path, "duration": dur, "text": ""}) + "\n"
            )

    with open(os.path.join(args.out, "noise_manifest.json"), "w") as f:
        for path, dur in noise_files:
            f.write(
                json.dumps({"audio_filepath": path, "duration": dur, "text": ""}) + "\n"
            )

    with open(os.path.join(args.out, "silent_noise_manifest.json"), "w") as f:
        for path, dur in main_files[: args.num_noise]:
            f.write(
                json.dumps({"audio_filepath": path, "duration": dur, "text": ""}) + "\n"
            )

    print(
        f"wrote {args.num_main} main wavs, {args.num_noise} noise wavs, {args.num_entries} manifest entries"
    )


if __name__ == "__main__":
    main()
