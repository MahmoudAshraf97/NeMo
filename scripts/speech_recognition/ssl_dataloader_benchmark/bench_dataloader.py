"""Benchmark the SSL AudioNoiseDataset dataloader (same construction path as
EncDecDenoiseMaskedTokenPredModel._setup_dataloader_from_config, non-lhotse, non-tarred).

Prints a single JSON result line prefixed with RESULT: for easy scraping.
"""

import argparse
import json
import time

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.data import ssl_dataset
from nemo.utils import logging

logging.setLevel(
    "ERROR"
)  # silence per-item warnings so timing isn't dominated by console I/O

BATCH_AUGMENTOR_CFG = {
    "_target_": "nemo.collections.asr.modules.ssl_modules.MultiSpeakerNoiseAugmentation",
    "prob": 0.5,
    "noise_ratio": 0.5,
    "min_r_speech": -5.0,
    "max_r_speech": 5.0,
    "min_r_noise": -5.0,
    "max_r_noise": 20.0,
    "min_mix_rate": 0.5,
    "max_mix_rate": 0.5,
    "min_num_segments": 1,
    "max_num_segments": 1,
    "min_num_speakers": 1,
    "max_num_speakers": 1,
}


def apply_patches():
    """Monkeypatch the two hotspots to measure fix headroom.

    1. load_noise_audio: replace Python builtin sum() emptiness/retry checks with
       a numpy energy check (fixes bogus 100x retry loop + 20ms/call sum).
    2. MultiSpeakerNoiseAugmentation: replace Counter(random.choices(range(n), k=mix_len))
       (O(mix_len) python loop, mix_len ~ 160k) with an equivalent multinomial draw.
    """
    import numpy as np

    from nemo.collections.asr.parts.preprocessing.perturb import WhiteNoisePerturbation
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    def load_noise_audio_fast(
        sample,
        sample_rate,
        max_audio_len=None,
        pad_to_max=True,
        min_white_noise_db=-90,
        max_white_noise_db=-46,
        max_trial=100,
    ):
        max_dur = None if max_audio_len is None else max_audio_len / sample_rate
        duration = sample.get("duration", None)
        offset = sample.get("offset", 0.0)

        if max_dur is not None and duration is not None and duration > max_dur:
            cnt = 0
            while cnt < max_trial:
                offset = np.random.uniform(0, duration - max_dur)
                audio_segment = AudioSegment.from_file(
                    audio_file=sample["audio_filepath"],
                    offset=offset,
                    duration=max_dur,
                    target_sr=sample_rate,
                )
                if np.abs(audio_segment.samples).max() > 0:
                    break
                cnt += 1
        else:
            audio_segment = AudioSegment.from_file(
                audio_file=sample["audio_filepath"],
                offset=offset,
                duration=duration,
                target_sr=sample_rate,
            )

        if np.abs(audio_segment.samples).max() == 0:
            WhiteNoisePerturbation(
                min_level=min_white_noise_db, max_level=max_white_noise_db
            ).perturb(audio_segment)

        noise = torch.tensor(audio_segment.samples, dtype=torch.float)
        noise_len = torch.tensor(noise.size(0)).long()
        if max_audio_len is not None and pad_to_max:
            if noise.size(0) < max_audio_len:
                noise = torch.nn.functional.pad(
                    noise, (0, max_audio_len - noise.size(0))
                )
            else:
                noise = noise[:max_audio_len]
                noise_len = torch.tensor(max_audio_len).long()
        return noise, noise_len

    ssl_dataset.load_noise_audio = load_noise_audio_fast

    import random as _random
    import types

    from nemo.collections.asr.modules.ssl_modules import augmentation

    class _LazyChoices(list):
        """Marker returned by the patched random.choices; consumed by the patched Counter."""

        def __init__(self, population, k):
            super().__init__()
            self.population = population
            self.k = k

    def patched_choices(population, *a, k=1, **kw):
        return _LazyChoices(population, k)

    def patched_counter(iterable=None):
        if isinstance(iterable, _LazyChoices):
            # same distribution as Counter(random.choices(range(n), k=k)) but O(n) instead of O(k)
            n = len(iterable.population)
            lens = np.random.multinomial(iterable.k, np.ones(n) / n)
            return {i: int(c) for i, c in enumerate(lens) if c > 0}
        from collections import Counter as _C

        return _C(iterable)

    shim = types.SimpleNamespace(
        **{k: getattr(_random, k) for k in dir(_random) if not k.startswith("_")}
    )
    shim.choices = patched_choices
    augmentation.random = shim
    augmentation.Counter = patched_counter


def build_dataloader(args):
    cfg = {
        "manifest_filepath": args.manifest,
        "noise_manifest": args.noise_manifest,
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "max_duration": 60.0,
        "min_duration": 1.0,
        "drop_last": True,
        "is_tarred": False,
        "is_concat": False,
    }
    if not args.no_augmentor:
        cfg["batch_augmentor"] = BATCH_AUGMENTOR_CFG
    cfg = OmegaConf.create(cfg)

    dataset = ssl_dataset.get_audio_noise_dataset_from_config(
        cfg, global_rank=0, world_size=1
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def build_lhotse_dataloader(args):
    """Same construction path as EncDecDenoiseMaskedTokenPredModel with use_lhotse=True."""
    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

    cfg = OmegaConf.create(
        {
            "manifest_filepath": args.manifest,
            "sample_rate": 16000,
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.num_workers,
            "pin_memory": False,
            "max_duration": 60.0,
            "min_duration": 1.0,
            "use_lhotse": True,
        }
    )
    dataset = ssl_dataset.LhotseAudioNoiseDataset(
        noise_manifest=args.noise_manifest,
        batch_augmentor_cfg=(
            None if args.no_augmentor else OmegaConf.create(BATCH_AUGMENTOR_CFG)
        ),
    )
    return get_lhotse_dataloader_from_config(
        cfg, global_rank=0, world_size=1, dataset=dataset
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--noise-manifest", default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--measure-batches", type=int, default=12)
    parser.add_argument(
        "--time-cap", type=float, default=90.0, help="max measurement seconds"
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.0,
        help="keep measuring until this many seconds elapsed, so the steady-state rate "
        "dominates over draining the num_workers*prefetch_factor batch queue",
    )
    parser.add_argument("--no-augmentor", action="store_true")
    parser.add_argument(
        "--patched",
        action="store_true",
        help="apply candidate hotspot fixes before benchmarking",
    )
    parser.add_argument(
        "--lhotse",
        action="store_true",
        help="benchmark LhotseAudioNoiseDataset instead",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.patched:
        apply_patches()

    dl = build_lhotse_dataloader(args) if args.lhotse else build_dataloader(args)

    t_start = time.perf_counter()
    it = iter(dl)
    first = next(it)
    ttfb = time.perf_counter() - t_start
    assert first.audio is not None

    # warmup (first batch already consumed)
    for _ in range(args.warmup_batches - 1):
        next(it)

    n_batches = 0
    n_samples = 0
    audio_secs = 0.0
    t0 = time.perf_counter()
    deadline = t0 + args.time_cap
    min_deadline = t0 + args.min_time
    exhausted = False
    while (
        n_batches < args.measure_batches or time.perf_counter() < min_deadline
    ) and time.perf_counter() < deadline:
        try:
            batch = next(it)
        except StopIteration:
            exhausted = True
            break
        n_batches += 1
        n_samples += batch.audio.size(0)
        audio_secs += batch.audio_len.sum().item() / 16000.0
    elapsed = time.perf_counter() - t0
    del it

    result = {
        "tag": args.tag,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "noise": args.noise_manifest is not None,
        "augmentor": not args.no_augmentor,
        "patched": args.patched,
        "lhotse": args.lhotse,
        "exhausted_epoch": exhausted,
        "time_to_first_batch_s": round(ttfb, 3),
        "measured_batches": n_batches,
        "elapsed_s": round(elapsed, 3),
        "sec_per_batch": round(elapsed / max(n_batches, 1), 4),
        "batches_per_s": round(n_batches / elapsed, 4),
        "samples_per_s": round(n_samples / elapsed, 2),
        "audio_hours_per_s": round(audio_secs / 3600.0 / elapsed, 4),
    }
    print("RESULT:" + json.dumps(result))


if __name__ == "__main__":
    main()
