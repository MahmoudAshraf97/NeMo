#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark full and pruned RNN-T losses from their common pre-joint boundary.

``k2_pruned_reference`` remains an optional correctness/performance reference;
``native_pruned_rnnt`` exercises NeMo's production PyTorch/Triton path.

Every method/profile/batch-size tuple runs in a fresh child process.  The
parent first completes and reports the standard-versus-graph baseline, then
runs the pruned reference adapter and writes the combined report.

Examples::

    # Default synthetic matrix, baseline first and then the pruned reference.
    python scripts/speech_recognition/benchmark_rnnt_losses.py run \
        --output-dir rnnt_benchmark

    # Baselines only.
    python scripts/speech_recognition/benchmark_rnnt_losses.py run \
        --methods warprnnt_numba graph_rnnt --output-dir rnnt_baseline

    # Build 200 representative length batches from text and a tokenizer.
    python scripts/speech_recognition/benchmark_rnnt_losses.py build-length-profile \
        --manifest tarteel.json --tokenizer-model tokenizer.model \
        --batch-size 32 --num-batches 200 --output tarteel_lengths.json

    # Add the generated padding distributions to the synthetic matrix.
    python scripts/speech_recognition/benchmark_rnnt_losses.py run \
        --real-length-profile tarteel_lengths.json --output-dir rnnt_benchmark
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


METHODS = (
    "warprnnt_numba",
    "graph_rnnt",
    "k2_pruned_reference",
    "native_pruned_rnnt",
)
BASELINE_METHODS = METHODS[:2]
SYNTHETIC_PROFILES = {
    "short": (128, 32),
    "target": (400, 128),
    "long": (800, 256),
}
DTYPES = ("bfloat16", "float16", "float32")


@dataclass(frozen=True)
class Job:
    method: str
    profile: str
    batch_size: int
    dtype: str
    vocab_size: int
    encoder_hidden: int
    predictor_hidden: int
    joint_hidden: int
    warmup: int
    iterations: int
    seed: int
    prune_range: int
    simple_loss_scale: float
    fused_batch_size: int | None = None
    real_length_profile: str | None = None
    trace_path: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _optional_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(repo: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_at": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "nemo_commit": _git_value(repo, "rev-parse", "HEAD"),
        "nemo_branch": _git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "nemo_dirty": bool(_git_value(repo, "status", "--porcelain")),
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_PATH",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "NUMBA_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
        "packages": {
            name: _optional_version(name)
            for name in (
                "torch",
                "triton",
                "numba",
                "numba-cuda",
                "nvidia-cuda-nvcc-cu12",
                "k2",
                "nemo_toolkit",
                "nvidia-ml-py",
            )
        },
    }
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info["gpu"] = {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            }
            try:
                info["driver_version"] = torch._C._cuda_getDriverVersion()
            except AttributeError:
                info["driver_version"] = None
            if info["driver_version"] is None:
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=driver_version",
                            "--format=csv,noheader",
                            "--id=0",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    info["driver_version"] = result.stdout.splitlines()[0].strip()
                except (OSError, subprocess.SubprocessError, IndexError):
                    pass
    except ImportError:
        info["cuda_available"] = False
    return info


def _dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return error.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message


def _clear_gradients(module: Any, encoder: Any, predictor: Any) -> None:
    module.zero_grad(set_to_none=True)
    encoder.grad = None
    predictor.grad = None


def _load_length_batches(
    path: str, batch_size: int, limit: int
) -> list[dict[str, list[int]]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    batches = payload.get("batches")
    if not isinstance(batches, list):
        raise ValueError(f"{path} does not contain a 'batches' list")

    selected = []
    for batch in batches:
        input_lengths = batch.get("input_lengths")
        target_lengths = batch.get("target_lengths")
        if not isinstance(input_lengths, list) or not isinstance(target_lengths, list):
            raise ValueError(
                "Every profile batch must contain input_lengths and target_lengths lists"
            )
        if len(input_lengths) < batch_size or len(target_lengths) < batch_size:
            continue
        selected.append(
            {
                "input_lengths": [int(value) for value in input_lengths[:batch_size]],
                "target_lengths": [int(value) for value in target_lengths[:batch_size]],
            }
        )
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError(
            f"No profile batch in {path} contains at least {batch_size} samples"
        )
    if len(selected) < limit:
        raise ValueError(
            f"Only {len(selected)} profile batches in {path} contain at least {batch_size} samples; "
            f"{limit} are required"
        )
    return selected


def _make_length_specs(job: Job) -> list[dict[str, list[int]]]:
    if job.real_length_profile:
        return _load_length_batches(
            job.real_length_profile, job.batch_size, job.iterations
        )
    frames, labels = SYNTHETIC_PROFILES[job.profile]
    return [
        {
            "input_lengths": [frames] * job.batch_size,
            "target_lengths": [labels] * job.batch_size,
        }
    ] * job.iterations


def _compilation_warmup_specs(
    specs: list[dict[str, list[int]]],
) -> list[dict[str, list[int]]]:
    selected = []
    seen = set()
    for spec in specs:
        signature = (max(spec["input_lengths"]), max(spec["target_lengths"]))
        if signature not in seen:
            seen.add(signature)
            selected.append(spec)
    return selected


def _make_inputs(
    torch: Any, job: Job, spec: dict[str, list[int]], device: Any, dtype: Any
) -> tuple[Any, ...]:
    input_lengths = torch.tensor(
        spec["input_lengths"], device=device, dtype=torch.int64
    )
    target_lengths = torch.tensor(
        spec["target_lengths"], device=device, dtype=torch.int64
    )
    max_t = int(input_lengths.max().item())
    max_u = int(target_lengths.max().item())

    encoder = torch.randn(
        job.batch_size,
        job.encoder_hidden,
        max_t,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    predictor = torch.randn(
        job.batch_size,
        job.predictor_hidden,
        max_u + 1,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    targets = torch.randint(
        0,
        job.vocab_size - 1,
        (job.batch_size, max_u),
        device=device,
        dtype=torch.int64,
    )
    return encoder, predictor, targets, input_lengths, target_lengths


def _build_pipeline(torch: Any, job: Job, dtype: Any, device: Any) -> Any:
    from nemo.collections.asr.losses.rnnt import RNNTLoss, resolve_rnnt_loss
    from nemo.collections.asr.modules.rnnt import RNNTJoint

    class BenchmarkPipeline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.method = job.method
            self.annotate_profile = False
            self.blank = job.vocab_size - 1
            self.joint = RNNTJoint(
                jointnet={
                    "encoder_hidden": job.encoder_hidden,
                    "pred_hidden": job.predictor_hidden,
                    "joint_hidden": job.joint_hidden,
                    "activation": "relu",
                    "dropout": 0.0,
                },
                num_classes=job.vocab_size - 1,
                log_softmax=False,
                preserve_memory=False,
                fuse_loss_wer=self.method == "native_pruned_rnnt",
                fused_batch_size=(
                    job.batch_size if self.method == "native_pruned_rnnt" else None
                ),
                masking_prob=-1.0,
            )
            if self.method == "warprnnt_numba":
                self.loss = RNNTLoss(
                    num_classes=self.blank,
                    reduction="mean_batch",
                    loss_name="warprnnt_numba",
                )
                # The Numba kernels support fp16 or fp32, but not bf16.  The regular
                # wrapper only distinguishes fp16 support, so force its established
                # fp32 path for a bf16 joint tensor.
                if dtype == torch.bfloat16:
                    self.loss._force_float32 = True
            elif self.method == "graph_rnnt":
                self.loss = resolve_rnnt_loss(
                    "graph_rnnt",
                    blank_idx=self.blank,
                    loss_kwargs={
                        "use_grid_implementation": True,
                        "use_triton": True,
                        "cast_to_float32": False,
                    },
                )
                if not self.loss.use_triton:
                    raise RuntimeError(
                        "graph_rnnt requested Triton, but NeMo reports Triton is unavailable"
                    )
            elif self.method == "native_pruned_rnnt":
                from nemo.collections.asr.parts.pruned_rnnt import (
                    _gather_predictor,
                    _pruned_logprobs_triton,
                    get_prune_ranges,
                    get_smoothed_rnnt_logprobs,
                    rnnt_loss_triton,
                )

                self.loss = RNNTLoss(
                    num_classes=self.blank,
                    reduction="mean_batch",
                    loss_name="pruned_rnnt",
                    loss_kwargs={
                        "prune_range": job.prune_range,
                        "simple_loss_scale": job.simple_loss_scale,
                        "warmup_steps": 0,
                    },
                )
                self.loss.bind_joint(self.joint)
                self.joint.set_loss(self.loss)
                # WER is disabled in timed jobs, but RNNTJoint's fused return
                # contract requires a bound metric object.
                self.joint.set_wer(object())
                self.native_get_prune_ranges = get_prune_ranges
                self.native_get_smoothed_logprobs = get_smoothed_rnnt_logprobs
                self.native_gather_predictor = _gather_predictor
                self.native_pruned_logprobs = _pruned_logprobs_triton
                self.native_rnnt_loss = rnnt_loss_triton
            else:
                try:
                    import k2
                except ImportError as error:
                    raise ImportError(
                        "k2_pruned_reference requires a k2 build with pruned RNN-T operations"
                    ) from error
                required = (
                    "rnnt_loss_smoothed",
                    "get_rnnt_prune_ranges",
                    "do_rnnt_pruning",
                    "rnnt_loss_pruned",
                )
                missing = [name for name in required if not hasattr(k2, name)]
                if missing:
                    raise RuntimeError(
                        f"Installed k2 is missing required pruned RNN-T APIs: {missing}"
                    )
                self.k2 = k2
                self.simple_am = torch.nn.Linear(job.encoder_hidden, job.vocab_size)
                self.simple_lm = torch.nn.Linear(job.predictor_hidden, job.vocab_size)

        def profile_range(self, name: str) -> Any:
            if self.annotate_profile:
                return torch.profiler.record_function(f"rnnt_benchmark::{name}")
            return nullcontext()

        def full_baseline_forward(
            self,
            encoder: Any,
            predictor: Any,
            targets: Any,
            input_lengths: Any,
            target_lengths: Any,
        ) -> Any:
            with self.profile_range("full_joint"):
                logits = self.joint(encoder_outputs=encoder, decoder_outputs=predictor)
            if self.method == "warprnnt_numba":
                with self.profile_range("dynamic_programming"):
                    return self.loss(
                        log_probs=logits,
                        targets=targets,
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                    )
            if self.annotate_profile:
                original_get_graphs = self.loss.get_graphs_batched

                def profiled_get_graphs(*args: Any, **kwargs: Any) -> Any:
                    with self.profile_range("graph_construction"):
                        return original_get_graphs(*args, **kwargs)

                self.loss.get_graphs_batched = profiled_get_graphs
            try:
                with self.profile_range("graph_and_logprob_extraction"):
                    target_fsas = self.loss.get_weighted_graphs(
                        logits=logits,
                        targets=targets.long(),
                        source_lengths=input_lengths.long(),
                        target_lengths=target_lengths.long(),
                        use_graph_weight=False,
                    )
            finally:
                if self.annotate_profile:
                    self.loss.get_graphs_batched = original_get_graphs
            with self.profile_range("dynamic_programming"):
                scores = -target_fsas.get_tot_scores(
                    use_double_scores=self.loss.double_scores, log_semiring=True
                )
            return scores.mean()

        def profiled_native_forward(
            self,
            encoder: Any,
            predictor: Any,
            targets: Any,
            input_lengths: Any,
            target_lengths: Any,
        ) -> Any:
            inner = self.loss._loss
            encoder = encoder.transpose(1, 2)
            predictor = predictor.transpose(1, 2)
            with self.profile_range("simple_projection_and_normalization"):
                simple_am = inner.simple_encoder(encoder)
                simple_lm = inner.simple_predictor(predictor)
                simple_target, simple_blank = self.native_get_smoothed_logprobs(
                    simple_lm,
                    simple_am,
                    targets,
                    inner.blank,
                    inner.lm_only_scale,
                    inner.am_only_scale,
                )
            with self.profile_range("simple_dynamic_programming"):
                simple_loss, target_occupation, blank_occupation = (
                    self.native_rnnt_loss(
                        simple_target, simple_blank, input_lengths, target_lengths
                    )
                )
            with self.profile_range("pruning_range"):
                ranges = self.native_get_prune_ranges(
                    target_occupation,
                    blank_occupation,
                    input_lengths,
                    target_lengths,
                    inner.prune_range,
                )
            with self.profile_range("joint_projection_and_gather"):
                projected_encoder = self.joint.project_encoder(encoder)
                projected_predictor = self.joint.project_prednet(predictor)
                gathered_predictor = self.native_gather_predictor(
                    projected_predictor, ranges
                )
            with self.profile_range("reduced_joint"):
                joint_input = projected_encoder.unsqueeze(2) + gathered_predictor
                if self.joint.is_adapter_available():
                    joint_input = self.joint.forward_enabled_adapters(joint_input)
                logits = self.joint.joint_net(joint_input)
                logit_scale = (
                    1.0 / self.joint.temperature if self.joint.log_softmax else 1.0
                )
            with self.profile_range("pruned_logprob_extraction"):
                pruned_target, pruned_blank = self.native_pruned_logprobs(
                    logits,
                    ranges,
                    targets,
                    inner.blank,
                    input_lengths,
                    target_lengths,
                    logit_scale,
                )
            with self.profile_range("pruned_dynamic_programming"):
                pruned_loss, _, _ = self.native_rnnt_loss(
                    pruned_target,
                    pruned_blank,
                    input_lengths,
                    target_lengths,
                )
            return (inner.simple_loss_scale * simple_loss + pruned_loss).mean()

        def forward(
            self,
            encoder: Any,
            predictor: Any,
            targets: Any,
            input_lengths: Any,
            target_lengths: Any,
        ) -> Any:
            if self.method in BASELINE_METHODS:
                fused_batch_size = job.fused_batch_size
                if not fused_batch_size or fused_batch_size >= encoder.shape[0]:
                    return self.full_baseline_forward(
                        encoder, predictor, targets, input_lengths, target_lengths
                    )
                weighted_losses = []
                batch = encoder.shape[0]
                for begin in range(0, batch, fused_batch_size):
                    end = min(begin + fused_batch_size, batch)
                    sub_source_lengths = input_lengths[begin:end]
                    sub_target_lengths = target_lengths[begin:end]
                    max_source = int(sub_source_lengths.max())
                    max_target = int(sub_target_lengths.max())
                    sub_loss = self.full_baseline_forward(
                        encoder[begin:end, :, :max_source],
                        predictor[begin:end, :, : max_target + 1],
                        targets[begin:end, :max_target],
                        sub_source_lengths,
                        sub_target_lengths,
                    )
                    weighted_losses.append(sub_loss * (end - begin))
                return torch.stack(weighted_losses).sum() / batch

            if self.method == "native_pruned_rnnt":
                if self.annotate_profile:
                    return self.profiled_native_forward(
                        encoder, predictor, targets, input_lengths, target_lengths
                    )
                loss, _, _, _ = self.joint(
                    encoder_outputs=encoder,
                    decoder_outputs=predictor,
                    encoder_lengths=input_lengths,
                    transcripts=targets,
                    transcript_lengths=target_lengths,
                    compute_wer=False,
                )
                return loss

            # k2 boundary is [B, 4]: [start_u, start_t, end_u, end_t].
            boundary = torch.zeros(
                (encoder.shape[0], 4), dtype=torch.int64, device=encoder.device
            )
            boundary[:, 2] = target_lengths
            boundary[:, 3] = input_lengths

            with self.profile_range("simple_joiner_and_alignment"):
                encoder_t = encoder.transpose(1, 2)
                predictor_t = predictor.transpose(1, 2)
                simple_am = self.simple_am(encoder_t)
                simple_lm = self.simple_lm(predictor_t)
                simple_loss, (px_grad, py_grad) = self.k2.rnnt_loss_smoothed(
                    lm=simple_lm.float(),
                    am=simple_am.float(),
                    symbols=targets,
                    termination_symbol=self.blank,
                    lm_only_scale=0.25,
                    am_only_scale=0.0,
                    boundary=boundary,
                    reduction="none",
                    return_grad=True,
                )
            with self.profile_range("pruning_range"):
                ranges = self.k2.get_rnnt_prune_ranges(
                    px_grad=px_grad,
                    py_grad=py_grad,
                    boundary=boundary,
                    s_range=job.prune_range,
                )

            with self.profile_range("joint_projection_gemms"):
                projected_am = self.joint.project_encoder(encoder_t)
                projected_lm = self.joint.project_prednet(predictor_t)
            with self.profile_range("pruning_gather"):
                am_pruned, lm_pruned = self.k2.do_rnnt_pruning(
                    am=projected_am, lm=projected_lm, ranges=ranges
                )
            with self.profile_range("reduced_joint"):
                pruned_logits = self.joint.joint_net(am_pruned + lm_pruned)
            with self.profile_range("dynamic_programming"):
                pruned_loss = self.k2.rnnt_loss_pruned(
                    logits=pruned_logits.float(),
                    symbols=targets,
                    ranges=ranges,
                    termination_symbol=self.blank,
                    boundary=boundary,
                    reduction="none",
                )
            return (job.simple_loss_scale * simple_loss + pruned_loss).mean()

    return BenchmarkPipeline().to(device=device, dtype=dtype).train()


def _check_grad(name: str, value: Any) -> None:
    if value is None:
        raise RuntimeError(f"Sanity check failed: {name} gradient is missing")
    if not value.isfinite().all().item():
        raise RuntimeError(
            f"Sanity check failed: {name} gradient contains non-finite values"
        )
    if not value.ne(0).any().item():
        raise RuntimeError(f"Sanity check failed: {name} gradient is all zero")


def _sanity_check(torch: Any, pipeline: Any, inputs: tuple[Any, ...]) -> float:
    _clear_gradients(pipeline, inputs[0], inputs[1])
    loss = pipeline(*inputs)
    if loss.ndim != 0 or not loss.isfinite().item():
        raise RuntimeError(
            f"Sanity check failed: expected finite scalar loss, got {loss}"
        )
    loss.backward()
    _check_grad("encoder input", inputs[0].grad)
    _check_grad("predictor input", inputs[1].grad)
    parameter_grad = next(
        (
            parameter.grad
            for parameter in pipeline.parameters()
            if parameter.grad is not None
        ),
        None,
    )
    _check_grad("joint parameter", parameter_grad)
    return float(loss.detach().float().item())


def _profile_iteration(
    torch: Any, pipeline: Any, inputs: tuple[Any, ...], path: str
) -> tuple[str, list[dict[str, Any]]]:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_gradients(pipeline, inputs[0], inputs[1])
    pipeline.annotate_profile = True
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            pipeline(*inputs).backward()
    finally:
        pipeline.annotate_profile = False
    torch.cuda.synchronize()
    profile.export_chrome_trace(str(trace_path))
    summary_path = trace_path.with_suffix(".summary.json")
    ranges = []
    for event in profile.key_averages():
        if not event.key.startswith("rnnt_benchmark::"):
            continue
        device_total = getattr(event, "self_device_time_total", None)
        if device_total is None:
            device_total = getattr(event, "self_cuda_time_total", None)
        inclusive_device_total = getattr(event, "device_time_total", None)
        if inclusive_device_total is None:
            inclusive_device_total = getattr(event, "cuda_time_total", None)
        ranges.append(
            {
                "name": event.key.removeprefix("rnnt_benchmark::"),
                "calls": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "self_cpu_time_total_us": event.self_cpu_time_total,
                "device_time_total_us": inclusive_device_total,
                "self_device_time_total_us": device_total,
            }
        )
    _write_json(summary_path, {"trace": str(trace_path), "ranges": ranges})
    return str(summary_path), ranges


def _run_job(job: Job) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": job.method,
        "profile": job.profile,
        "batch_size": job.batch_size,
        "fused_batch_size": job.fused_batch_size,
        "dtype": job.dtype,
        "status": "error",
        "started_at": _utc_now(),
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this process")
        if job.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support bfloat16")

        torch.manual_seed(job.seed)
        torch.cuda.manual_seed_all(job.seed)
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = _dtype_from_name(torch, job.dtype)
        specs = _make_length_specs(job)
        pipeline = _build_pipeline(torch, job, dtype, device)
        # The pruned adapter owns two additional simple-joiner projections. Reset
        # the RNG after module construction so all methods still receive exactly
        # the same synthetic inputs and targets.
        torch.manual_seed(job.seed)
        torch.cuda.manual_seed_all(job.seed)
        first_inputs = _make_inputs(torch, job, specs[0], device, dtype)
        joint_parameter_sum = sum(
            parameter.detach().float().sum().item()
            for parameter in pipeline.joint.parameters()
        )
        common_input_sum = (
            first_inputs[0].detach().float().sum().item()
            + first_inputs[1].detach().float().sum().item()
            + first_inputs[2].detach().sum().item()
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        common_allocated = torch.cuda.memory_allocated(device)
        common_reserved = torch.cuda.memory_reserved(device)
        sanity_loss = _sanity_check(torch, pipeline, first_inputs)

        measured_specs = specs[: job.iterations]
        compilation_specs = _compilation_warmup_specs(measured_specs)
        for spec in compilation_specs:
            compilation_inputs = _make_inputs(torch, job, spec, device, dtype)
            _clear_gradients(pipeline, compilation_inputs[0], compilation_inputs[1])
            pipeline(*compilation_inputs).backward()
            del compilation_inputs

        # Warm clocks and allocators after all target-axis Triton variants have
        # compiled so no compilation can enter aggregate throughput.
        for warmup_idx in range(job.warmup):
            if warmup_idx == 0:
                warmup_inputs = first_inputs
            else:
                warmup_inputs = _make_inputs(
                    torch, job, specs[warmup_idx % len(specs)], device, dtype
                )
            _clear_gradients(pipeline, warmup_inputs[0], warmup_inputs[1])
            pipeline(*warmup_inputs).backward()
            if warmup_inputs is not first_inputs:
                del warmup_inputs
        torch.cuda.synchronize()

        _clear_gradients(pipeline, first_inputs[0], first_inputs[1])
        del first_inputs
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        host_ms: list[float] = []
        gpu_ms: list[float] = []
        peak_allocated: list[int] = []
        peak_reserved: list[int] = []
        incremental_allocated: list[int] = []
        incremental_reserved: list[int] = []
        total_samples = 0
        total_states = 0
        total_valid_states = 0
        for spec in measured_specs:
            inputs = _make_inputs(torch, job, spec, device, dtype)
            iteration_common_allocated = torch.cuda.memory_allocated(device)
            iteration_common_reserved = torch.cuda.memory_reserved(device)
            common_allocated = max(common_allocated, iteration_common_allocated)
            common_reserved = max(common_reserved, iteration_common_reserved)
            _clear_gradients(pipeline, inputs[0], inputs[1])
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            host_start = time.perf_counter()
            start_event.record()
            pipeline(*inputs).backward()
            end_event.record()
            torch.cuda.synchronize()
            host_ms.append((time.perf_counter() - host_start) * 1000.0)
            gpu_ms.append(float(start_event.elapsed_time(end_event)))
            iteration_peak_allocated = torch.cuda.max_memory_allocated(device)
            iteration_peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated.append(iteration_peak_allocated)
            peak_reserved.append(iteration_peak_reserved)
            incremental_allocated.append(
                max(0, iteration_peak_allocated - iteration_common_allocated)
            )
            incremental_reserved.append(
                max(0, iteration_peak_reserved - iteration_common_reserved)
            )
            total_samples += job.batch_size
            total_states += (
                job.batch_size
                * max(spec["input_lengths"])
                * (max(spec["target_lengths"]) + 1)
            )
            total_valid_states += sum(
                input_length * (target_length + 1)
                for input_length, target_length in zip(
                    spec["input_lengths"], spec["target_lengths"]
                )
            )
            del inputs

        profile_summary_path = None
        profile_ranges = None
        profile_error = None
        if job.trace_path:
            try:
                trace_inputs = _make_inputs(torch, job, specs[0], device, dtype)
                profile_summary_path, profile_ranges = _profile_iteration(
                    torch, pipeline, trace_inputs, job.trace_path
                )
                del trace_inputs
            except Exception as error:
                profile_error = f"{type(error).__name__}: {error}"
                if "trace_inputs" in locals():
                    del trace_inputs
                pipeline.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        total_host_seconds = sum(host_ms) / 1000.0
        maximum_allocated = max(peak_allocated)
        maximum_reserved = max(peak_reserved)
        row.update(
            {
                "status": "ok",
                "iterations_measured": len(host_ms),
                "compilation_signatures_warmed": len(compilation_specs),
                "sanity_loss": sanity_loss,
                "joint_parameter_sum": joint_parameter_sum,
                "common_input_sum": common_input_sum,
                "host_p50_ms": _percentile(host_ms, 0.50),
                "host_p95_ms": _percentile(host_ms, 0.95),
                "host_mean_ms": statistics.fmean(host_ms),
                "gpu_p50_ms": _percentile(gpu_ms, 0.50),
                "gpu_p95_ms": _percentile(gpu_ms, 0.95),
                "gpu_mean_ms": statistics.fmean(gpu_ms),
                "samples_per_second": total_samples / total_host_seconds,
                "states_per_second": total_states / total_host_seconds,
                "valid_states_per_second": total_valid_states / total_host_seconds,
                "common_allocated_bytes": common_allocated,
                "common_reserved_bytes": common_reserved,
                "peak_allocated_bytes": maximum_allocated,
                "peak_reserved_bytes": maximum_reserved,
                "incremental_allocated_bytes": max(incremental_allocated),
                "incremental_reserved_bytes": max(incremental_reserved),
                "host_samples_ms": host_ms,
                "gpu_samples_ms": gpu_ms,
                "trace_path": job.trace_path,
                "profile_summary_path": profile_summary_path,
                "profile_ranges": profile_ranges,
                "profile_error": profile_error,
            }
        )
    except BaseException as error:
        row.update(
            {
                "status": "oom" if _is_oom(error) else "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    row["finished_at"] = _utc_now()
    return row


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _worker_command(script: Path, job_path: Path, result_path: Path) -> list[str]:
    return [
        sys.executable,
        str(script),
        "_worker",
        "--job",
        str(job_path),
        "--result",
        str(result_path),
    ]


def _execute_isolated(script: Path, job: Job, jobs_dir: Path) -> dict[str, Any]:
    fused_suffix = f"__f{job.fused_batch_size}" if job.fused_batch_size else "__full"
    stem = f"{job.profile}__b{job.batch_size}__{job.method}{fused_suffix}"
    job_path = jobs_dir / f"{stem}.job.json"
    result_path = jobs_dir / f"{stem}.result.json"
    stdout_path = jobs_dir / f"{stem}.stdout.log"
    result_path.unlink(missing_ok=True)
    if job.trace_path:
        trace_path = Path(job.trace_path)
        trace_path.unlink(missing_ok=True)
        trace_path.with_suffix(".summary.json").unlink(missing_ok=True)
    _write_json(job_path, asdict(job))
    print(
        f"[{_utc_now()}] {job.profile} B={job.batch_size} {job.method} fused={job.fused_batch_size or 'full'}",
        flush=True,
    )
    with stdout_path.open("w", encoding="utf-8") as output:
        process = subprocess.run(
            _worker_command(script, job_path, result_path),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result_path.exists():
        with result_path.open(encoding="utf-8") as stream:
            row = json.load(stream)
    else:
        child_output = stdout_path.read_text(encoding="utf-8", errors="replace")
        child_oom = "out of memory" in child_output.lower()
        row = {
            "method": job.method,
            "profile": job.profile,
            "batch_size": job.batch_size,
            "fused_batch_size": job.fused_batch_size,
            "dtype": job.dtype,
            "status": "oom" if child_oom else "error",
            "error_type": "ChildProcessError",
            "error": f"Child exited with code {process.returncode} without writing a result",
        }
    row["child_exit_code"] = process.returncode
    row["child_log"] = str(stdout_path)
    print(f"  -> {row['status']}", flush=True)
    return row


def _baseline_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    fastest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("method") not in BASELINE_METHODS:
            continue
        key = (row["profile"], int(row["batch_size"]))
        if key not in fastest or row["host_p50_ms"] < fastest[key]["host_p50_ms"]:
            fastest[key] = row
    return fastest


def _add_comparisons(rows: list[dict[str, Any]]) -> None:
    baselines = _baseline_rows(rows)
    for row in rows:
        baseline = baselines.get((row["profile"], int(row["batch_size"])))
        if row.get("status") != "ok" or baseline is None:
            row["reference_method"] = None
            row["speedup_vs_reference"] = None
            row["incremental_memory_ratio_vs_reference"] = None
            continue
        row["reference_method"] = baseline["method"]
        row["speedup_vs_reference"] = baseline["host_p50_ms"] / row["host_p50_ms"]
        reference_memory = baseline["incremental_allocated_bytes"]
        row["incremental_memory_ratio_vs_reference"] = (
            row["incremental_allocated_bytes"] / reference_memory
            if reference_memory
            else None
        )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _markdown(rows: Sequence[dict[str, Any]], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Reference is the faster successful baseline (`warprnnt_numba` or `graph_rnnt`) for the same workload.",
        "No automatic go/no-go threshold is applied.",
        "States/s counts padded joint states; valid-only states/s remains available in CSV/JSON.",
        "",
        "| Profile | B | Fused B | Method | Status | Host p50 ms | Host p95 ms | GPU p50 ms | samples/s | states/s | Peak GiB | Incremental GiB | Speedup | Memory ratio |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") == "ok":
            lines.append(
                "| {profile} | {batch_size} | {fused} | {method} | ok | {host50} | {host95} | {gpu50} | {samples} | "
                "{states} | {peak} | {incremental} | {speedup} | {memory_ratio} |".format(
                    **row,
                    fused=row.get("fused_batch_size") or "full",
                    host50=_fmt(row.get("host_p50_ms")),
                    host95=_fmt(row.get("host_p95_ms")),
                    gpu50=_fmt(row.get("gpu_p50_ms")),
                    samples=_fmt(row.get("samples_per_second")),
                    states=_fmt(row.get("states_per_second"), 0),
                    peak=_fmt(row.get("peak_allocated_bytes", 0) / (1024**3), 3),
                    incremental=_fmt(
                        row.get("incremental_allocated_bytes", 0) / (1024**3), 3
                    ),
                    speedup=_fmt(row.get("speedup_vs_reference"), 3),
                    memory_ratio=_fmt(
                        row.get("incremental_memory_ratio_vs_reference"), 3
                    ),
                )
            )
        else:
            error = (
                str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")[:160]
            )
            lines.append(
                f"| {row['profile']} | {row['batch_size']} | {row.get('fused_batch_size') or 'full'} | "
                f"{row['method']} | {row['status']}: {error} | - | - | - | - | - | - | - | - | - |"
            )

    lines.extend(
        [
            "",
            "## Largest successful batch",
            "",
            "| Profile | Method | Largest B |",
            "|---|---|---:|",
        ]
    )
    keys = sorted({(row["profile"], row["method"]) for row in rows})
    for profile, method in keys:
        successful = [
            int(row["batch_size"])
            for row in rows
            if row["profile"] == profile
            and row["method"] == method
            and row.get("status") == "ok"
        ]
        lines.append(
            f"| {profile} | {method} | {max(successful) if successful else '-'} |"
        )

    profiled = [row for row in rows if row.get("profile_ranges")]
    if profiled:
        lines.extend(
            [
                "",
                "## Target profiler attribution",
                "",
                "Times are inclusive per named range; nested graph ranges therefore overlap.",
                "",
                "| Method | B | Range | CPU total ms | Device total ms |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in profiled:
            for event in row["profile_ranges"]:
                cpu_ms = event.get("cpu_time_total_us")
                device_ms = event.get("device_time_total_us")
                lines.append(
                    f"| {row['method']} | {row['batch_size']} | {event['name']} | "
                    f"{_fmt(cpu_ms / 1000.0 if cpu_ms is not None else None, 3)} | "
                    f"{_fmt(device_ms / 1000.0 if device_ms is not None else None, 3)} |"
                )
    return "\n".join(lines) + "\n"


def _write_report(
    output_dir: Path, name: str, rows: list[dict[str, Any]], environment: dict[str, Any]
) -> str:
    _add_comparisons(rows)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "environment": environment,
        "results": rows,
    }
    _write_json(output_dir / f"{name}.json", payload)
    _write_csv(output_dir / f"{name}.csv", rows)
    markdown = _markdown(rows, name.replace("_", " ").title())
    (output_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    return markdown


def _profiles_for_run(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    profiles = [(profile, None) for profile in args.synthetic_profiles]
    if args.real_length_profile:
        profiles.append(
            (args.real_profile_name, str(Path(args.real_length_profile).resolve()))
        )
    return profiles


def _make_jobs(args: argparse.Namespace, methods: Iterable[str]) -> list[Job]:
    jobs = []
    for profile, real_path in _profiles_for_run(args):
        iterations = args.real_iterations if real_path else args.iterations
        for batch_size in args.batch_sizes:
            for method in methods:
                fused_sizes: list[int | None] = [None]
                if method in BASELINE_METHODS:
                    fused_sizes.extend(
                        size for size in args.fused_batch_sizes if size <= batch_size
                    )
                for fused_batch_size in fused_sizes:
                    trace_path = None
                    if (
                        args.profile_traces
                        and fused_batch_size is None
                        and profile == "target"
                        and batch_size == args.profile_batch_size
                    ):
                        trace_path = str(
                            (
                                Path(args.output_dir)
                                / "traces"
                                / f"{method}__target__b{batch_size}.json"
                            ).resolve()
                        )
                    jobs.append(
                        Job(
                            method=method,
                            profile=profile,
                            batch_size=batch_size,
                            dtype=args.dtype,
                            vocab_size=args.vocab_size,
                            encoder_hidden=args.encoder_hidden,
                            predictor_hidden=args.predictor_hidden,
                            joint_hidden=args.joint_hidden,
                            warmup=args.warmup,
                            iterations=iterations,
                            seed=args.seed,
                            prune_range=args.prune_range,
                            simple_loss_scale=args.simple_loss_scale,
                            fused_batch_size=fused_batch_size,
                            real_length_profile=real_path,
                            trace_path=trace_path,
                        )
                    )
    return jobs


def _run_command(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    output_dir = Path(args.output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment(repo)
    _write_json(output_dir / "environment.json", environment)
    configuration = {
        key: value for key, value in vars(args).items() if key != "handler"
    }
    if args.real_length_profile:
        source_profile = Path(args.real_length_profile).resolve()
        snapshot_profile = output_dir / f"{args.real_profile_name}_rnnt_lengths.json"
        if source_profile != snapshot_profile:
            shutil.copy2(source_profile, snapshot_profile)
        configuration["real_length_profile_snapshot"] = str(snapshot_profile)
    _write_json(
        output_dir / "configuration.json",
        configuration,
    )

    requested = list(dict.fromkeys(args.methods))
    baseline_methods = [method for method in BASELINE_METHODS if method in requested]
    pruned_methods = [
        method
        for method in requested
        if method in {"k2_pruned_reference", "native_pruned_rnnt"}
    ]
    rows: list[dict[str, Any]] = []

    if baseline_methods:
        print("Running isolated standard/graph baseline jobs first.", flush=True)
        for job in _make_jobs(args, baseline_methods):
            rows.append(_execute_isolated(script, job, jobs_dir))
        baseline_markdown = _write_report(
            output_dir, "baseline_results", rows, environment
        )
        print("\n" + baseline_markdown, flush=True)

    if pruned_methods:
        print("Running isolated pruned RNN-T jobs.", flush=True)
        for job in _make_jobs(args, pruned_methods):
            rows.append(_execute_isolated(script, job, jobs_dir))

    final_markdown = _write_report(
        output_dir, "rnnt_benchmark_results", rows, environment
    )
    print("\n" + final_markdown, flush=True)
    print(f"Artifacts written to {output_dir}", flush=True)
    errors = sum(row.get("status") == "error" for row in rows)
    if errors:
        print(
            f"Benchmark completed with {errors} non-OOM job error(s).", file=sys.stderr
        )
        return 1
    return 0


def _worker_main(args: argparse.Namespace) -> int:
    with open(args.job, encoding="utf-8") as stream:
        job = Job(**json.load(stream))
    result = _run_job(job)
    _write_json(Path(args.result), result)
    return 0 if result["status"] in {"ok", "oom"} else 1


def _length_summary(values: Sequence[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _build_length_profile(args: argparse.Namespace) -> int:
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError(
            "Profile generation requires the sentencepiece package"
        ) from error

    if not 0.0 < args.tail_fraction < 1.0:
        raise ValueError("--tail-fraction must be strictly between zero and one")
    if args.main_min_frames > args.main_max_frames:
        raise ValueError("--main-min-frames cannot exceed --main-max-frames")
    if args.tail_max_frames <= args.main_max_frames:
        raise ValueError("--tail-max-frames must exceed --main-max-frames")

    manifest_path = Path(args.manifest).resolve()
    tokenizer_path = Path(args.tokenizer_model).resolve()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.vocab_size() != args.expected_vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer.vocab_size()} pieces; expected {args.expected_vocab_size}"
        )

    target_lengths = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            text = record.get(args.text_field)
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"Manifest line {line_number} has no non-empty {args.text_field!r} field"
                )
            target_lengths.append(len(tokenizer.encode(text, out_type=int)))

    sample_count = args.batch_size * args.num_batches
    if len(target_lengths) < sample_count:
        raise ValueError(
            f"Manifest contains {len(target_lengths)} texts, but {sample_count} are required for sampling without replacement"
        )

    rng = random.Random(args.seed)
    selected_target_lengths = rng.sample(target_lengths, sample_count)
    tail_per_column = round(args.num_batches * args.tail_fraction)
    main_per_column = args.num_batches - tail_per_column
    columns = []
    for _ in range(args.batch_size):
        column = [
            rng.randint(args.main_min_frames, args.main_max_frames)
            for _ in range(main_per_column)
        ]
        column.extend(
            rng.randint(args.main_max_frames + 1, args.tail_max_frames)
            for _ in range(tail_per_column)
        )
        rng.shuffle(column)
        columns.append(column)
    input_lengths = [
        columns[column][row]
        for row in range(args.num_batches)
        for column in range(args.batch_size)
    ]

    batches = []
    for start in range(0, sample_count, args.batch_size):
        end = start + args.batch_size
        batches.append(
            {
                "input_lengths": input_lengths[start:end],
                "target_lengths": selected_target_lengths[start:end],
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "manifest": str(manifest_path),
        "tokenizer_model": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "tokenizer_vocab_size": tokenizer.vocab_size(),
        "text_field": args.text_field,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_batches": len(batches),
        "length_boundary": "synthetic encoder output and tokenized manifest text",
        "encoder_length_generation": {
            "main_fraction": 1.0 - args.tail_fraction,
            "main_range": [args.main_min_frames, args.main_max_frames],
            "tail_fraction": args.tail_fraction,
            "tail_range": [args.main_max_frames + 1, args.tail_max_frames],
            "sampling": "discrete uniform within each range; rounded tail count per batch column",
        },
        "manifest_target_length_summary": _length_summary(target_lengths),
        "selected_target_length_summary": _length_summary(selected_target_lengths),
        "selected_encoder_length_summary": _length_summary(input_lengths),
        "batches": batches,
    }
    output_path = Path(args.output).resolve()
    _write_json(output_path, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "batches"}, indent=2
        )
    )
    print(f"Wrote {len(batches)} length-only batches to {output_path}")
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _at_least_200(value: str) -> int:
    parsed = int(value)
    if parsed < 200:
        raise argparse.ArgumentTypeError(
            "representative replay requires at least 200 batches"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run isolated benchmark jobs")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run.add_argument(
        "--batch-sizes", nargs="+", type=_positive_int, default=[1, 2, 4, 8, 16, 32]
    )
    run.add_argument(
        "--fused-batch-sizes",
        nargs="+",
        type=_positive_int,
        default=[4, 8, 16, 32],
        help="sub-batch sizes benchmarked for standard and graph baselines",
    )
    run.add_argument(
        "--synthetic-profiles",
        nargs="*",
        choices=tuple(SYNTHETIC_PROFILES),
        default=list(SYNTHETIC_PROFILES),
    )
    run.add_argument("--real-length-profile")
    run.add_argument("--real-profile-name", default="tarteel")
    run.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    run.add_argument(
        "--vocab-size",
        type=_positive_int,
        default=1025,
        help="including the final blank symbol",
    )
    run.add_argument("--encoder-hidden", type=_positive_int, default=512)
    run.add_argument("--predictor-hidden", type=_positive_int, default=640)
    run.add_argument("--joint-hidden", type=_positive_int, default=640)
    run.add_argument("--warmup", type=_positive_int, default=20)
    run.add_argument("--iterations", type=_positive_int, default=50)
    run.add_argument("--real-iterations", type=_positive_int, default=200)
    run.add_argument("--prune-range", type=_positive_int, default=5)
    run.add_argument("--simple-loss-scale", type=float, default=0.5)
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument(
        "--profile-traces", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--profile-batch-size", type=_positive_int, default=4)
    run.set_defaults(handler=_run_command)

    profile = subparsers.add_parser(
        "build-length-profile",
        help="tokenize manifest text and build representative synthetic encoder lengths",
    )
    profile.add_argument("--manifest", required=True)
    profile.add_argument("--tokenizer-model", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--text-field", default="text")
    profile.add_argument("--expected-vocab-size", type=_positive_int, default=1024)
    profile.add_argument("--batch-size", type=_positive_int, default=32)
    profile.add_argument("--num-batches", type=_at_least_200, default=200)
    profile.add_argument("--tail-fraction", type=float, default=0.10)
    profile.add_argument("--main-min-frames", type=_positive_int, default=100)
    profile.add_argument("--main-max-frames", type=_positive_int, default=250)
    profile.add_argument("--tail-max-frames", type=_positive_int, default=500)
    profile.add_argument("--seed", type=int, default=12345)
    profile.set_defaults(handler=_build_length_profile)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True)
    worker.add_argument("--result", required=True)
    worker.set_defaults(handler=_worker_main)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
