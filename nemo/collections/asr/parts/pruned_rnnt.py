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

"""Native building blocks for the two-pass pruned RNN-T loss.

The functions in this module intentionally operate on transition scores rather
than a dense ``[B, T, U + 1, V]`` joint tensor.
"""

from __future__ import annotations

from typing import Tuple

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl


# One Triton block scans U + 1 alignment states, with at most 1024 lanes.
MAX_TARGET_TOKENS = 1023


def _validate_transition_inputs(
    target_scores: torch.Tensor,
    blank_scores: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> None:
    if target_scores.ndim != 3 or blank_scores.ndim != 3:
        raise ValueError(
            "RNN-T transition scores must have shape [B, T, U] and [B, T, U + 1]"
        )
    if (
        blank_scores.shape[:2] != target_scores.shape[:2]
        or blank_scores.shape[2] != target_scores.shape[2] + 1
    ):
        raise ValueError(
            f"Incompatible target/blank score shapes: {tuple(target_scores.shape)} and {tuple(blank_scores.shape)}"
        )
    batch, max_source, max_target = target_scores.shape
    if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("source_lengths and target_lengths must have shape [B]")
    if any(
        tensor.device != target_scores.device
        for tensor in (blank_scores, source_lengths, target_lengths)
    ):
        raise ValueError(
            "Transition scores and length tensors must be on the same device"
        )
    # Value checks are intentionally kept on the CPU oracle path.  CUDA callers
    # are fed by NeMo's validated batch lengths and must not synchronize the host
    # in the timed training path.
    if not source_lengths.is_cuda:
        if bool(torch.any(source_lengths < 1)) or bool(
            torch.any(source_lengths > max_source)
        ):
            raise ValueError("Every source length must be in [1, T]")
        if bool(torch.any(target_lengths < 0)) or bool(
            torch.any(target_lengths > max_target)
        ):
            raise ValueError("Every target length must be in [0, U]")


if TRITON_AVAILABLE:

    @triton.jit
    def _logadd_scan(left, right):
        maximum = tl.maximum(left, right)
        value = maximum + tl.log(tl.exp(left - maximum) + tl.exp(right - maximum))
        return tl.where(maximum == -float("inf"), -float("inf"), value)

    @triton.jit
    def _log_semiring_compose(a_left, b_left, a_right, b_right):
        # Compose F(x) = logadd(a, b + x).  The ``a`` component of an
        # inclusive scan is the RNN-T recurrence value at that lane.
        return _logadd_scan(a_right, b_right + a_left), b_right + b_left

    @triton.jit
    def _rnnt_loss_kernel(
        target_scores_ptr,
        blank_scores_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        alpha_ptr,
        losses_ptr,
        beta_ptr,
        blank_occupation_ptr,
        max_source,
        max_target,
        block_target: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        symbols = tl.arange(0, block_target)
        valid_symbol = symbols <= target_len
        previous = tl.where(symbols == 0, 0.0, -float("inf"))

        for time_idx in tl.range(0, max_source):
            active_time = time_idx < source_len
            blank_offset = (batch_idx * max_source + time_idx - 1) * (
                max_target + 1
            ) + symbols
            from_blank = previous + tl.load(
                blank_scores_ptr + blank_offset,
                mask=(time_idx > 0) & active_time & valid_symbol,
                other=0.0,
            )
            base = tl.where(
                time_idx == 0, tl.where(symbols == 0, 0.0, -float("inf")), from_blank
            )

            target_offset = (
                (batch_idx * max_source + time_idx) * max_target + symbols - 1
            )
            step = tl.load(
                target_scores_ptr + target_offset,
                mask=active_time & (symbols > 0) & (symbols <= target_len),
                other=-float("inf"),
            )
            current, _ = tl.associative_scan(
                (base, step), axis=0, combine_fn=_log_semiring_compose
            )
            current = tl.where(valid_symbol, current, -float("inf"))

            alpha_offset = (batch_idx * max_source + time_idx) * (
                max_target + 1
            ) + symbols
            tl.store(alpha_ptr + alpha_offset, current, mask=active_time & valid_symbol)
            previous = tl.where(active_time, current, previous)

        final_alpha = tl.sum(tl.where(symbols == target_len, previous, 0.0), axis=0)
        final_blank_offset = (batch_idx * max_source + source_len - 1) * (
            max_target + 1
        ) + target_len
        final_blank = tl.load(blank_scores_ptr + final_blank_offset)
        log_likelihood = final_alpha + final_blank
        tl.store(losses_ptr + batch_idx, -log_likelihood)
        tl.debug_barrier()

        reverse_idx = tl.arange(0, block_target)
        symbols = max_target - reverse_idx
        valid_symbol = (symbols >= 0) & (symbols <= target_len)
        beta_next = tl.full((block_target,), -float("inf"), tl.float32)

        for reverse_time in tl.range(0, max_source):
            time_idx = max_source - reverse_time - 1
            active_time = time_idx < source_len
            blank_offset = (batch_idx * max_source + time_idx) * (
                max_target + 1
            ) + symbols
            blank = tl.load(
                blank_scores_ptr + blank_offset,
                mask=active_time & valid_symbol,
                other=0.0,
            )
            last_time = time_idx == source_len - 1
            base = tl.where(
                last_time,
                tl.where(symbols == target_len, blank, -float("inf")),
                blank + beta_next,
            )
            target_offset = (batch_idx * max_source + time_idx) * max_target + symbols
            target = tl.load(
                target_scores_ptr + target_offset,
                mask=active_time & (symbols >= 0) & (symbols < target_len),
                other=-float("inf"),
            )
            step = tl.where(
                (reverse_idx > 0) & (symbols >= 0) & (symbols < target_len),
                target,
                -float("inf"),
            )
            beta, _ = tl.associative_scan(
                (base, step), axis=0, combine_fn=_log_semiring_compose
            )
            beta = tl.where(valid_symbol, beta, -float("inf"))

            beta_offset = (batch_idx * max_source + time_idx) * (
                max_target + 1
            ) + symbols
            alpha = tl.load(
                alpha_ptr + beta_offset,
                mask=active_time & valid_symbol,
                other=-float("inf"),
            )
            blank_occ = tl.where(
                last_time,
                tl.where(symbols == target_len, 1.0, 0.0),
                tl.exp(alpha + blank + beta_next - log_likelihood),
            )
            tl.store(
                blank_occupation_ptr + blank_offset,
                tl.where(active_time & valid_symbol, blank_occ, 0.0),
                mask=(symbols >= 0) & (symbols <= max_target),
            )
            tl.store(beta_ptr + beta_offset, beta, mask=active_time & valid_symbol)
            beta_next = tl.where(active_time, beta, beta_next)

    @triton.jit
    def _rnnt_target_occupation_kernel(
        target_scores_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        alpha_ptr,
        beta_ptr,
        losses_ptr,
        target_occupation_ptr,
        max_source,
        max_target,
        block_target: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        time_idx = tl.program_id(1)
        symbols = tl.arange(0, block_target)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        valid = (time_idx < source_len) & (symbols < target_len)
        alpha_offset = (batch_idx * max_source + time_idx) * (max_target + 1) + symbols
        target_offset = (batch_idx * max_source + time_idx) * max_target + symbols
        alpha = tl.load(alpha_ptr + alpha_offset, mask=valid, other=-float("inf"))
        target = tl.load(
            target_scores_ptr + target_offset, mask=valid, other=-float("inf")
        )
        beta_after = tl.load(
            beta_ptr + alpha_offset + 1, mask=valid, other=-float("inf")
        )
        log_likelihood = -tl.load(losses_ptr + batch_idx)
        occupation = tl.exp(alpha + target + beta_after - log_likelihood)
        tl.store(
            target_occupation_ptr + target_offset,
            tl.where(valid, occupation, 0.0),
            mask=symbols < max_target,
        )


class _RNNTLossTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target_scores, blank_scores, source_lengths, target_lengths):
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is required for native pruned RNN-T CUDA training"
            )
        if not target_scores.is_cuda:
            raise RuntimeError("Native pruned RNN-T training requires CUDA tensors")
        _validate_transition_inputs(
            target_scores, blank_scores, source_lengths, target_lengths
        )

        target_scores = target_scores.contiguous()
        blank_scores = blank_scores.contiguous()
        if source_lengths.dtype != torch.int32:
            source_lengths = source_lengths.to(dtype=torch.int32)
        if target_lengths.dtype != torch.int32:
            target_lengths = target_lengths.to(dtype=torch.int32)
        source_lengths = source_lengths.contiguous()
        target_lengths = target_lengths.contiguous()
        batch, max_source, max_target = target_scores.shape
        if max_target > MAX_TARGET_TOKENS:
            raise ValueError(
                f"Native pruned RNN-T supports at most {MAX_TARGET_TOKENS} padded target tokens with its "
                f"one-block Triton recurrence, got {max_target}"
            )
        block_target = triton.next_power_of_2(max_target + 1)

        alpha = torch.empty_like(blank_scores, dtype=torch.float32)
        beta = torch.empty_like(blank_scores, dtype=torch.float32)
        losses = torch.empty((batch,), device=target_scores.device, dtype=torch.float32)
        target_occupation = torch.empty_like(target_scores, dtype=torch.float32)
        blank_occupation = torch.empty_like(blank_scores, dtype=torch.float32)
        _rnnt_loss_kernel[(batch,)](
            target_scores,
            blank_scores,
            source_lengths,
            target_lengths,
            alpha,
            losses,
            beta,
            blank_occupation,
            max_source=max_source,
            max_target=max_target,
            block_target=block_target,
            num_warps=8 if block_target >= 128 else 4,
        )
        if max_target:
            _rnnt_target_occupation_kernel[(batch, max_source)](
                target_scores,
                source_lengths,
                target_lengths,
                alpha,
                beta,
                losses,
                target_occupation,
                max_source=max_source,
                max_target=max_target,
                block_target=triton.next_power_of_2(max_target),
            )
        ctx.save_for_backward(target_occupation, blank_occupation)
        ctx.mark_non_differentiable(target_occupation, blank_occupation)
        return losses, target_occupation, blank_occupation

    @staticmethod
    def backward(ctx, grad_losses, _grad_target_occupation, _grad_blank_occupation):
        target_occupation, blank_occupation = ctx.saved_tensors
        scale = grad_losses.float().reshape(-1, 1, 1)
        target_grad = (-target_occupation * scale).to(target_occupation.dtype)
        blank_grad = (-blank_occupation * scale).to(blank_occupation.dtype)
        return target_grad, blank_grad, None, None


def rnnt_loss_triton(
    target_scores: torch.Tensor,
    blank_scores: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA RNN-T recursion returning per-sample loss and occupations.

    The one-block target scan supports at most ``MAX_TARGET_TOKENS`` padded
    target positions.
    """
    return _RNNTLossTriton.apply(
        target_scores, blank_scores, source_lengths, target_lengths
    )


def get_smoothed_rnnt_logprobs(
    lm: torch.Tensor,
    am: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    lm_only_scale: float,
    am_only_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute simple-joiner transition scores without a dense joint tensor.

    This follows the smoothed simple RNN-T objective from the pruned RNN-T
    paper.  Normalization of the combined AM/LM scores is a batched matrix
    multiplication over exponentiated, stabilized logits.
    """
    am = am.float()
    lm = lm.float()
    batch, max_source, vocab = am.shape
    max_target = targets.shape[1]
    if lm.shape != (batch, max_target + 1, vocab):
        raise ValueError(
            f"Expected LM logits {(batch, max_target + 1, vocab)}, got {tuple(lm.shape)}"
        )

    am_max = am.amax(dim=2, keepdim=True)
    lm_max = lm.amax(dim=2, keepdim=True)
    am_probs = torch.exp(am - am_max)
    lm_probs = torch.exp(lm - lm_max)
    tiny = torch.finfo(torch.float32).tiny
    normalizers = torch.bmm(lm_probs, am_probs.transpose(1, 2)).clamp_min(tiny).log()
    normalizers = normalizers + lm_max + am_max.transpose(1, 2)

    lm_only_normalizers = lm_probs.sum(dim=2, keepdim=True)
    unigram_lm = (
        (lm_probs / lm_only_normalizers).mean(dim=(0, 1), keepdim=True).clamp_min(tiny)
    )
    am_only_normalizers = (
        torch.mv(am_probs.reshape(-1, vocab), unigram_lm.reshape(vocab))
        .reshape(batch, max_source, 1)
        .log()
        + am_max
    ).transpose(1, 2)
    log_unigram_lm = unigram_lm.log()
    lm_only_normalizers = lm_only_normalizers.log() + lm_max

    if max_target:
        target_index_am = targets.unsqueeze(2).expand(batch, max_target, max_source)
        target_am = torch.gather(
            am.transpose(1, 2), dim=1, index=target_index_am
        ).transpose(1, 2)
        target_lm = torch.gather(
            lm[:, :max_target], dim=2, index=targets.unsqueeze(-1)
        ).transpose(1, 2)
        target_lm_unigram = torch.gather(
            log_unigram_lm.expand(batch, max_target, vocab),
            dim=2,
            index=targets.unsqueeze(-1),
        ).transpose(1, 2)
        target_combined = (
            target_am + target_lm - normalizers[:, :max_target].transpose(1, 2)
        )
        target_lm_only = target_lm - lm_only_normalizers[:, :max_target].transpose(1, 2)
        target_am_only = (
            target_am + target_lm_unigram - am_only_normalizers.transpose(1, 2)
        )
    else:
        target_combined = am.new_empty((batch, max_source, 0))
        target_lm_only = target_combined
        target_am_only = target_combined

    blank_am = am[:, :, blank_id].unsqueeze(1)
    blank_lm = lm[:, :, blank_id].unsqueeze(2)
    blank_combined = (blank_am + blank_lm - normalizers).transpose(1, 2)
    blank_lm_only = (blank_lm - lm_only_normalizers).transpose(1, 2)
    blank_am_only = (
        blank_am + log_unigram_lm[0, 0, blank_id] - am_only_normalizers
    ).transpose(1, 2)

    combined_scale = 1.0 - lm_only_scale - am_only_scale
    target_scores = (
        combined_scale * target_combined
        + lm_only_scale * target_lm_only
        + am_only_scale * target_am_only
    )
    blank_scores = (
        combined_scale * blank_combined
        + lm_only_scale * blank_lm_only
        + am_only_scale * blank_am_only
    )
    return target_scores.contiguous(), blank_scores.contiguous()


def get_prune_ranges(
    target_occupation: torch.Tensor,
    blank_occupation: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    prune_range: int,
) -> torch.Tensor:
    """Select and monotonically adjust the symbol windows kept per frame."""
    batch, max_source, max_target = target_occupation.shape
    effective_range = min(prune_range, max_target + 1)
    if max_target == 0:
        return torch.zeros(
            (batch, max_source, 1), dtype=torch.long, device=target_occupation.device
        )
    if effective_range < 2:
        raise ValueError(
            "A regular RNN-T batch with non-empty targets requires prune_range >= 2"
        )

    blank_window_score = blank_occupation.unfold(2, effective_range, 1).sum(dim=-1)
    padded_target = torch.nn.functional.pad(target_occupation, (1, 0))
    candidates = blank_window_score - padded_target[:, :, : blank_window_score.shape[2]]
    starts = torch.argmax(candidates, dim=2)

    frame_index = torch.arange(max_source, device=starts.device).unsqueeze(0)
    padding_start = (target_lengths - effective_range + 1).clamp_min(0).unsqueeze(1)
    starts = torch.where(
        frame_index < source_lengths.unsqueeze(1) - 1, starts, padding_start
    )

    starts = torch.flip(
        torch.cummin(torch.flip(starts, dims=(-1,)), dim=-1).values, dims=(-1,)
    )
    offset = (effective_range - 1) * torch.arange(max_source, device=starts.device)
    transformed = -(starts - offset)
    transformed = torch.flip(
        torch.cummin(torch.flip(transformed, dims=(-1,)), dim=-1).values, dims=(-1,)
    )
    transformed = transformed.clamp_min(0)
    starts = -(transformed - offset)

    return starts.unsqueeze(2) + torch.arange(
        effective_range, device=starts.device
    ).reshape(1, 1, -1)


if TRITON_AVAILABLE:

    @triton.jit
    def _pruned_logprobs_fwd_kernel(
        logits_ptr,
        ranges_ptr,
        targets_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        target_scores_ptr,
        blank_scores_ptr,
        max_source,
        max_target,
        prune_range: tl.constexpr,
        vocab_size: tl.constexpr,
        blank_id: tl.constexpr,
        block_vocab: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        time_idx = tl.program_id(1)
        prune_idx = tl.program_id(2)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        range_offset = (batch_idx * max_source + time_idx) * prune_range + prune_idx
        symbol_idx = tl.load(ranges_ptr + range_offset)
        if time_idx >= source_len or symbol_idx > target_len:
            return

        logits_offset = range_offset * vocab_size
        vocab_idx = tl.arange(0, block_vocab)
        vocab_mask = vocab_idx < vocab_size
        logits = tl.load(
            logits_ptr + logits_offset + vocab_idx, mask=vocab_mask, other=-float("inf")
        ).to(tl.float32)
        maximum = tl.max(logits, axis=0)
        denominator = tl.log(tl.sum(tl.exp(logits - maximum), axis=0)) + maximum
        blank_logit = tl.load(logits_ptr + logits_offset + blank_id).to(tl.float32)
        score_offset = (batch_idx * max_source + time_idx) * (
            max_target + 1
        ) + symbol_idx
        tl.store(blank_scores_ptr + score_offset, blank_logit - denominator)
        if symbol_idx < target_len:
            target_id = tl.load(targets_ptr + batch_idx * max_target + symbol_idx)
            target_logit = tl.load(logits_ptr + logits_offset + target_id).to(
                tl.float32
            )
            target_offset = (
                batch_idx * max_source + time_idx
            ) * max_target + symbol_idx
            tl.store(target_scores_ptr + target_offset, target_logit - denominator)

    @triton.jit
    def _pruned_logprobs_bwd_kernel(
        logits_ptr,
        grad_logits_ptr,
        ranges_ptr,
        targets_ptr,
        source_lengths_ptr,
        target_lengths_ptr,
        grad_target_scores_ptr,
        grad_blank_scores_ptr,
        max_source,
        max_target,
        prune_range: tl.constexpr,
        vocab_size: tl.constexpr,
        blank_id: tl.constexpr,
        block_vocab: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        time_idx = tl.program_id(1)
        prune_idx = tl.program_id(2)
        source_len = tl.load(source_lengths_ptr + batch_idx)
        target_len = tl.load(target_lengths_ptr + batch_idx)
        range_offset = (batch_idx * max_source + time_idx) * prune_range + prune_idx
        symbol_idx = tl.load(ranges_ptr + range_offset)
        if time_idx >= source_len or symbol_idx > target_len:
            return

        logits_offset = range_offset * vocab_size
        vocab_idx = tl.arange(0, block_vocab)
        vocab_mask = vocab_idx < vocab_size
        logits = tl.load(
            logits_ptr + logits_offset + vocab_idx, mask=vocab_mask, other=-float("inf")
        ).to(tl.float32)
        maximum = tl.max(logits, axis=0)
        softmax = tl.exp(logits - maximum)
        softmax = softmax / tl.sum(softmax, axis=0)
        blank_score_offset = (batch_idx * max_source + time_idx) * (
            max_target + 1
        ) + symbol_idx
        blank_grad = tl.load(grad_blank_scores_ptr + blank_score_offset).to(tl.float32)
        has_target = symbol_idx < target_len
        target_score_offset = (
            batch_idx * max_source + time_idx
        ) * max_target + symbol_idx
        target_grad = tl.load(
            grad_target_scores_ptr + target_score_offset, mask=has_target, other=0.0
        ).to(tl.float32)
        target_id = tl.load(
            targets_ptr + batch_idx * max_target + symbol_idx, mask=has_target, other=-1
        )
        grad = -softmax * (blank_grad + target_grad)
        grad += tl.where(vocab_idx == blank_id, blank_grad, 0.0)
        grad += tl.where(vocab_idx == target_id, target_grad, 0.0)
        tl.store(grad_logits_ptr + logits_offset + vocab_idx, grad, mask=vocab_mask)


class _PrunedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, ranges, targets, blank_id, source_lengths, target_lengths):
        if not TRITON_AVAILABLE or not logits.is_cuda:
            raise RuntimeError(
                "Native pruned RNN-T log-probabilities require Triton and CUDA"
            )
        logits = logits.contiguous()
        ranges = ranges.contiguous()
        targets = targets.contiguous()
        if source_lengths.dtype != torch.int32:
            source_lengths = source_lengths.to(dtype=torch.int32)
        if target_lengths.dtype != torch.int32:
            target_lengths = target_lengths.to(dtype=torch.int32)
        source_lengths = source_lengths.contiguous()
        target_lengths = target_lengths.contiguous()
        batch, max_source, prune_range, vocab_size = logits.shape
        max_target = targets.shape[1]
        target_scores = torch.full(
            (batch, max_source, max_target),
            float("-inf"),
            dtype=torch.float32,
            device=logits.device,
        )
        blank_scores = torch.full(
            (batch, max_source, max_target + 1),
            float("-inf"),
            dtype=torch.float32,
            device=logits.device,
        )
        _pruned_logprobs_fwd_kernel[(batch, max_source, prune_range)](
            logits,
            ranges,
            targets,
            source_lengths,
            target_lengths,
            target_scores,
            blank_scores,
            max_source=max_source,
            max_target=max_target,
            prune_range=prune_range,
            vocab_size=vocab_size,
            blank_id=blank_id,
            block_vocab=triton.next_power_of_2(vocab_size),
        )
        ctx.save_for_backward(logits, ranges, targets, source_lengths, target_lengths)
        ctx.blank_id = blank_id
        return target_scores, blank_scores

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        logits, ranges, targets, source_lengths, target_lengths = ctx.saved_tensors
        batch, max_source, prune_range, vocab_size = logits.shape
        max_target = targets.shape[1]
        grad_logits = torch.zeros_like(logits)
        _pruned_logprobs_bwd_kernel[(batch, max_source, prune_range)](
            logits,
            grad_logits,
            ranges,
            targets,
            source_lengths,
            target_lengths,
            grad_target_scores.contiguous(),
            grad_blank_scores.contiguous(),
            max_source=max_source,
            max_target=max_target,
            prune_range=prune_range,
            vocab_size=vocab_size,
            blank_id=ctx.blank_id,
            block_vocab=triton.next_power_of_2(vocab_size),
        )
        return grad_logits, None, None, None, None, None


def pruned_logprobs_triton(
    logits: torch.Tensor,
    ranges: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract reduced-joint target and blank scores with softmax recomputation."""
    return _PrunedLogProbs.apply(
        logits, ranges, targets, blank_id, source_lengths, target_lengths
    )
