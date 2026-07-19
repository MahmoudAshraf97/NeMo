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

"""Exact RNN-T with a bounded joint workspace and activation recomputation."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo.collections.asr.losses.native_rnnt import NativeRNNTLoss
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


def _relu_join_impl(encoder: torch.Tensor, predictor: torch.Tensor) -> torch.Tensor:
    return F.relu(encoder.unsqueeze(2) + predictor.unsqueeze(1))


_relu_join = torch.compile(
    _relu_join_impl, fullgraph=True, options={"triton.cudagraphs": False}
)


def _activate(value: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "relu":
        return F.relu(value)
    if activation == "sigmoid":
        return torch.sigmoid(value)
    if activation == "tanh":
        return torch.tanh(value)
    raise ValueError(f"Unsupported RNN-T joint activation: {activation}")


def _join_hidden(
    encoder: torch.Tensor, predictor: torch.Tensor, activation: str
) -> torch.Tensor:
    if activation == "relu":
        return _relu_join(encoder, predictor)
    return _activate(encoder.unsqueeze(2) + predictor.unsqueeze(1), activation)


def _chunk_scores(
    projected_encoder: torch.Tensor,
    projected_predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    activation: str,
    blank: int,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate transition scores for one disposable joint chunk."""
    from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import (
        rnnt_logprobs_triton,
    )

    hidden = _join_hidden(projected_encoder, projected_predictor, activation)
    logits = F.linear(hidden, output_weight, output_bias)
    return rnnt_logprobs_triton(
        logits,
        targets,
        blank,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        clamp=clamp,
        reuse_logits_for_grad=True,
    )


def flash_rnnt_loss_from_joint(
    joint,
    encoder: torch.Tensor,
    predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    loss: NativeRNNTLoss,
    workspace_batch_size: int,
    *,
    state_budget: int | None = None,
    sort_by_target: bool = True,
) -> torch.Tensor:
    """Run exact RNN-T with a bounded vocabulary workspace recomputed in backward."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for flash RNN-T training")
    if not encoder.is_cuda:
        raise RuntimeError("Flash RNN-T training requires CUDA tensors")
    if workspace_batch_size < 1:
        raise ValueError("workspace_batch_size must be positive")
    if state_budget is not None and state_budget < 1:
        raise ValueError("state_budget must be positive")
    if joint.is_adapter_available() or joint.masking_prob > 0.0:
        raise ValueError("Flash RNN-T does not support adapters or HAINAN masking")
    if len(joint.joint_net) != 2:
        raise ValueError("Flash RNN-T currently requires joint dropout to be disabled")
    if joint.log_softmax is True or joint.temperature != 1.0:
        raise ValueError(
            "Flash RNN-T requires unnormalized joint logits with temperature 1"
        )

    if sort_by_target:
        order = torch.argsort(target_lengths, stable=True)
        inverse_order = torch.empty_like(order)
        inverse_order.scatter_(
            0, order, torch.arange(order.numel(), device=order.device)
        )
        encoder = encoder.index_select(0, order)
        predictor = predictor.index_select(0, order)
        targets = targets.index_select(0, order)
        source_lengths = source_lengths.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)

    length_pairs = torch.stack((source_lengths, target_lengths), dim=1).tolist()
    chunk_bounds = []
    begin = 0
    while begin < len(length_pairs):
        end = begin
        max_source = 0
        max_target = 0
        limit = min(begin + workspace_batch_size, len(length_pairs))
        while end < limit:
            source, target = length_pairs[end]
            candidate_source = max(max_source, source)
            candidate_target = max(max_target, target)
            candidate_states = (
                (end - begin + 1) * candidate_source * (candidate_target + 1)
            )
            if (
                end > begin
                and state_budget is not None
                and candidate_states > state_budget
            ):
                break
            max_source = candidate_source
            max_target = candidate_target
            end += 1
        chunk_bounds.append(
            (
                begin,
                end,
                max_source,
                max_target,
            )
        )
        begin = end

    from nemo.collections.asr.parts.native_rnnt import rnnt_loss_triton

    encoder = encoder.transpose(1, 2)
    predictor = predictor.transpose(1, 2)
    projected_encoder = F.linear(encoder, joint.enc.weight, joint.enc.bias)
    projected_predictor = F.linear(predictor, joint.pred.weight, joint.pred.bias)
    target_score_chunks = []
    blank_score_chunks = []
    output = joint.joint_net[-1]
    for begin, end, max_source, max_target in chunk_bounds:
        chunk_target_scores, chunk_blank_scores = checkpoint(
            _chunk_scores,
            projected_encoder[begin:end, :max_source],
            projected_predictor[begin:end, : max_target + 1],
            targets[begin:end, :max_target],
            source_lengths[begin:end],
            target_lengths[begin:end],
            output.weight,
            output.bias,
            joint.activation,
            loss.blank,
            loss.clamp,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        padding = (
            0,
            predictor.shape[1] - max_target - 1,
            0,
            encoder.shape[1] - max_source,
        )
        if any(padding):
            chunk_target_scores = F.pad(chunk_target_scores, padding)
            chunk_blank_scores = F.pad(chunk_blank_scores, padding)
        target_score_chunks.append(chunk_target_scores)
        blank_score_chunks.append(chunk_blank_scores)

    target_scores = torch.cat(target_score_chunks)
    blank_scores = torch.cat(blank_score_chunks)

    losses = rnnt_loss_triton(
        target_scores[..., :-1],
        blank_scores,
        source_lengths,
        target_lengths,
        loss.fastemit_lambda,
    )
    return losses.index_select(0, inverse_order) if sort_by_target else losses
