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

from nemo.collections.asr.losses.native_rnnt import NativeRNNTLoss
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


def _relu_join_impl(encoder: torch.Tensor, predictor: torch.Tensor) -> torch.Tensor:
    return F.relu(encoder.unsqueeze(2) + predictor.unsqueeze(1))


def _relu_join_backward_impl(
    grad_hidden: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_preactivation = grad_hidden * (hidden > 0)
    return grad_preactivation.sum(2), grad_preactivation.sum(1)


_relu_join = torch.compile(
    _relu_join_impl, fullgraph=True, options={"triton.cudagraphs": False}
)
_relu_join_backward = torch.compile(
    _relu_join_backward_impl, fullgraph=True, options={"triton.cudagraphs": False}
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


class _FlashRNNT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        encoder,
        predictor,
        targets,
        source_lengths,
        target_lengths,
        encoder_weight,
        encoder_bias,
        predictor_weight,
        predictor_bias,
        output_weight,
        output_bias,
        activation,
        blank,
        fastemit_lambda,
        clamp,
        chunk_bounds,
    ):
        from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import (
            rnnt_logprobs_triton,
        )
        from nemo.collections.asr.parts.native_rnnt import (
            rnnt_loss_triton_with_occupations,
        )

        projected_encoder = F.linear(encoder, encoder_weight, encoder_bias)
        projected_predictor = F.linear(predictor, predictor_weight, predictor_bias)
        target_scores = torch.zeros(
            (encoder.shape[0], encoder.shape[1], predictor.shape[1]),
            device=encoder.device,
            dtype=torch.float32,
        )
        blank_scores = torch.zeros_like(target_scores)
        for begin, end, max_source, max_target in chunk_bounds:
            hidden = _join_hidden(
                projected_encoder[begin:end, :max_source],
                projected_predictor[begin:end, : max_target + 1],
                activation,
            )
            logits = F.linear(hidden, output_weight, output_bias)
            chunk_target_scores, chunk_blank_scores = rnnt_logprobs_triton(
                logits,
                targets[begin:end, :max_target],
                blank,
                source_lengths=source_lengths[begin:end],
                target_lengths=target_lengths[begin:end],
            )
            target_scores[begin:end, :max_source, : max_target + 1] = (
                chunk_target_scores
            )
            blank_scores[begin:end, :max_source, : max_target + 1] = chunk_blank_scores
        losses, target_occupation, blank_occupation = rnnt_loss_triton_with_occupations(
            target_scores[..., :-1],
            blank_scores,
            source_lengths,
            target_lengths,
            fastemit_lambda,
        )

        ctx.save_for_backward(
            encoder,
            predictor,
            targets,
            source_lengths,
            target_lengths,
            projected_encoder,
            projected_predictor,
            encoder_weight,
            predictor_weight,
            output_weight,
            output_bias,
            target_occupation,
            blank_occupation,
        )
        ctx.activation = activation
        ctx.blank = blank
        ctx.fastemit_scale = 1.0 + fastemit_lambda
        ctx.clamp = clamp
        ctx.chunk_bounds = chunk_bounds
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import (
            rnnt_logprobs_grad_triton,
        )

        (
            encoder,
            predictor,
            targets,
            source_lengths,
            target_lengths,
            projected_encoder,
            projected_predictor,
            encoder_weight,
            predictor_weight,
            output_weight,
            output_bias,
            target_occupation,
            blank_occupation,
        ) = ctx.saved_tensors

        grad_projected_encoder = torch.zeros_like(projected_encoder)
        grad_projected_predictor = torch.zeros_like(projected_predictor)
        grad_output_weight = torch.zeros_like(output_weight)
        grad_output_bias = torch.zeros_like(output_bias)
        for begin, end, max_source, max_target in ctx.chunk_bounds:
            hidden = _join_hidden(
                projected_encoder[begin:end, :max_source],
                projected_predictor[begin:end, : max_target + 1],
                ctx.activation,
            )
            logits = F.linear(hidden, output_weight, output_bias)
            scale = grad_losses[begin:end].float().reshape(-1, 1, 1)
            chunk_blank_occupation = blank_occupation[
                begin:end, :max_source, : max_target + 1
            ]
            grad_target_scores = torch.zeros_like(chunk_blank_occupation)
            grad_target_scores[..., :-1] = -target_occupation[
                begin:end, :max_source, :max_target
            ] * (scale * ctx.fastemit_scale)
            grad_blank_scores = -chunk_blank_occupation * scale
            grad_logits = rnnt_logprobs_grad_triton(
                logits,
                targets[begin:end, :max_target],
                ctx.blank,
                grad_target_scores,
                grad_blank_scores,
                source_lengths[begin:end],
                target_lengths[begin:end],
                clamp=ctx.clamp,
                reuse_logits=True,
            )
            grad_logits_2d = grad_logits.flatten(0, 2)
            hidden_2d = hidden.flatten(0, 2)
            grad_hidden = torch.matmul(grad_logits, output_weight)
            grad_output_weight.add_(
                torch.matmul(grad_logits_2d.transpose(0, 1), hidden_2d)
            )
            grad_output_bias.add_(grad_logits_2d.sum(0))
            if ctx.activation == "relu":
                grad_encoder_chunk, grad_predictor_chunk = _relu_join_backward(
                    grad_hidden, hidden
                )
                grad_projected_encoder[begin:end, :max_source] = grad_encoder_chunk
                grad_projected_predictor[begin:end, : max_target + 1] = (
                    grad_predictor_chunk
                )
                continue
            if ctx.activation == "sigmoid":
                grad_preactivation = grad_hidden * hidden * (1 - hidden)
            else:
                grad_preactivation = grad_hidden * (1 - hidden * hidden)
            grad_projected_encoder[begin:end, :max_source] = grad_preactivation.sum(2)
            grad_projected_predictor[begin:end, : max_target + 1] = (
                grad_preactivation.sum(1)
            )

        encoder_2d = encoder.flatten(0, 1)
        predictor_2d = predictor.flatten(0, 1)
        grad_encoder_2d = grad_projected_encoder.flatten(0, 1)
        grad_predictor_2d = grad_projected_predictor.flatten(0, 1)
        grad_encoder = torch.matmul(grad_projected_encoder, encoder_weight)
        grad_predictor = torch.matmul(grad_projected_predictor, predictor_weight)
        grad_encoder_weight = torch.matmul(grad_encoder_2d.transpose(0, 1), encoder_2d)
        grad_predictor_weight = torch.matmul(
            grad_predictor_2d.transpose(0, 1), predictor_2d
        )
        grad_encoder_bias = grad_encoder_2d.sum(0)
        grad_predictor_bias = grad_predictor_2d.sum(0)
        return (
            grad_encoder,
            grad_predictor,
            None,
            None,
            None,
            grad_encoder_weight,
            grad_encoder_bias,
            grad_predictor_weight,
            grad_predictor_bias,
            grad_output_weight,
            grad_output_bias,
            None,
            None,
            None,
            None,
            None,
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

    output = joint.joint_net[-1]
    losses = _FlashRNNT.apply(
        encoder.transpose(1, 2),
        predictor.transpose(1, 2),
        targets,
        source_lengths,
        target_lengths,
        joint.enc.weight,
        joint.enc.bias,
        joint.pred.weight,
        joint.pred.bias,
        output.weight,
        output.bias,
        joint.activation,
        loss.blank,
        loss.fastemit_lambda,
        loss.clamp,
        tuple(chunk_bounds),
    )
    return losses.index_select(0, inverse_order) if sort_by_target else losses
