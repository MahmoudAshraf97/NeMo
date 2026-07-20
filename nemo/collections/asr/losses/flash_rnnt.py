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

from nemo.collections.asr.parts.rnnt_loss_triton import MAX_TARGET_TOKENS
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


def _validate_joint(joint, blank: int) -> torch.nn.Dropout | None:
    """Validate the narrow RNNTJoint structure consumed by the flash path."""
    if joint.is_adapter_available() or joint.masking_prob > 0.0:
        raise ValueError("Flash RNN-T does not support adapters or HAINAN masking")
    if joint.num_extra_outputs != 0 or blank != joint.num_classes_with_blank - 1:
        raise ValueError("Flash RNN-T requires standard RNN-T with a final blank output")
    if len(joint.joint_net) not in (2, 3) or not isinstance(joint.joint_net[-1], torch.nn.Linear):
        raise ValueError("Flash RNN-T requires activation, optional dropout, and one output linear layer")

    output = joint.joint_net[-1]
    if output.out_features != joint.num_classes_with_blank:
        raise ValueError("Flash RNN-T requires the joint output to include every label and the blank")
    if not 0 <= blank < output.out_features:
        raise ValueError(f"blank={blank} must be in [0, {output.out_features})")

    dropout = joint.joint_net[1] if len(joint.joint_net) == 3 else None
    if dropout is not None and not isinstance(dropout, torch.nn.Dropout):
        raise ValueError("Flash RNN-T only supports torch.nn.Dropout in the joint network")
    if joint.log_softmax is True or joint.temperature != 1.0:
        raise ValueError("Flash RNN-T requires unnormalized joint logits with temperature 1")
    return dropout


class FlashRNNTLoss(torch.nn.Module):
    """Exact Flash RNN-T loss configured for the fused joint path.

    ``RNNTJoint.fused_batch_size`` supplies ``max_samples_per_chunk``. It limits
    samples in each chunk, not the number of chunks. A local batch of size ``B``
    produces ``ceil(B / fused_batch_size)`` chunks.
    Larger chunks generally reduce launch/recomputation overhead but use a
    larger disposable joint workspace; smaller chunks trade throughput for a
    smaller workspace.

    ``max_target_tokens`` is a safety guard, not a compile-time reservation.
    The actual padded target length selects the Triton scan width, so leaving
    the guard at its maximum does not slow ordinary batches. Very large actual
    target dimensions require new, increasingly expensive kernel compilations
    and accumulate more float32 scan error.
    """

    def __init__(
        self,
        blank: int,
        fastemit_lambda: float = 0.0,
        clamp: float = -1.0,
        max_target_tokens: int = MAX_TARGET_TOKENS,
    ):
        super().__init__()
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be nonnegative")
        if not 0 <= max_target_tokens <= MAX_TARGET_TOKENS:
            raise ValueError(f"max_target_tokens must be in [0, {MAX_TARGET_TOKENS}]")
        self.blank = blank
        self.fastemit_lambda = float(fastemit_lambda)
        self.clamp = float(clamp) if clamp > 0.0 else 0.0
        self.max_target_tokens = max_target_tokens

    def forward(
        self,
        joint,
        encoder: torch.Tensor,
        predictor: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        max_samples_per_chunk: int,
    ) -> torch.Tensor:
        """Return per-sample losses from channel-first joint inputs."""
        return _compute_flash_rnnt(
            joint=joint,
            encoder=encoder,
            predictor=predictor,
            targets=targets,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            blank=self.blank,
            fastemit_lambda=self.fastemit_lambda,
            clamp=self.clamp,
            max_samples_per_chunk=max_samples_per_chunk,
            max_target_tokens=self.max_target_tokens,
        )


def _relu_join_impl(encoder: torch.Tensor, predictor: torch.Tensor) -> torch.Tensor:
    return F.relu(encoder.unsqueeze(2) + predictor.unsqueeze(1))


_relu_join = torch.compile(_relu_join_impl, fullgraph=True, options={"triton.cudagraphs": False})


def _activate(value: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "relu":
        return F.relu(value)
    if activation == "sigmoid":
        return torch.sigmoid(value)
    if activation == "tanh":
        return torch.tanh(value)
    raise ValueError(f"Unsupported RNN-T joint activation: {activation}")


def _join_hidden(encoder: torch.Tensor, predictor: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "relu":
        # Chunk views retain padded-batch strides; normalize the smaller projected
        # inputs so Dynamo can generalize shapes instead of specializing every stride.
        return _relu_join(encoder.contiguous(), predictor.contiguous())
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
    dropout_p: float,
    blank: int,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate transition scores for one disposable joint chunk."""
    from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import rnnt_logprobs_triton

    hidden = _join_hidden(projected_encoder, projected_predictor, activation)
    hidden = F.dropout(hidden, p=dropout_p, training=True)
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


def _compute_flash_rnnt(
    joint,
    encoder: torch.Tensor,
    predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    fastemit_lambda: float,
    clamp: float,
    max_samples_per_chunk: int,
    max_target_tokens: int,
) -> torch.Tensor:
    """Run exact RNN-T with a bounded vocabulary workspace recomputed in backward."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for flash RNN-T training")
    if not encoder.is_cuda:
        raise RuntimeError("Flash RNN-T training requires CUDA tensors")
    if max_samples_per_chunk < 1:
        raise ValueError("max_samples_per_chunk must be positive")
    dropout = _validate_joint(joint, blank)
    batch = encoder.shape[0]
    if predictor.shape[0] != batch or targets.shape[0] != batch:
        raise ValueError("encoder, predictor, and targets must have the same batch size")
    if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("source_lengths and target_lengths must contain one value per batch item")

    order = torch.argsort(target_lengths, stable=True)
    source_lengths = source_lengths.index_select(0, order)
    target_lengths = target_lengths.index_select(0, order)

    num_chunks = (batch + max_samples_per_chunk - 1) // max_samples_per_chunk
    padded_batch = num_chunks * max_samples_per_chunk
    length_pairs = torch.stack((source_lengths, target_lengths), dim=1)
    if padded_batch != batch:
        length_pairs = F.pad(length_pairs, (0, 0, 0, padded_batch - batch))
    chunk_maxima = length_pairs.view(num_chunks, max_samples_per_chunk, 2).amax(dim=1).tolist()
    if chunk_maxima[-1][1] > max_target_tokens:
        raise ValueError(
            f"Batch target length {chunk_maxima[-1][1]} exceeds configured max_target_tokens={max_target_tokens}"
        )

    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    encoder = encoder.index_select(0, order)
    predictor = predictor.index_select(0, order)
    targets = targets.index_select(0, order)

    encoder = encoder.transpose(1, 2)
    predictor = predictor.transpose(1, 2)
    projected_encoder = F.linear(encoder, joint.enc.weight, joint.enc.bias)
    projected_predictor = F.linear(predictor, joint.pred.weight, joint.pred.bias)

    from nemo.collections.asr.parts.rnnt_loss_triton import rnnt_loss_triton

    target_score_chunks = []
    blank_score_chunks = []
    output = joint.joint_net[-1]
    dropout_p = dropout.p if dropout is not None and dropout.training else 0.0
    for chunk_index, (max_source, max_target) in enumerate(chunk_maxima):
        begin = chunk_index * max_samples_per_chunk
        end = min(begin + max_samples_per_chunk, batch)
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
            dropout_p,
            blank,
            clamp,
            use_reentrant=False,
            # Backward recomputation must use the same mask as the forward chunk.
            preserve_rng_state=dropout_p > 0.0,
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
        fastemit_lambda,
    )
    return losses.index_select(0, inverse_order)
