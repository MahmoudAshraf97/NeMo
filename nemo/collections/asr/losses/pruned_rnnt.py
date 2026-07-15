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

"""Two-pass native pruned RNN-T loss for NeMo's pre-joint path."""

from __future__ import annotations

from typing import Optional

import torch

from nemo.collections.asr.parts.pruned_rnnt import (
    MAX_TARGET_TOKENS,
    _gather_predictor,
    _pruned_logprobs_triton,
    get_prune_ranges,
    get_smoothed_rnnt_logprobs,
    rnnt_loss_triton,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


class PrunedRNNTLoss(torch.nn.Module):
    """Native implementation of the two-pass pruned RNN-T objective.

    This module cannot consume a conventional joint tensor: doing so would
    forfeit the memory reduction. :class:`RNNTJoint` binds its dimensions and
    calls :meth:`forward_from_joint` once for the complete local batch.

    The one-block Triton recurrence supports padded targets up to
    ``MAX_TARGET_TOKENS`` tokens.
    """

    requires_joint_inputs = True

    def __init__(
        self,
        blank: int,
        prune_range: int = 5,
        simple_loss_scale: float = 0.5,
        lm_only_scale: float = 0.25,
        am_only_scale: float = 0.0,
        warmup_steps: int = 2000,
        initial_simple_loss_scale: float = 1.0,
        initial_pruned_loss_scale: float = 0.1,
    ):
        super().__init__()
        if prune_range < 1:
            raise ValueError(f"prune_range must be positive, got {prune_range}")
        scales = {
            "simple_loss_scale": simple_loss_scale,
            "lm_only_scale": lm_only_scale,
            "am_only_scale": am_only_scale,
            "initial_simple_loss_scale": initial_simple_loss_scale,
            "initial_pruned_loss_scale": initial_pruned_loss_scale,
        }
        for name, value in scales.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative, got {value}")
        if lm_only_scale + am_only_scale > 1.0:
            raise ValueError("lm_only_scale + am_only_scale must be no greater than 1")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be nonnegative, got {warmup_steps}")

        self.blank = blank
        self.prune_range = prune_range
        self.simple_loss_scale = simple_loss_scale
        self.lm_only_scale = lm_only_scale
        self.am_only_scale = am_only_scale
        self.warmup_steps = warmup_steps
        self.initial_simple_loss_scale = initial_simple_loss_scale
        self.initial_pruned_loss_scale = initial_pruned_loss_scale

        self.simple_encoder: Optional[torch.nn.Linear] = None
        self.simple_predictor: Optional[torch.nn.Linear] = None
        self._schedule_step = 0
        self._diagnostics: dict[str, torch.Tensor] = {}

    def bind_joint(self, joint: torch.nn.Module) -> None:
        """Validate and bind the compatible joint dimensions once."""
        if joint.__class__.__name__ != "RNNTJoint":
            raise ValueError(
                "pruned_rnnt supports the standard RNNTJoint only; sampled, HAT, TDT, multi-blank, and "
                "multi-output joints are not supported"
            )
        if not joint.fuse_loss_wer:
            raise ValueError(
                "pruned_rnnt requires joint.fuse_loss_wer=true so encoder/predictor tensors reach the loss "
                "before a full [B,T,U+1,V] joint is materialized"
            )
        if (
            joint.num_extra_outputs != 0
            or joint.num_classes_with_blank != self.blank + 1
        ):
            raise ValueError(
                "pruned_rnnt supports only a standard single-blank RNN-T vocabulary"
            )
        if joint.masking_prob > 0:
            raise ValueError(
                "pruned_rnnt does not support HAINAN joint masking (masking_prob must be <= 0)"
            )

        if self.simple_encoder is None:
            reference = next(joint.parameters())
            self.simple_encoder = torch.nn.Linear(
                joint.encoder_hidden, joint.num_classes_with_blank
            ).to(device=reference.device, dtype=reference.dtype)
            self.simple_predictor = torch.nn.Linear(
                joint.pred_hidden, joint.num_classes_with_blank
            ).to(device=reference.device, dtype=reference.dtype)
        elif (
            self.simple_encoder.in_features != joint.encoder_hidden
            or self.simple_predictor.in_features != joint.pred_hidden
        ):
            raise ValueError(
                "The bound RNNTJoint dimensions changed after pruned_rnnt was initialized"
            )

    def set_step(self, global_step: int) -> None:
        self._schedule_step = max(int(global_step), 0)

    def _current_scales(self, training: bool) -> tuple[float, float]:
        """Keep a zero initial pruned scale disabled for the complete warm-up."""
        if not training or self.warmup_steps == 0:
            return self.simple_loss_scale, 1.0
        if self._schedule_step >= self.warmup_steps:
            return self.simple_loss_scale, 1.0
        progress = min(self._schedule_step / self.warmup_steps, 1.0)
        simple_scale = self.initial_simple_loss_scale + progress * (
            self.simple_loss_scale - self.initial_simple_loss_scale
        )
        pruned_scale = self.initial_pruned_loss_scale
        if pruned_scale > 0.0:
            pruned_scale += progress * (1.0 - pruned_scale)
        return simple_scale, pruned_scale

    @staticmethod
    def _zero_joint_dependency(
        joint: torch.nn.Module, reference: torch.Tensor
    ) -> torch.Tensor:
        """Keep joint parameters in the DDP graph while the pruned pass is disabled."""
        zero = reference.new_zeros(())
        for parameter in joint.parameters():
            if parameter.requires_grad and parameter.numel():
                zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero

    @property
    def diagnostics(self) -> dict[str, torch.Tensor]:
        return self._diagnostics

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "RNNTLoss.forward(log_probs, ...) cannot be used with pruned_rnnt: a full joint tensor is already "
            "too late. Set joint.fuse_loss_wer=true and call the model's fused RNNTJoint path."
        )

    def forward_from_joint(
        self,
        joint: torch.nn.Module,
        encoder_outputs: torch.Tensor,
        predictor_outputs: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError(
                "Training with loss_name=pruned_rnnt requires Triton. Check the NeMo CUDA training environment; "
                "inference and checkpoint loading do not require Triton."
            )
        if self.simple_encoder is None or self.simple_predictor is None:
            raise RuntimeError(
                "pruned_rnnt was not bound to RNNTJoint during model setup"
            )
        if (
            encoder_outputs.ndim != 3
            or predictor_outputs.ndim != 3
            or targets.ndim != 2
        ):
            raise ValueError(
                "Expected encoder, predictor, and target tensors with shapes [B,T,D], [B,U+1,D], and [B,U]"
            )
        batch, max_source, _ = encoder_outputs.shape
        max_target = targets.shape[1]
        if targets.shape[0] != batch or predictor_outputs.shape[:2] != (
            batch,
            max_target + 1,
        ):
            raise ValueError(
                "Encoder, predictor, and target batches must match, and predictor length must equal target length plus one"
            )
        if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
            raise ValueError("source_lengths and target_lengths must have shape [B]")
        if max_target > MAX_TARGET_TOKENS:
            raise ValueError(
                f"pruned_rnnt supports at most {MAX_TARGET_TOKENS} padded target tokens with its one-block "
                f"Triton recurrence, got {max_target}"
            )
        tensors = (predictor_outputs, targets, source_lengths, target_lengths)
        if any(tensor.device != encoder_outputs.device for tensor in tensors):
            raise ValueError("All pruned_rnnt inputs must be on the same device")
        if not encoder_outputs.is_cuda:
            raise RuntimeError(
                "Training with loss_name=pruned_rnnt requires CUDA tensors"
            )
        if (
            not encoder_outputs.is_floating_point()
            or not predictor_outputs.is_floating_point()
        ):
            raise ValueError(
                "Encoder and predictor outputs must be floating-point tensors"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if any(
            tensor.dtype not in integer_dtypes
            for tensor in (targets, source_lengths, target_lengths)
        ):
            raise ValueError("Targets and length tensors must have integer dtypes")

        targets = targets.long().contiguous()
        source_lengths = source_lengths.to(dtype=torch.int32).contiguous()
        target_lengths = target_lengths.to(dtype=torch.int32).contiguous()
        simple_am = self.simple_encoder(encoder_outputs)
        simple_lm = self.simple_predictor(predictor_outputs)
        simple_target, simple_blank = get_smoothed_rnnt_logprobs(
            lm=simple_lm,
            am=simple_am,
            targets=targets,
            blank_id=self.blank,
            lm_only_scale=self.lm_only_scale,
            am_only_scale=self.am_only_scale,
        )
        simple_loss, target_occupation, blank_occupation = rnnt_loss_triton(
            simple_target, simple_blank, source_lengths, target_lengths
        )
        simple_scale, pruned_scale = self._current_scales(joint.training)
        if pruned_scale == 0.0:
            self._diagnostics = {
                "simple_loss": simple_loss.detach().mean(),
                "pruned_loss": simple_loss.new_zeros(()),
                "simple_loss_scale": torch.tensor(simple_scale),
                "pruned_loss_scale": torch.tensor(0.0),
            }
            return simple_scale * simple_loss + self._zero_joint_dependency(
                joint, simple_loss
            )

        ranges = get_prune_ranges(
            target_occupation,
            blank_occupation,
            source_lengths,
            target_lengths,
            self.prune_range,
        )

        projected_encoder = joint.project_encoder(encoder_outputs)
        projected_predictor = joint.project_prednet(predictor_outputs)
        gathered_predictor = _gather_predictor(projected_predictor, ranges)
        joint_input = projected_encoder.unsqueeze(2) + gathered_predictor
        if joint.is_adapter_available():
            joint_input = joint.forward_enabled_adapters(joint_input)
        pruned_logits = joint.joint_net(joint_input)
        logit_scale = 1.0 / joint.temperature if joint.log_softmax else 1.0
        pruned_target, pruned_blank = _pruned_logprobs_triton(
            pruned_logits,
            ranges,
            targets,
            self.blank,
            source_lengths,
            target_lengths,
            logit_scale,
        )
        pruned_loss, _, _ = rnnt_loss_triton(
            pruned_target, pruned_blank, source_lengths, target_lengths
        )

        self._diagnostics = {
            "simple_loss": simple_loss.detach().mean(),
            "pruned_loss": pruned_loss.detach().mean(),
            "simple_loss_scale": torch.tensor(simple_scale),
            "pruned_loss_scale": torch.tensor(pruned_scale),
        }
        return simple_scale * simple_loss + pruned_scale * pruned_loss
