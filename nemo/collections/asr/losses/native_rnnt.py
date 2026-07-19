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

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE


class NativeRNNTLoss(torch.nn.Module):
    """Exact CUDA RNN-T loss using native Triton kernels."""

    def __init__(self, blank: int, fastemit_lambda: float = 0.0, clamp: float = -1.0):
        super().__init__()
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be nonnegative")
        self.blank = blank
        self.fastemit_lambda = float(fastemit_lambda)
        self.clamp = float(clamp) if clamp > 0.0 else 0.0
        self.reduction = None

    def forward(self, acts, labels, act_lens, label_lens, reuse_logits_for_grad=False):
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for native RNN-T training")
        if not acts.is_cuda:
            raise RuntimeError("Native RNN-T training requires CUDA tensors")

        from nemo.collections.asr.parts.k2.rnnt_logprobs_triton import (
            rnnt_logprobs_triton,
        )
        from nemo.collections.asr.parts.native_rnnt import (
            MAX_TARGET_TOKENS,
            rnnt_loss_triton,
        )

        max_target = acts.shape[2] - 1
        if max_target > MAX_TARGET_TOKENS:
            raise ValueError(
                f"Native RNN-T supports at most {MAX_TARGET_TOKENS} padded target tokens with its "
                f"one-block Triton recurrence, got {max_target}"
            )

        target_scores, blank_scores = rnnt_logprobs_triton(
            acts,
            labels,
            self.blank,
            source_lengths=act_lens,
            target_lengths=label_lens,
            clamp=self.clamp,
            reuse_logits_for_grad=reuse_logits_for_grad,
        )
        return rnnt_loss_triton(
            target_scores[..., :-1],
            blank_scores,
            act_lens,
            label_lens,
            fastemit_lambda=self.fastemit_lambda,
        )
