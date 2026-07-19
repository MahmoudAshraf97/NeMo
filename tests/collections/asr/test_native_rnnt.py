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

import pytest
import torch

from nemo.collections.asr.losses.native_rnnt import NativeRNNTLoss
from nemo.collections.asr.losses.flash_rnnt import flash_rnnt_loss_from_joint
from nemo.collections.asr.losses.rnnt import NUMBA_RNNT_AVAILABLE, RNNTLoss
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.native_rnnt import MAX_TARGET_TOKENS
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()


def _rnnt_loss_torch(logits, labels, source_lengths, target_lengths, blank):
    log_probs = logits.float().log_softmax(dim=-1)
    losses = []
    for batch_idx in range(logits.shape[0]):
        source_len = int(source_lengths[batch_idx])
        target_len = int(target_lengths[batch_idx])
        alpha = log_probs.new_full((target_len + 1,), -torch.inf)
        alpha[0] = 0.0
        for time_idx in range(source_len):
            if time_idx:
                alpha = (
                    alpha + log_probs[batch_idx, time_idx - 1, : target_len + 1, blank]
                )
            values = [alpha[0]]
            for target_idx in range(target_len):
                target_score = log_probs[
                    batch_idx, time_idx, target_idx, labels[batch_idx, target_idx]
                ]
                values.append(
                    torch.logaddexp(alpha[target_idx + 1], values[-1] + target_score)
                )
            alpha = torch.stack(values)
        losses.append(
            -(alpha[-1] + log_probs[batch_idx, source_len - 1, target_len, blank])
        )
    return torch.stack(losses)


def _inputs(blank, dtype=torch.float32, vocab=8):
    torch.manual_seed(7)
    batch, source, target = 3, 6, 4
    logits = torch.randn(
        batch, source, target + 1, vocab, device="cuda", dtype=dtype, requires_grad=True
    )
    low = 1 if blank == 0 else 0
    high = vocab if blank == 0 else vocab - 1
    labels = torch.randint(low, high, (batch, target), device="cuda")
    source_lengths = torch.tensor([6, 4, 3], device="cuda", dtype=torch.int64)
    target_lengths = torch.tensor([4, 2, 0], device="cuda", dtype=torch.int64)
    return logits, labels, source_lengths, target_lengths


@pytest.mark.unit
def test_native_rnnt_configuration_and_validation(monkeypatch):
    loss = RNNTLoss(
        num_classes=8,
        loss_name="native_rnnt",
        loss_kwargs={"fastemit_lambda": 0.01, "clamp": -1.0},
    )
    assert isinstance(loss._loss, NativeRNNTLoss)
    assert loss._loss.fastemit_lambda == 0.01
    assert loss._loss.clamp == 0.0
    assert not loss._force_float32
    with pytest.raises(ValueError, match="nonnegative"):
        NativeRNNTLoss(blank=8, fastemit_lambda=-0.01)

    monkeypatch.setattr(
        "nemo.collections.asr.losses.native_rnnt.TRITON_AVAILABLE", False
    )
    with pytest.raises(RuntimeError, match="Triton is required"):
        NativeRNNTLoss(blank=2)(
            torch.randn(1, 2, 2, 3),
            torch.ones(1, 1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
        )


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize(("blank", "vocab"), [(0, 8), (7, 8), (0, 9), (8, 9)])
def test_native_rnnt_matches_torch_loss_and_gradient(blank, vocab):
    logits, labels, source_lengths, target_lengths = _inputs(blank, vocab=vocab)
    original_logits = logits.detach().clone()
    reference_logits = logits.detach().clone().requires_grad_(True)

    native_loss = NativeRNNTLoss(blank)(logits, labels, source_lengths, target_lengths)
    reference_loss = _rnnt_loss_torch(
        reference_logits, labels, source_lengths, target_lengths, blank
    )
    native_grad = torch.autograd.grad(native_loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(native_loss, reference_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native_grad, reference_grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(logits, original_logits, atol=0.0, rtol=0.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_native_rnnt_can_reuse_private_logits_for_gradient():
    logits, labels, source_lengths, target_lengths = _inputs(blank=7)
    reference_logits = logits.detach().clone().requires_grad_(True)
    loss = NativeRNNTLoss(7)(
        logits, labels, source_lengths, target_lengths, reuse_logits_for_grad=True
    )
    reference_loss = NativeRNNTLoss(7)(
        reference_logits, labels, source_lengths, target_lengths
    )
    grad = torch.autograd.grad(loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(loss, reference_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(grad, reference_grad, atol=0.0, rtol=0.0)
    assert logits.data_ptr() == grad.data_ptr()


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize(
    ("fastemit_lambda", "clamp"), [(0.0, -1.0), (0.01, -1.0), (0.01, 0.02)]
)
@pytest.mark.parametrize("blank", [0, 7])
def test_native_rnnt_matches_numba_fastemit_and_clamp(fastemit_lambda, clamp, blank):
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    logits, labels, source_lengths, target_lengths = _inputs(blank=blank)
    reference_logits = logits.detach().clone().requires_grad_(True)
    native = NativeRNNTLoss(blank, fastemit_lambda=fastemit_lambda, clamp=clamp)
    reference = RNNTLossNumba(
        blank=blank,
        reduction="none",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
    )

    native_loss = native(logits, labels, source_lengths, target_lengths)
    reference_loss = reference(reference_logits, labels, source_lengths, target_lengths)
    native_grad = torch.autograd.grad(native_loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(native_loss, reference_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native_grad, reference_grad, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_rnnt_mixed_precision_matches_float32(dtype):
    logits, labels, source_lengths, target_lengths = _inputs(blank=7, dtype=dtype)
    reference_logits = logits.detach().float().requires_grad_(True)

    loss = NativeRNNTLoss(7)(logits, labels, source_lengths, target_lengths)
    reference_loss = NativeRNNTLoss(7)(
        reference_logits, labels, source_lengths, target_lengths
    )
    grad = torch.autograd.grad(loss.sum(), logits)[0].float()
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(loss, reference_loss, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
    relative_error = torch.linalg.vector_norm(
        grad - reference_grad
    ) / torch.linalg.vector_norm(reference_grad)
    assert relative_error < 2e-3


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_native_rnnt_target_limit_fails_before_extraction():
    loss = NativeRNNTLoss(blank=2)
    logits = torch.empty(1, 1, MAX_TARGET_TOKENS + 2, 3, device="cuda")
    labels = torch.zeros(1, MAX_TARGET_TOKENS + 1, dtype=torch.long, device="cuda")
    lengths = torch.ones(1, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match=str(MAX_TARGET_TOKENS)):
        loss(logits, labels, lengths, torch.full_like(lengths, MAX_TARGET_TOKENS + 1))


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize(
    "reduction", [None, "sum", "mean_batch", "mean", "mean_volume"]
)
def test_native_rnnt_reductions(reduction):
    logits, labels, source_lengths, _ = _inputs(blank=7)
    target_lengths = torch.tensor([4, 2, 1], device="cuda")
    per_sample = NativeRNNTLoss(7)(logits, labels, source_lengths, target_lengths)
    if reduction is None:
        expected = per_sample
    elif reduction == "sum":
        expected = per_sample.sum()
    elif reduction == "mean_batch":
        expected = per_sample.mean()
    elif reduction == "mean":
        expected = (per_sample / target_lengths).mean()
    else:
        expected = per_sample.sum() / target_lengths.sum()

    actual = RNNTLoss(num_classes=7, reduction=reduction, loss_name="native_rnnt")(
        log_probs=logits,
        targets=labels,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _make_joint(fused_batch_size, activation="relu", log_softmax=False):
    return RNNTJoint(
        jointnet={
            "encoder_hidden": 6,
            "pred_hidden": 7,
            "joint_hidden": 8,
            "activation": activation,
            "dropout": 0.0,
        },
        num_classes=7,
        log_softmax=log_softmax,
        fuse_loss_wer=True,
        fused_batch_size=fused_batch_size,
    ).cuda()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("fused_batch_size", [1, 2, 4])
def test_native_rnnt_fused_batch_equivalence(fused_batch_size):
    torch.manual_seed(11)
    reference_joint = _make_joint(4)
    joint = _make_joint(fused_batch_size)
    joint.load_state_dict(reference_joint.state_dict())
    loss = RNNTLoss(num_classes=7, reduction="mean_batch", loss_name="native_rnnt")
    reference_loss = RNNTLoss(
        num_classes=7, reduction="mean_batch", loss_name="native_rnnt"
    )
    joint.set_loss(loss)
    reference_joint.set_loss(reference_loss)
    joint.set_wer(object())
    reference_joint.set_wer(object())

    encoder = torch.randn(4, 6, 7, device="cuda", requires_grad=True)
    predictor = torch.randn(4, 7, 5, device="cuda", requires_grad=True)
    reference_encoder = encoder.detach().clone().requires_grad_(True)
    reference_predictor = predictor.detach().clone().requires_grad_(True)
    source_lengths = torch.tensor([7, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([4, 3, 2, 0], device="cuda")
    labels = torch.randint(0, 7, (4, 4), device="cuda")

    value = joint(
        encoder_outputs=encoder,
        decoder_outputs=predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    reference_value = reference_joint(
        encoder_outputs=reference_encoder,
        decoder_outputs=reference_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    gradients = torch.autograd.grad(value, (encoder, predictor, *joint.parameters()))
    reference_gradients = torch.autograd.grad(
        reference_value,
        (reference_encoder, reference_predictor, *reference_joint.parameters()),
    )

    torch.testing.assert_close(value, reference_value, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_matches_dense_native_loss_and_gradients(dtype):
    torch.manual_seed(19)
    dense_joint = _make_joint(4).to(dtype)
    flash_joint = _make_joint(4).to(dtype)
    flash_joint.load_state_dict(dense_joint.state_dict())
    dense_loss = NativeRNNTLoss(blank=7, fastemit_lambda=0.01)
    flash_loss = NativeRNNTLoss(blank=7, fastemit_lambda=0.01)
    encoder = torch.randn(4, 6, 7, device="cuda", dtype=dtype, requires_grad=True)
    predictor = torch.randn(4, 7, 5, device="cuda", dtype=dtype, requires_grad=True)
    flash_encoder = encoder.detach().clone().requires_grad_(True)
    flash_predictor = predictor.detach().clone().requires_grad_(True)
    source_lengths = torch.tensor([7, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([4, 3, 2, 0], device="cuda")
    labels = torch.randint(0, 7, (4, 4), device="cuda")

    logits = dense_joint.joint(encoder.transpose(1, 2), predictor.transpose(1, 2))
    dense_value = dense_loss(logits, labels, source_lengths, target_lengths).mean()
    flash_value = flash_rnnt_loss_from_joint(
        flash_joint,
        flash_encoder,
        flash_predictor,
        labels,
        source_lengths,
        target_lengths,
        flash_loss,
        workspace_batch_size=2,
        state_budget=40,
    ).mean()
    dense_gradients = torch.autograd.grad(
        dense_value, (encoder, predictor, *dense_joint.parameters())
    )
    flash_gradients = torch.autograd.grad(
        flash_value, (flash_encoder, flash_predictor, *flash_joint.parameters())
    )

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(flash_value, dense_value, atol=atol, rtol=rtol)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize(
    ("log_softmax", "expected_reuse"), [(False, True), (None, True), (True, False)]
)
def test_native_rnnt_only_reuses_final_linear_output(
    monkeypatch, log_softmax, expected_reuse
):
    joint = _make_joint(1, log_softmax=log_softmax)
    loss = RNNTLoss(num_classes=7, reduction="mean_batch", loss_name="native_rnnt")
    observed = []
    native_forward = loss._loss.forward

    def record_reuse(*args, **kwargs):
        observed.append(kwargs["reuse_logits_for_grad"])
        return native_forward(*args, **kwargs)

    monkeypatch.setattr(loss._loss, "forward", record_reuse)
    joint.set_loss(loss)
    joint.set_wer(object())
    joint(
        encoder_outputs=torch.randn(1, 6, 3, device="cuda"),
        decoder_outputs=torch.randn(1, 7, 3, device="cuda"),
        encoder_lengths=torch.tensor([3], device="cuda"),
        transcripts=torch.randint(0, 7, (1, 2), device="cuda"),
        transcript_lengths=torch.tensor([2], device="cuda"),
    )
    assert observed == [expected_reuse]
