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
from omegaconf import OmegaConf

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.pruned_rnnt import (
    MAX_TARGET_TOKENS,
    get_prune_ranges,
    get_smoothed_rnnt_logprobs,
    pruned_logprobs_triton,
    rnnt_loss_triton,
)
from nemo.collections.common.parts import adapter_modules
from nemo.core.utils.optional_libs import K2_AVAILABLE, TRITON_AVAILABLE


def _rnnt_loss_torch(
    target_scores,
    blank_scores,
    source_lengths,
    target_lengths,
    return_occupations=False,
):
    losses = []
    target_occupations = torch.zeros_like(target_scores, dtype=torch.float32)
    blank_occupations = torch.zeros_like(blank_scores, dtype=torch.float32)
    for batch_idx in range(target_scores.shape[0]):
        source_len = int(source_lengths[batch_idx])
        target_len = int(target_lengths[batch_idx])
        alpha_rows = []
        alpha = torch.cat(
            (
                target_scores.new_zeros(1),
                target_scores.new_full((target_len,), -torch.inf),
            )
        )
        for time_idx in range(source_len):
            if time_idx:
                alpha = alpha + blank_scores[batch_idx, time_idx - 1, : target_len + 1]
            values = [alpha[0]]
            for symbol_idx in range(1, target_len + 1):
                values.append(
                    torch.logaddexp(
                        alpha[symbol_idx],
                        values[-1] + target_scores[batch_idx, time_idx, symbol_idx - 1],
                    )
                )
            alpha = torch.stack(values)
            alpha_rows.append(alpha)

        log_likelihood = alpha[-1] + blank_scores[batch_idx, source_len - 1, target_len]
        losses.append(-log_likelihood)
        if not return_occupations:
            continue

        with torch.no_grad():
            beta_next = None
            for time_idx in range(source_len - 1, -1, -1):
                if time_idx == source_len - 1:
                    base = blank_scores.new_full((target_len + 1,), -torch.inf)
                    base[target_len] = blank_scores[batch_idx, time_idx, target_len]
                else:
                    base = (
                        blank_scores[batch_idx, time_idx, : target_len + 1] + beta_next
                    )
                beta_values = [base[target_len]]
                for symbol_idx in range(target_len - 1, -1, -1):
                    beta_values.append(
                        torch.logaddexp(
                            base[symbol_idx],
                            target_scores[batch_idx, time_idx, symbol_idx]
                            + beta_values[-1],
                        )
                    )
                beta = torch.stack(list(reversed(beta_values)))
                alpha = alpha_rows[time_idx]
                if target_len:
                    target_occupations[batch_idx, time_idx, :target_len] = torch.exp(
                        alpha[:target_len]
                        + target_scores[batch_idx, time_idx, :target_len]
                        + beta[1:]
                        - log_likelihood
                    )
                if time_idx == source_len - 1:
                    blank_occupations[batch_idx, time_idx, target_len] = 1.0
                else:
                    blank_occupations[batch_idx, time_idx, : target_len + 1] = (
                        torch.exp(
                            alpha
                            + blank_scores[batch_idx, time_idx, : target_len + 1]
                            + beta_next
                            - log_likelihood
                        )
                    )
                beta_next = beta

    loss = torch.stack(losses)
    if return_occupations:
        return loss, target_occupations, blank_occupations
    return loss


def _make_joint(vocab_size=9, dropout=0.0, fused_batch_size=2):
    return RNNTJoint(
        jointnet={
            "encoder_hidden": 6,
            "pred_hidden": 7,
            "joint_hidden": 8,
            "activation": "relu",
            "dropout": dropout,
        },
        num_classes=vocab_size - 1,
        fuse_loss_wer=True,
        fused_batch_size=fused_batch_size,
    )


@pytest.mark.unit
def test_smoothed_scores_match_dense_simple_joiner():
    torch.manual_seed(0)
    batch, source, target, vocab = 2, 5, 3, 7
    am = torch.randn(batch, source, vocab, dtype=torch.float64)
    lm = torch.randn(batch, target + 1, vocab, dtype=torch.float64)
    labels = torch.randint(0, vocab - 1, (batch, target))
    target_scores, blank_scores = get_smoothed_rnnt_logprobs(
        lm, am, labels, vocab - 1, 0.0, 0.0
    )

    dense = (am.unsqueeze(2) + lm.unsqueeze(1)).log_softmax(dim=-1)
    expected_target = torch.gather(
        dense[:, :, :target],
        dim=3,
        index=labels[:, None, :, None].expand(batch, source, target, 1),
    ).squeeze(-1)
    expected_blank = dense[..., vocab - 1]
    assert torch.allclose(target_scores, expected_target.float(), atol=1e-5)
    assert torch.allclose(blank_scores, expected_blank.float(), atol=1e-5)


@pytest.mark.unit
def test_torch_recursion_gradients_and_blank_only():
    torch.manual_seed(1)
    target_scores = torch.randn(3, 6, 4, requires_grad=True)
    blank_scores = torch.randn(3, 6, 5, requires_grad=True)
    source_lengths = torch.tensor([6, 4, 3])
    target_lengths = torch.tensor([4, 2, 0])
    loss, target_occupation, blank_occupation = _rnnt_loss_torch(
        target_scores,
        blank_scores,
        source_lengths,
        target_lengths,
        return_occupations=True,
    )
    target_grad, blank_grad = torch.autograd.grad(
        loss.sum(), (target_scores, blank_scores)
    )
    assert torch.allclose(target_grad, -target_occupation, atol=1e-5)
    assert torch.allclose(blank_grad, -blank_occupation, atol=1e-5)
    assert torch.isfinite(loss).all()
    assert blank_occupation[2, :3, 0].sum() == 3


@pytest.mark.unit
def test_pruning_ranges_obey_constraints():
    torch.manual_seed(2)
    target_scores = torch.randn(2, 9, 5)
    blank_scores = torch.randn(2, 9, 6)
    source_lengths = torch.tensor([9, 7])
    target_lengths = torch.tensor([5, 3])
    _, target_occupation, blank_occupation = _rnnt_loss_torch(
        target_scores,
        blank_scores,
        source_lengths,
        target_lengths,
        return_occupations=True,
    )
    ranges = get_prune_ranges(
        target_occupation,
        blank_occupation,
        source_lengths,
        target_lengths,
        prune_range=3,
    )
    starts = ranges[..., 0]
    assert torch.all(starts[:, 1:] >= starts[:, :-1])
    assert torch.all(starts[:, 1:] - starts[:, :-1] < ranges.shape[-1])
    assert torch.all(ranges[..., 1:] == ranges[..., :-1] + 1)
    assert starts[0, 0] == 0
    assert starts[0, source_lengths[0] - 1] == target_lengths[0] - ranges.shape[-1] + 1


@pytest.mark.unit
def test_public_configuration_and_full_joint_error():
    loss = RNNTLoss(
        num_classes=8,
        loss_name="pruned_rnnt",
        loss_kwargs={"prune_range": 5, "warmup_steps": 10},
    )
    assert loss.requires_joint_inputs
    with pytest.raises(RuntimeError, match="full joint tensor is already too late"):
        loss(
            log_probs=torch.randn(1, 2, 2, 9),
            targets=torch.ones(1, 1, dtype=torch.long),
            input_lengths=torch.tensor([2]),
            target_lengths=torch.tensor([1]),
        )


@pytest.mark.unit
def test_schedule_and_joint_validation():
    loss = RNNTLoss(
        num_classes=8,
        loss_name="pruned_rnnt",
        loss_kwargs={
            "simple_loss_scale": 0.5,
            "initial_simple_loss_scale": 1.0,
            "initial_pruned_loss_scale": 0.1,
            "warmup_steps": 100,
        },
    )
    joint = _make_joint()
    loss.bind_joint(joint)
    joint.set_loss(loss)
    loss.set_step(50)
    assert loss._loss._current_scales(training=True) == pytest.approx((0.75, 0.55))
    loss.set_step(100)
    assert loss._loss._current_scales(training=True) == (0.5, 1.0)
    assert loss._loss._current_scales(training=False) == (0.5, 1.0)
    assert loss._loss.simple_encoder.in_features == joint.encoder_hidden
    assert loss._loss.simple_predictor.in_features == joint.pred_hidden

    default_loss = RNNTLoss(
        num_classes=8, loss_name="pruned_rnnt", loss_kwargs={"warmup_steps": 100}
    )
    default_loss.set_step(50)
    assert default_loss._loss._current_scales(training=True) == pytest.approx(
        (0.75, 0.55)
    )

    zero_initial_loss = RNNTLoss(
        num_classes=8,
        loss_name="pruned_rnnt",
        loss_kwargs={"initial_pruned_loss_scale": 0.0, "warmup_steps": 100},
    )
    zero_initial_loss.set_step(50)
    assert zero_initial_loss._loss._current_scales(training=True) == (0.75, 0.0)


@pytest.mark.unit
def test_pre_joint_loss_processes_full_batch_once():
    class RecordingPreJointLoss(torch.nn.Module):
        requires_joint_inputs = True

        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward_from_joint(
            self,
            joint,
            encoder_outputs,
            predictor_outputs,
            targets,
            input_lengths,
            target_lengths,
        ):
            self.batch_sizes.append(encoder_outputs.shape[0])
            return encoder_outputs.sum() * 0.0 + 3.0

    class RecordingWer:
        _to_sync = True

        def __init__(self):
            self.batch_sizes = []

        def update(self, predictions, **kwargs):
            self.batch_sizes.append(predictions.shape[0])

        def get_hypotheses(self):
            return ["hypothesis"] * self.batch_sizes[-1]

        def compute(self):
            return torch.tensor(0.25), torch.tensor(1), torch.tensor(4)

        def reset(self):
            pass

    joint = _make_joint(fused_batch_size=1)
    loss = RecordingPreJointLoss()
    wer = RecordingWer()
    joint.set_loss(loss)
    joint.set_wer(wer)

    value, wer, wer_num, wer_denom = joint(
        encoder_outputs=torch.randn(4, 6, 5),
        decoder_outputs=torch.randn(4, 7, 4),
        encoder_lengths=torch.tensor([5, 4, 3, 2]),
        transcripts=torch.randint(0, 8, (4, 3)),
        transcript_lengths=torch.tensor([3, 2, 2, 1]),
        compute_wer=True,
        keep_hypotheses=True,
    )

    assert value == 3.0
    assert (wer, wer_num, wer_denom) == (
        torch.tensor(0.25),
        torch.tensor(4),
        torch.tensor(16),
    )
    assert loss.batch_sizes == [4]
    assert joint.wer.batch_sizes == [1] * 4
    assert joint.get_hypotheses() == ["hypothesis"] * 4


@pytest.mark.unit
def test_simple_joiner_checkpoint_round_trip_without_executing_loss():
    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.joint = _make_joint()
            self.loss = RNNTLoss(num_classes=8, loss_name="pruned_rnnt")
            self.loss.bind_joint(self.joint)
            self.joint.set_loss(self.loss)

    torch.manual_seed(9)
    source = Container()
    state = source.state_dict()
    restored = Container()
    restored.load_state_dict(state, strict=True)
    assert torch.equal(
        source.loss._loss.simple_encoder.weight,
        restored.loss._loss.simple_encoder.weight,
    )
    assert torch.equal(
        source.loss._loss.simple_predictor.weight,
        restored.loss._loss.simple_predictor.weight,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        (None, torch.tensor([2.0, 6.0])),
        ("sum", torch.tensor(8.0)),
        ("mean_batch", torch.tensor(4.0)),
        ("mean", torch.tensor(2.0)),
        ("mean_volume", torch.tensor(2.0)),
    ],
)
def test_reductions(reduction, expected):
    loss = RNNTLoss(num_classes=8, reduction=reduction, loss_name="pruned_rnnt")
    actual = loss.reduce(torch.tensor([2.0, 6.0]), torch.tensor([1, 3]))
    assert torch.equal(actual, expected)


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
def test_warmup_skips_pruned_pass_and_connects_joint_gradients():
    torch.manual_seed(11)
    device = torch.device("cuda")
    joint = _make_joint(vocab_size=17).to(device=device, dtype=torch.bfloat16).train()
    loss = RNNTLoss(
        num_classes=16,
        reduction="mean_batch",
        loss_name="pruned_rnnt",
        loss_kwargs={
            "prune_range": 5,
            "warmup_steps": 100,
            "initial_pruned_loss_scale": 0.0,
        },
    ).to(device)
    loss.bind_joint(joint)
    loss.set_step(50)

    joint_calls = 0

    def count_joint_call(_module, _inputs):
        nonlocal joint_calls
        joint_calls += 1

    handle = joint.joint_net.register_forward_pre_hook(count_joint_call)
    value = loss.forward_from_joint(
        joint=joint,
        encoder_outputs=torch.randn(
            2, 6, 6, device=device, dtype=torch.bfloat16, requires_grad=True
        ),
        predictor_outputs=torch.randn(
            2, 4, 7, device=device, dtype=torch.bfloat16, requires_grad=True
        ),
        targets=torch.randint(0, 16, (2, 3), device=device),
        input_lengths=torch.tensor([6, 5], device=device),
        target_lengths=torch.tensor([3, 2], device=device),
    )
    value.backward()
    handle.remove()

    assert joint_calls == 0
    assert loss.diagnostics["pruned_loss"] == 0
    assert loss.diagnostics["pruned_loss_scale"] == 0
    assert all(
        parameter.grad is not None
        for parameter in joint.parameters()
        if parameter.requires_grad
    )
    assert torch.count_nonzero(joint.joint_net[-1].weight.grad) == 0


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
def test_target_limit_fails_before_simple_projection():
    device = torch.device("cuda")
    joint = _make_joint().to(device)
    loss = RNNTLoss(num_classes=8, loss_name="pruned_rnnt").to(device)
    loss.bind_joint(joint)
    projection_calls = 0

    def count_projection_call(_module, _inputs):
        nonlocal projection_calls
        projection_calls += 1

    handle = loss._loss.simple_encoder.register_forward_pre_hook(count_projection_call)
    target = MAX_TARGET_TOKENS + 1
    with pytest.raises(
        ValueError, match=f"at most {MAX_TARGET_TOKENS} padded target tokens"
    ):
        loss.forward_from_joint(
            joint=joint,
            encoder_outputs=torch.randn(1, 2, 6, device=device),
            predictor_outputs=torch.randn(1, target + 1, 7, device=device),
            targets=torch.zeros(1, target, dtype=torch.long, device=device),
            input_lengths=torch.tensor([2], device=device),
            target_lengths=torch.tensor([target], device=device),
        )
    handle.remove()
    assert projection_calls == 0


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("max_target", [4, 65])
def test_triton_recursion_forward_and_gradient_parity(max_target):
    torch.manual_seed(3)
    device = torch.device("cuda")
    target_scores = torch.randn(3, 7, max_target, device=device, requires_grad=True)
    blank_scores = torch.randn(3, 7, max_target + 1, device=device, requires_grad=True)
    source_lengths = torch.tensor([7, 5, 3], device=device)
    target_lengths = torch.tensor([max_target, max_target - 1, 0], device=device)

    expected = _rnnt_loss_torch(
        target_scores, blank_scores, source_lengths, target_lengths
    )
    expected_grad = torch.autograd.grad(
        expected.sum(), (target_scores, blank_scores), retain_graph=True
    )
    actual, _, _ = rnnt_loss_triton(
        target_scores, blank_scores, source_lengths, target_lengths
    )
    actual_grad = torch.autograd.grad(actual.sum(), (target_scores, blank_scores))
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(actual_grad[0], expected_grad[0], atol=2e-5)
    assert torch.allclose(actual_grad[1], expected_grad[1], atol=2e-5)


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("blank_id", [0, 8])
def test_unpruned_range_matches_dense_loss_and_gradients(blank_id):
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, source, target, vocab = 2, 6, 4, 9
    logits = torch.randn(
        batch, source, target + 1, vocab, device=device, requires_grad=True
    )
    if blank_id == 0:
        labels = torch.randint(1, vocab, (batch, target), device=device)
    else:
        labels = torch.randint(0, vocab - 1, (batch, target), device=device)
    source_lengths = torch.tensor([6, 5], device=device)
    target_lengths = torch.tensor([4, 3], device=device)
    ranges = (
        torch.arange(target + 1, device=device)
        .reshape(1, 1, -1)
        .expand(batch, source, -1)
    )

    actual_target, actual_blank = pruned_logprobs_triton(
        logits, ranges, labels, blank_id, source_lengths, target_lengths
    )
    actual, _, _ = rnnt_loss_triton(
        actual_target, actual_blank, source_lengths, target_lengths
    )
    actual_grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]

    log_probs = logits.log_softmax(dim=-1)
    expected_target = torch.gather(
        log_probs[:, :, :target],
        dim=3,
        index=labels[:, None, :, None].expand(batch, source, target, 1),
    ).squeeze(-1)
    expected_blank = log_probs[..., blank_id]
    expected = _rnnt_loss_torch(
        expected_target, expected_blank, source_lengths, target_lengths
    )
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(actual_grad, expected_grad, atol=2e-5)


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
def test_unpruned_full_loss_matches_dense_joint_and_gradients():
    torch.manual_seed(12)
    device = torch.device("cuda")
    batch, source, target, vocab = 2, 6, 4, 9
    joint = _make_joint(vocab_size=vocab).to(device).train()
    loss = RNNTLoss(
        num_classes=vocab - 1,
        reduction=None,
        loss_name="pruned_rnnt",
        loss_kwargs={
            "prune_range": target + 1,
            "simple_loss_scale": 0.0,
            "warmup_steps": 0,
        },
    ).to(device)
    loss.bind_joint(joint)

    encoder = torch.randn(batch, source, 6, device=device, requires_grad=True)
    predictor = torch.randn(batch, target + 1, 7, device=device, requires_grad=True)
    labels = torch.randint(0, vocab - 1, (batch, target), device=device)
    source_lengths = torch.tensor([source, source - 1], device=device)
    target_lengths = torch.tensor([target, target - 1], device=device)
    parameters = tuple(joint.parameters())

    actual = loss.forward_from_joint(
        joint,
        encoder,
        predictor,
        labels,
        source_lengths,
        target_lengths,
    )
    actual_grad = torch.autograd.grad(
        actual.sum(), (encoder, predictor, *parameters), retain_graph=True
    )

    logits = joint.joint(encoder, predictor)
    log_probs = logits.log_softmax(dim=-1)
    expected_target = torch.gather(
        log_probs[:, :, :target],
        dim=3,
        index=labels[:, None, :, None].expand(batch, source, target, 1),
    ).squeeze(-1)
    expected_blank = log_probs[..., vocab - 1]
    expected = _rnnt_loss_torch(
        expected_target, expected_blank, source_lengths, target_lengths
    )
    expected_grad = torch.autograd.grad(
        expected.sum(), (encoder, predictor, *parameters)
    )

    assert torch.allclose(actual, expected, atol=1e-5)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        assert torch.allclose(actual_value, expected_value, atol=3e-5)


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_joint_mixed_precision_parity(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("The CUDA device does not support BF16")
    torch.manual_seed(10)
    device = torch.device("cuda")
    batch, source, target, vocab = 2, 7, 4, 9
    labels = torch.randint(0, vocab - 1, (batch, target), device=device)
    source_lengths = torch.tensor([7, 5], device=device)
    target_lengths = torch.tensor([4, 3], device=device)
    ranges = (
        torch.arange(target + 1, device=device)
        .reshape(1, 1, -1)
        .expand(batch, source, -1)
    )

    fp32_logits = torch.randn(
        batch, source, target + 1, vocab, device=device, requires_grad=True
    )
    fp32_target, fp32_blank = pruned_logprobs_triton(
        fp32_logits, ranges, labels, vocab - 1, source_lengths, target_lengths
    )
    fp32_loss, _, _ = rnnt_loss_triton(
        fp32_target, fp32_blank, source_lengths, target_lengths
    )
    fp32_grad = torch.autograd.grad(fp32_loss.sum(), fp32_logits)[0]

    mixed_logits = fp32_logits.detach().to(dtype).requires_grad_(True)
    mixed_target, mixed_blank = pruned_logprobs_triton(
        mixed_logits, ranges, labels, vocab - 1, source_lengths, target_lengths
    )
    mixed_loss, _, _ = rnnt_loss_triton(
        mixed_target, mixed_blank, source_lengths, target_lengths
    )
    mixed_grad = torch.autograd.grad(mixed_loss.sum(), mixed_logits)[0].float()
    assert torch.allclose(mixed_loss, fp32_loss, atol=0.08, rtol=0.005)
    assert torch.allclose(mixed_grad, fp32_grad, atol=0.02, rtol=0.02)


@pytest.mark.unit
@pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="CUDA and Triton are required",
)
def test_fused_joint_bf16_backward_and_diagnostics():
    torch.manual_seed(4)
    device = torch.device("cuda")
    joint = _make_joint(vocab_size=17, dropout=0.1)
    adapter_cfg = OmegaConf.structured(
        adapter_modules.LinearAdapterConfig(
            in_features=joint.joint_hidden, dim=4, norm_position="pre"
        )
    )
    joint.add_adapter("pruned_test", cfg=adapter_cfg)
    joint = joint.to(device=device).train()
    loss = RNNTLoss(
        num_classes=16,
        reduction="mean_batch",
        loss_name="pruned_rnnt",
        loss_kwargs={"prune_range": 5, "warmup_steps": 100},
    ).to(device)
    loss.bind_joint(joint)
    joint.set_loss(loss)
    joint.set_wer(object())
    loss.set_step(100)
    encoder = torch.randn(4, 6, 20, device=device, requires_grad=True)
    predictor = torch.randn(4, 7, 8, device=device, requires_grad=True)
    targets = torch.randint(0, 16, (4, 7), device=device)
    source_lengths = torch.tensor([20, 18, 16, 12], device=device)
    target_lengths = torch.tensor([7, 5, 3, 0], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value, _, _, _ = joint(
            encoder_outputs=encoder,
            decoder_outputs=predictor,
            encoder_lengths=source_lengths,
            transcripts=targets,
            transcript_lengths=target_lengths,
            compute_wer=False,
        )
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(encoder.grad).all() and torch.count_nonzero(encoder.grad)
    assert torch.isfinite(predictor.grad).all() and torch.count_nonzero(predictor.grad)
    assert torch.isfinite(joint.joint_net[-1].weight.grad).all()
    adapter = joint.get_adapter_module("pruned_test")
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in adapter.parameters()
    )
    assert loss.diagnostics["simple_loss_scale"] == pytest.approx(torch.tensor(0.5))
    assert loss.diagnostics["pruned_loss_scale"] == pytest.approx(torch.tensor(1.0))


@pytest.mark.unit
@pytest.mark.skipif(
    not (K2_AVAILABLE and TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="k2, CUDA, and Triton are required",
)
@pytest.mark.parametrize(
    ("batch", "source", "target", "vocab"), [(2, 9, 5, 11), (3, 257, 129, 67)]
)
def test_pruning_ranges_match_k2(batch, source, target, vocab):
    import k2

    torch.manual_seed(5)
    device = torch.device("cuda")
    am = torch.randn(batch, source, vocab, device=device)
    lm = torch.randn(batch, target + 1, vocab, device=device)
    labels = torch.randint(0, vocab - 1, (batch, target), device=device)
    source_lengths = torch.linspace(source, source // 2, batch, device=device).long()
    target_lengths = torch.linspace(target, target // 2, batch, device=device).long()
    boundary = torch.stack(
        (
            torch.zeros_like(source_lengths),
            torch.zeros_like(source_lengths),
            target_lengths,
            source_lengths,
        ),
        dim=1,
    )
    target_scores, blank_scores = get_smoothed_rnnt_logprobs(
        lm, am, labels, vocab - 1, 0.25, 0.0
    )
    _, target_occupation, blank_occupation = rnnt_loss_triton(
        target_scores, blank_scores, source_lengths, target_lengths
    )
    actual = get_prune_ranges(
        target_occupation,
        blank_occupation,
        source_lengths,
        target_lengths,
        prune_range=3,
    )
    _, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
        lm,
        am,
        labels,
        vocab - 1,
        lm_only_scale=0.25,
        am_only_scale=0.0,
        boundary=boundary,
        reduction="none",
        return_grad=True,
    )
    expected_target_occupation = px_grad[:, :, :source].transpose(1, 2)
    expected_blank_occupation = py_grad.transpose(1, 2)
    assert torch.allclose(
        target_occupation, expected_target_occupation, atol=5e-4, rtol=5e-4
    ), torch.max(torch.abs(target_occupation - expected_target_occupation))
    assert torch.allclose(
        blank_occupation, expected_blank_occupation, atol=5e-4, rtol=5e-4
    ), torch.max(torch.abs(blank_occupation - expected_blank_occupation))
    expected = k2.get_rnnt_prune_ranges(px_grad, py_grad, boundary, 3)
    ranges_from_reference_occupations = get_prune_ranges(
        expected_target_occupation,
        expected_blank_occupation,
        source_lengths,
        target_lengths,
        prune_range=3,
    )
    assert torch.equal(ranges_from_reference_occupations, expected)
    assert torch.max(torch.abs(actual - expected)) <= 1


@pytest.mark.unit
@pytest.mark.skipif(
    not (K2_AVAILABLE and TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="k2, CUDA, and Triton are required",
)
@pytest.mark.parametrize(
    ("batch", "source", "target", "vocab", "prune_range"),
    [(2, 8, 5, 11, 3), (2, 129, 65, 257, 5)],
)
def test_fp32_loss_and_gradient_parity_with_k2(
    batch, source, target, vocab, prune_range
):
    import k2

    torch.manual_seed(8)
    device = torch.device("cuda")
    labels = torch.randint(0, vocab - 1, (batch, target), device=device)
    source_lengths = torch.linspace(
        source, source * 3 // 4, batch, device=device
    ).long()
    target_lengths = torch.linspace(
        target, target * 3 // 4, batch, device=device
    ).long()
    boundary = torch.stack(
        (
            torch.zeros_like(source_lengths),
            torch.zeros_like(source_lengths),
            target_lengths,
            source_lengths,
        ),
        dim=1,
    )

    native_am = torch.randn(batch, source, vocab, device=device, requires_grad=True)
    native_lm = torch.randn(batch, target + 1, vocab, device=device, requires_grad=True)
    reference_am = native_am.detach().clone().requires_grad_(True)
    reference_lm = native_lm.detach().clone().requires_grad_(True)
    native_target, native_blank = get_smoothed_rnnt_logprobs(
        native_lm, native_am, labels, vocab - 1, 0.25, 0.0
    )
    native_simple, target_occupation, blank_occupation = rnnt_loss_triton(
        native_target, native_blank, source_lengths, target_lengths
    )
    native_simple_grad = torch.autograd.grad(
        native_simple.sum(), (native_am, native_lm)
    )
    reference_simple = k2.rnnt_loss_smoothed(
        reference_lm,
        reference_am,
        labels,
        vocab - 1,
        lm_only_scale=0.25,
        am_only_scale=0.0,
        boundary=boundary,
        reduction="none",
    )
    reference_simple_grad = torch.autograd.grad(
        reference_simple.sum(), (reference_am, reference_lm)
    )
    assert torch.allclose(native_simple, reference_simple, atol=2e-5)
    for native_grad, reference_grad in zip(native_simple_grad, reference_simple_grad):
        relative_error = torch.linalg.vector_norm(
            native_grad - reference_grad
        ) / torch.linalg.vector_norm(reference_grad)
        assert relative_error < (3e-5 if target < 64 else 2e-3)

    ranges = get_prune_ranges(
        target_occupation,
        blank_occupation,
        source_lengths,
        target_lengths,
        prune_range=prune_range,
    )
    native_logits = torch.randn(
        batch, source, prune_range, vocab, device=device, requires_grad=True
    )
    reference_logits = native_logits.detach().clone().requires_grad_(True)
    pruned_target, pruned_blank = pruned_logprobs_triton(
        native_logits, ranges, labels, vocab - 1, source_lengths, target_lengths
    )
    native_pruned, _, _ = rnnt_loss_triton(
        pruned_target, pruned_blank, source_lengths, target_lengths
    )
    native_pruned_grad = torch.autograd.grad(native_pruned.sum(), native_logits)[0]
    reference_pruned = k2.rnnt_loss_pruned(
        reference_logits,
        labels,
        ranges,
        vocab - 1,
        boundary=boundary,
        reduction="none",
    )
    reference_pruned_grad = torch.autograd.grad(
        reference_pruned.sum(), reference_logits
    )[0]
    assert torch.allclose(native_pruned, reference_pruned, atol=2e-5)
    relative_error = torch.linalg.vector_norm(
        native_pruned_grad - reference_pruned_grad
    ) / torch.linalg.vector_norm(reference_pruned_grad)
    assert relative_error < (3e-5 if target < 64 else 2e-3)
