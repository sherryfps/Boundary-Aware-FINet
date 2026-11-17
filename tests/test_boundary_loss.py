"""Analytic boundary and gradient checks, not experimental model results."""

from dataclasses import replace
import pytest
import torch
from src.boundary_loss import boundary_loss, finet_objective, morphological_boundary
from src.training import TrainingConfig, run_signature
from train import structure_loss


def test_boundary_matches_manual_square_and_has_no_frame_edges():
    mask = torch.zeros(1, 1, 9, 9)
    mask[:, :, 3:6, 3:6] = 1
    expected = torch.zeros_like(mask)
    expected[:, :, 2:7, 2:7] = 1
    expected[:, :, 4, 4] = 0
    torch.testing.assert_close(morphological_boundary(mask), expected, rtol=0, atol=0)
    for value in (0.0, 1.0):
        assert morphological_boundary(torch.full_like(mask, value)).count_nonzero() == 0


def test_boundary_gradient_flows_to_all_four_heads():
    torch.manual_seed(42)
    outputs = [torch.randn(2, 1, 32, 32, requires_grad=True) for _ in range(4)]
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 7:20, 10:25] = 1
    boundary = sum(boundary_loss(output, mask) for output in outputs)
    boundary.backward()
    assert torch.isfinite(boundary)
    for output in outputs:
        assert torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0


def test_lambda_zero_is_exact_original_loss_and_gradient():
    outputs = [torch.randn(2, 1, 32, 32, requires_grad=True) for _ in range(4)]
    mask = torch.rand(2, 1, 32, 32)
    actual, structure, boundary, _ = finet_objective(outputs, mask, 0)
    expected = sum(structure_loss(output, mask) for output in outputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual is structure and boundary.item() == 0
    actual_grad = torch.autograd.grad(actual, outputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, outputs)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('value', [-100.0, 100.0])
@pytest.mark.parametrize('foreground', [0.0, 1.0])
def test_extreme_logits_and_empty_or_full_masks_are_finite(value, foreground):
    logits = torch.full((2, 1, 12, 12), value, requires_grad=True)
    loss = boundary_loss(logits, torch.full_like(logits, foreground))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_boundary_configuration_changes_run_signature():
    baseline = TrainingConfig()
    proposed = replace(baseline, boundary_lambda=0.1)
    proposed.validate()
    manifest = {'fingerprint': 'unit'}
    assert run_signature(baseline, manifest) != run_signature(proposed, manifest)
    for value in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='lambda'):
            replace(baseline, boundary_lambda=value).validate()


@pytest.mark.parametrize('kernel', [0, 1, 2, 4, 3.5])
def test_invalid_kernels_fail(kernel):
    with pytest.raises(ValueError, match='kernel'):
        morphological_boundary(torch.zeros(1, 1, 8, 8), kernel)
