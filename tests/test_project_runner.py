"""Protocol gates only; these tests do not simulate research results."""

from dataclasses import asdict

import pytest

from scripts.run_colab_project import is_finished, matched_boundary_config
from src.training import TrainingConfig


def test_boundary_runner_changes_only_objective_and_locations():
    original = asdict(TrainingConfig(initial_checkpoint="original-initial.pth"))
    boundary = matched_boundary_config(original, "boundary", "shared-initial.pth")
    assert {k for k in original if original[k] != boundary[k]} == {
        "output_dir", "initial_checkpoint", "boundary_lambda"}
    assert original["boundary_lambda"] == 0
    assert boundary["boundary_lambda"] == 0.1
    assert boundary["epochs"] == original["epochs"] == 200


@pytest.mark.parametrize("coefficient", [0, -1, float("nan"), float("inf")])
def test_boundary_runner_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError):
        matched_boundary_config(asdict(TrainingConfig()), "boundary", "initial.pth", coefficient)


@pytest.mark.parametrize("completed,status,expected", [
    (0, "baseline_training", False), (7, "baseline_training", False),
    (199, "baseline_training", False), (200, "baseline_training", True),
    (201, "baseline_training", False), (200, "smoke_test", False),
    (200, "boundary_training", False)])
def test_runner_requires_exact_full_budget_and_correct_checkpoint_type(completed, status, expected):
    summary = {"training_status": status, "completed_epochs": completed, "config": {"epochs": 200}}
    assert is_finished(summary, TrainingConfig()) is expected
