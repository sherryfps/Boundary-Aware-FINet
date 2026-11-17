"""Training infrastructure tests; synthetic unit fixtures are not model results."""

from dataclasses import replace
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from src.finet_adapter import CheckpointError, FINetPredictor
from src.training import (TrainingConfig, restore_checkpoint, run_baseline, save_checkpoint,
                          structure_loss, run_signature)
from src.training_data import (FINetTrainingDataset, audit_pairs, create_manifest, load_manifest,
                               manifest_hash)
from train import structure_loss as original_structure_loss
from utils.dataloader_freq import TrainDataset as OriginalTrainDataset
from utils.LRScheduler import CosineDecay


def make_unit_pairs(root: Path, count=6):
    (root / "Imgs").mkdir(parents=True)
    (root / "GT").mkdir()
    for index in range(count):
        name = f"COD10K-unit-{index:03d}"
        Image.new("RGB", (64, 80), (index * 30, 20, 40)).save(root / "Imgs" / f"{name}.png")
        mask = np.zeros((80, 64), dtype=np.uint8)
        mask[10:40, 20:40] = 255
        Image.fromarray(mask).save(root / "GT" / f"{name}.png")


def test_original_loss_is_preserved_and_differentiable():
    assert structure_loss is original_structure_loss
    logits = [torch.linspace(-2, 2, 64 * 64).reshape(1, 1, 64, 64).clone().requires_grad_() for _ in range(4)]
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:40, 10:30] = 1
    loss = sum(structure_loss(output, mask) for output in logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(output.grad is not None and torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
               for output in logits)


def test_scalar_bce_source_behavior_is_not_silently_corrected():
    logits = torch.linspace(-3, 3, 64 * 64).reshape(1, 1, 64, 64)
    mask = torch.zeros_like(logits)
    mask[:, :, :25, :20] = 1
    weight = 1 + 5 * (torch.nn.functional.avg_pool2d(mask, 31, 1, 15) - mask).abs()
    probability = logits.sigmoid()
    intersection = (probability * mask * weight).sum((2, 3))
    union = ((probability + mask) * weight).sum((2, 3))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask) + (
        1 - (intersection + 1) / (union - intersection + 1)).mean()
    torch.testing.assert_close(structure_loss(logits, mask), expected)


def test_scheduler_keeps_original_200_epoch_formula_and_restores():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=2.6e-4)
    scheduler = CosineDecay(optimizer, 2.6e-4, 1e-7, 200)
    for step in range(1, 8):
        scheduler.step()
        expected = (2.6e-4 - 1e-7) / 2 * math.cos(math.pi / 199 * step) + (2.6e-4 + 1e-7) / 2
        assert scheduler.get_lr() == pytest.approx(expected)
    other = CosineDecay(optimizer, 2.6e-4, 1e-7, 200)
    other.load_state_dict(scheduler.state_dict())
    other.step()
    scheduler.step()
    assert other.state_dict() == scheduler.state_dict()


def test_single_epoch_scheduler_bug_is_fixed():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = CosineDecay(optimizer, 0.01, 0.001, 1)
    scheduler.step()
    assert scheduler.get_lr() == 0.01
    with pytest.raises(ValueError, match="positive"):
        CosineDecay(optimizer, 0.01, 0.001, 0)
    with pytest.raises(ValueError, match="mismatch"):
        CosineDecay(optimizer, 0.01, 0.001, 5).load_state_dict(scheduler.state_dict())


def test_audit_split_reproducibility_and_content_fingerprint(tmp_path):
    make_unit_pairs(tmp_path)
    manifest = create_manifest(tmp_path, validation_fraction=1/3, allow_small=True)
    assert manifest == create_manifest(tmp_path, validation_fraction=1/3, allow_small=True)
    assert len(manifest["train_ids"]) == 4 and len(manifest["validation_ids"]) == 2
    assert not set(manifest["train_ids"]) & set(manifest["validation_ids"])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_manifest(path, tmp_path) == manifest
    with pytest.raises(ValueError, match="3040"):
        create_manifest(tmp_path)
    changed_image = tmp_path / manifest["records"][0]["image"]
    Image.new("RGB", (64, 80), "white").save(changed_image)
    with pytest.raises(ValueError, match="contents changed"):
        load_manifest(path, tmp_path)


def test_audit_rejects_wrong_pairing_and_soft_original_masks(tmp_path):
    make_unit_pairs(tmp_path)
    mask_path = next((tmp_path / "GT").glob("*.png"))
    mask_path.rename(tmp_path / "GT" / "different.png")
    with pytest.raises(ValueError, match="matching"):
        audit_pairs(tmp_path)
    (tmp_path / "GT" / "different.png").rename(mask_path)
    Image.new("L", (64, 80), 127).save(mask_path)
    with pytest.raises(ValueError, match="binary"):
        audit_pairs(tmp_path)


def test_manifest_overlap_and_tampering_are_rejected(tmp_path):
    make_unit_pairs(tmp_path)
    manifest = create_manifest(tmp_path, allow_small=True)
    path = tmp_path / "split.json"
    manifest["train_ids"].append(manifest["validation_ids"][0])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_manifest(path, tmp_path)
    manifest["fingerprint"] = manifest_hash(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        load_manifest(path, tmp_path)


def test_training_dataset_reuses_original_preprocessing_outside_cwd(tmp_path, monkeypatch):
    root = tmp_path / "pairs"
    make_unit_pairs(root)
    records = audit_pairs(root)
    monkeypatch.chdir(tmp_path)
    reference = OriginalTrainDataset(str(root / "Imgs"), str(root / "GT"), 384,
                                    rVFlip=False, rCrop=False, rRotate=False, colorEnhance=False, rPeper=False)
    actual = FINetTrainingDataset(root, records, [record["id"] for record in records], augment=False)
    for value, expected in zip(actual[0], reference[0]):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_epoch_resume_preserves_adam_scheduler_rng_and_next_update(tmp_path):
    config = TrainingConfig(device="cpu", num_workers=0)
    manifest = {"fingerprint": "unit-test-split", "smoke_only": True}
    generator = torch.Generator().manual_seed(55)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineDecay(optimizer, config.learning_rate, config.min_lr, config.epochs)
    inputs = torch.tensor([[1.0, 2.0]])

    def step(current_model, current_optimizer):
        current_optimizer.zero_grad()
        current_model(inputs).square().sum().backward()
        current_optimizer.step()

    step(model, optimizer)
    scheduler.step()
    path = tmp_path / "unit_resume.bin"
    save_checkpoint(path, model, optimizer, scheduler, generator, config, manifest, completed_epochs=1, global_step=1)
    expected_rng = (random.random(), np.random.rand(), torch.rand(1), torch.randperm(5, generator=generator))
    step(model, optimizer)
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=config.learning_rate)
    restored_scheduler = CosineDecay(restored_optimizer, config.learning_rate, config.min_lr, config.epochs)
    payload = restore_checkpoint(path, restored, restored_optimizer, restored_scheduler, generator, config, manifest)
    assert payload["completed_epochs"] == 1 and payload["global_step"] == 1
    assert random.random() == expected_rng[0] and np.random.rand() == expected_rng[1]
    torch.testing.assert_close(torch.rand(1), expected_rng[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.randperm(5, generator=generator), expected_rng[3], rtol=0, atol=0)
    step(restored, restored_optimizer)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint(path, restored, restored_optimizer, restored_scheduler, generator,
                           replace(config, batch_size=8), manifest)


@pytest.mark.parametrize("status", ["initialization", "smoke_test"])
def test_inference_rejects_engineering_checkpoint(tmp_path, status):
    path = tmp_path / "unit_status.bin"
    torch.save({"training_status": status, "model_state_dict": {"a": torch.ones(1)}}, path)
    with pytest.raises(CheckpointError, match="not a trained segmentation"):
        FINetPredictor(path)


def test_malformed_checkpoint_provenance_is_rejected(tmp_path):
    path = tmp_path / "unit_invalid_metadata.bin"
    torch.save({"training_status": {"invalid": True}, "model_state_dict": {"a": torch.ones(1)}}, path)
    with pytest.raises(CheckpointError, match="must be a string"):
        FINetPredictor(path)


def test_full_training_guard_and_invalid_batch():
    with pytest.raises(ValueError, match=">=2"):
        TrainingConfig(batch_size=1).validate()
    with pytest.raises(ValueError, match="confirmed CUDA"):
        run_baseline(TrainingConfig(device="cpu"))
    import train_baseline
    with pytest.raises(SystemExit) as exc:
        train_baseline.main(["--mode", "train"])
    assert exc.value.code == 2


def test_colab_notebook_is_valid_python_and_full_run_is_disabled():
    path = Path(__file__).resolve().parents[1] / "notebooks/baseline_training_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
    assert any("RUN_FULL_TRAINING = False" in "".join(cell["source"]) for cell in notebook["cells"])
