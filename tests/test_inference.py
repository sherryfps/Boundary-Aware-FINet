import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import models

from Model.FINet import FINet
from src.finet_adapter import CheckpointError, FINetPredictor, extract_state_dict, select_device
from src.preprocessing import FINetPreprocessor


def test_architecture_shapes_and_no_backbone_download(monkeypatch, patterned_image):
    factory = models.efficientnet_b0
    calls = []

    def offline_factory(*args, **kwargs):
        calls.append(kwargs["weights"])
        return factory(weights=None)

    monkeypatch.setattr(models, "efficientnet_b0", offline_factory)
    model = FINet(channels=(8, 24, 32, 64), pretrained_backbone=False).eval()
    assert calls == [None]
    default_model = FINet(channels=(8, 24, 32, 64))
    assert calls[-1] == models.EfficientNet_B0_Weights.DEFAULT
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in default_model.state_dict().items()}
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in default_model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 3_742_988
    assert len(model.state_dict()) == 965
    prepared = FINetPreprocessor()(patterned_image)
    # An explicitly UNTRAINED architecture check. No predictions are saved or used as demo evidence.
    with torch.inference_mode():
        outputs = model(prepared.rgb, prepared.high, prepared.low)
    assert len(outputs) == 4
    assert all(output.shape == (1, 1, 384, 384) and torch.isfinite(output).all() for output in outputs)


def test_missing_checkpoint_fails_before_model_construction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Should not construct a model for a missing checkpoint")
    monkeypatch.setattr("src.finet_adapter.FINet", forbidden)
    with pytest.raises(FileNotFoundError, match="official full"):
        FINetPredictor(tmp_path / "missing.pth")


def test_html_download_is_rejected(tmp_path):
    checkpoint = tmp_path / "download.html"
    checkpoint.write_text("<html>Sign in to download</html>", encoding="utf-8")
    with pytest.raises(CheckpointError, match="Cannot safely read"):
        FINetPredictor(checkpoint)


def test_incomplete_state_is_rejected(tmp_path):
    checkpoint = tmp_path / "invalid_state.bin"
    # Deliberately invalid test data, not a fabricated trained checkpoint.
    torch.save({"only_one_tensor": torch.ones(1)}, checkpoint)
    with pytest.raises(CheckpointError, match="No partial weights"):
        FINetPredictor(checkpoint, device="cpu")


def test_missing_batchnorm_buffer_is_rejected(tmp_path, monkeypatch):
    # Use a tiny stand-in for the key-validation unit only; no inference is performed.
    small_model = torch.nn.BatchNorm2d(1)
    incomplete = dict(small_model.state_dict())
    del incomplete["num_batches_tracked"]
    invalid_file = tmp_path / "invalid_buffer_state.bin"
    torch.save(incomplete, invalid_file)
    monkeypatch.setattr("src.finet_adapter.FINet", lambda **kwargs: small_model)
    with pytest.raises(CheckpointError, match="complete FINet state"):
        FINetPredictor(invalid_file, device="cpu")


@pytest.mark.parametrize('completed,expected_complete', [(1, False), (200, True)])
def test_partial_training_provenance_is_preserved(tmp_path, monkeypatch, completed, expected_complete):
    # Tiny stand-in validates metadata only; no synthetic segmentation is produced.
    small_model = torch.nn.BatchNorm2d(1)
    checkpoint = tmp_path / 'unit_progress.bin'
    torch.save({'model_state_dict': small_model.state_dict(), 'training_status': 'baseline_training',
                'completed_epochs': completed, 'config': {'epochs': 200}}, checkpoint)
    monkeypatch.setattr('src.finet_adapter.FINet', lambda **kwargs: small_model)
    predictor = FINetPredictor(checkpoint, 'cpu')
    assert predictor.training_progress == {'completed_epochs': completed, 'planned_epochs': 200,
                                           'training_complete': expected_complete}


@pytest.mark.parametrize("wrapper", [None, "state_dict", "model_state_dict"])
def test_state_dict_wrappers_and_module_prefix(wrapper):
    tensor = torch.tensor([1.0])
    payload = {"module.encoder.weight": tensor}
    if wrapper:
        payload = {wrapper: payload, "epoch": 10}
    state = extract_state_dict(payload)
    assert list(state) == ["encoder.weight"]
    torch.testing.assert_close(state["encoder.weight"], tensor)


@pytest.mark.parametrize("payload", [{}, {"a": "not a tensor"}, {"a": torch.tensor([float("nan")])}])
def test_invalid_state_dict(payload):
    with pytest.raises(CheckpointError):
        extract_state_dict(payload)


def test_cpu_auto_and_invalid_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto") == torch.device("cpu")
    assert select_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")
    for device in ("potato", "mps"):
        with pytest.raises(ValueError):
            select_device(device)


@pytest.mark.pretrained
def test_full_trained_checkpoint_loads_every_tensor():
    checkpoint = os.environ.get("FINET_CHECKPOINT")
    if not checkpoint:
        pytest.skip("FINET_CHECKPOINT not supplied: no full trained FINet segmentation checkpoint yet")
    predictor = FINetPredictor(checkpoint, device="cpu")
    expected = extract_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    actual = predictor.model.state_dict()
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with Path(checkpoint).open("rb") as handle:
        assert predictor.checkpoint_sha256 == hashlib.file_digest(handle, "sha256").hexdigest()


@pytest.mark.pretrained
def test_real_pretrained_image_inference():
    checkpoint, image = os.environ.get("FINET_CHECKPOINT"), os.environ.get("FINET_TEST_IMAGE")
    if not checkpoint or not image:
        pytest.skip("FINET_CHECKPOINT and FINET_TEST_IMAGE required for real-image pretrained inference")
    from src.preprocessing import read_image
    predictor = FINetPredictor(checkpoint, device="cpu")
    result = predictor.predict(image)
    original = read_image(image)
    assert result.probability.shape == (original.height, original.width)
    assert np.isfinite(result.probability).all()
    assert 0 <= result.probability.min() <= result.probability.max() <= 1
    assert result.metadata["strict_checkpoint_loading"] is True
    assert result.metadata["device"] == "cpu" and result.metadata["inference_ms"] > 0
