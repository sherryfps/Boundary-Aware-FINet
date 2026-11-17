from pathlib import Path

import pytest
import torch
from PIL import Image

from src.preprocessing import FINetPreprocessor, PROJECT_ROOT, read_image
from utils.dataloader_freq import TestDataset as OfficialTestDataset


def test_exact_parity_with_official_loader(tmp_path, monkeypatch, patterned_image):
    """The independent reference is the actual upstream TestDataset, not our wrapper."""
    images, masks = tmp_path / "Imgs", tmp_path / "GT"
    images.mkdir()
    masks.mkdir()
    patterned_image.save(images / "input.png")
    # Loader-only fixture: a synthetic target is needed to call the original dataset.
    Image.new("L", patterned_image.size, 0).save(masks / "input.png")
    monkeypatch.chdir(PROJECT_ROOT)
    reference = OfficialTestDataset(str(images), str(masks), 384)[0]
    actual = FINetPreprocessor()(patterned_image)
    for tensor, expected in zip((actual.rgb, actual.high, actual.low), (reference[0], reference[4], reference[5])):
        torch.testing.assert_close(tensor[0], expected, rtol=0, atol=0)
    assert actual.rgb.shape == (1, 3, 384, 384)
    assert actual.high.shape == actual.low.shape == (1, 96, 48, 48)
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all()
               for t in (actual.rgb, actual.high, actual.low))
    assert actual.original.size == patterned_image.size


def test_statistics_resolve_outside_project(tmp_path, monkeypatch, patterned_image):
    monkeypatch.chdir(tmp_path)
    prepared = FINetPreprocessor()(patterned_image)
    assert torch.isfinite(prepared.high).all()


@pytest.mark.parametrize("size", [0, -64, 8, 96, 383, 384.0, True])
def test_invalid_dimensions(size):
    with pytest.raises(ValueError, match="multiple of 64"):
        FINetPreprocessor(size)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_color_modes(mode):
    prepared = FINetPreprocessor()(Image.new(mode, (19, 31)))
    assert prepared.original.mode == "RGB"
    assert prepared.rgb.shape == (1, 3, 384, 384)
    assert torch.isfinite(prepared.low).all()


@pytest.mark.parametrize("extension,format_name", [(".JPG", "JPEG"), (".jpeg", "JPEG"), (".PNG", "PNG")])
def test_supported_files(tmp_path, patterned_image, extension, format_name):
    image_path = tmp_path / f"image{extension}"
    patterned_image.save(image_path, format=format_name)
    assert read_image(image_path).size == patterned_image.size


def test_invalid_and_unsupported_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        read_image(tmp_path / "missing.png")
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("This is not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot decode"):
        read_image(corrupt)
    unsupported = tmp_path / "image.gif"
    Image.new("RGB", (4, 4)).save(unsupported)
    with pytest.raises(ValueError, match="Unsupported"):
        read_image(unsupported)
    disguised = tmp_path / "disguised.png"
    Image.new("RGB", (4, 4)).save(disguised, format="GIF")
    with pytest.raises(ValueError, match="contents"):
        read_image(disguised)
