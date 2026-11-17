import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from src.postprocessing import make_visualizations, restore_probability, save_prediction


def test_restore_matches_original_resize_then_sigmoid():
    logits = torch.tensor([[[[-3.0, 1.0], [2.0, 7.0]]]])
    actual = restore_probability(logits, (13, 7))
    expected = F.interpolate(logits, size=(7, 13), mode="bilinear", align_corners=True).sigmoid()[0, 0].numpy()
    np.testing.assert_array_equal(actual, expected)
    wrong_order = F.interpolate(logits.sigmoid(), size=(7, 13), mode="bilinear", align_corners=True)[0, 0].numpy()
    assert not np.allclose(actual, wrong_order)
    assert actual.shape == (7, 13)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()


def test_threshold_changes_binary_and_aligned_overlay_only():
    original = Image.new("RGB", (3, 2), (30, 40, 50))
    probability = np.array([[0.1, 0.5, 0.9], [0.4, 0.7, 0.2]], dtype=np.float32)
    low = make_visualizations(original, probability, threshold=0.3)
    high = make_visualizations(original, probability, threshold=0.8)
    np.testing.assert_array_equal(np.asarray(low["probability"]), np.asarray(high["probability"]))
    np.testing.assert_array_equal(np.asarray(high["binary"]), (probability >= 0.8).astype(np.uint8) * 255)
    foreground = probability >= 0.8
    np.testing.assert_array_equal(np.asarray(high["overlay"])[~foreground], np.asarray(original)[~foreground])
    assert np.all(np.asarray(high["overlay"])[foreground] != np.asarray(original)[foreground])
    assert np.count_nonzero(np.asarray(low["binary"])) > np.count_nonzero(np.asarray(high["binary"]))


@pytest.mark.parametrize("value,expected", [(0.0, 0), (1.0, 255)])
def test_constant_masks_remain_constant(value, expected):
    outputs = make_visualizations(Image.new("RGB", (2, 3)), np.full((3, 2), value))
    assert np.all(np.asarray(outputs["probability"]) == expected)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        make_visualizations(Image.new("RGB", (2, 2)), np.zeros((2, 2)), threshold=threshold)


@pytest.mark.parametrize("probability", [np.array([[float("nan")]]), np.array([[-0.1]]),
                                       np.array([[1.1]]), np.zeros((1, 1, 1)), np.array([])])
def test_invalid_probability(probability):
    with pytest.raises(ValueError, match="Probability"):
        make_visualizations(Image.new("RGB", (1, 1)), probability)


def test_mismatched_image_shape():
    with pytest.raises(ValueError, match="dimensions"):
        make_visualizations(Image.new("RGB", (5, 3)), np.zeros((5, 3)))


@pytest.mark.parametrize("logits", [torch.zeros(1, 2, 3, 3), torch.zeros(3, 3),
                                  torch.full((1, 1, 2, 2), float("inf"))])
def test_invalid_logits(logits):
    with pytest.raises(ValueError):
        restore_probability(logits, (5, 3))


def test_save_dimensions_values_and_float_precision(tmp_path, patterned_image):
    # Unit fixture only: these are known numbers, not saved demo/model predictions.
    probability = np.linspace(0, 1, patterned_image.width * patterned_image.height,
                              dtype=np.float32).reshape(patterned_image.height, patterned_image.width)
    paths = save_prediction(patterned_image, probability, tmp_path, {"unit_test_fixture": True})
    expected = make_visualizations(patterned_image, probability)
    for name in ("probability", "binary", "overlay"):
        with Image.open(paths[name]) as saved:
            assert saved.size == patterned_image.size
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(expected[name]))
    np.testing.assert_array_equal(np.load(paths["probability_float"], allow_pickle=False), probability)
    report = json.loads((tmp_path / "prediction.json").read_text(encoding="utf-8"))
    assert report["output_size_wh"] == list(patterned_image.size)
    assert report["probability_range"] == [0.0, 1.0]
