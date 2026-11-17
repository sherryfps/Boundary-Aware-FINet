"""Metric identities on analytic masks, never claimed as model performance."""

import numpy as np
import pytest
pytest.importorskip('py_sod_metrics')
from src.evaluation_metrics import boundary_f1, metrics_for_pair, read_ground_truth
from PIL import Image


def test_perfect_object_has_perfect_scores():
    target = np.zeros((64, 80), dtype=bool)
    target[15:40, 20:55] = True
    result = metrics_for_pair(target.astype(np.float32), target)
    assert result['mae'] == 0 and result['boundary_f1'] == 1
    assert result['s_measure'] == pytest.approx(1)


def test_empty_and_full_masks_and_boundary_tolerance():
    empty = np.zeros((40, 40), dtype=bool)
    full = np.ones_like(empty)
    assert boundary_f1(empty, empty) == 1
    assert boundary_f1(full, empty) == 0
    first = empty.copy(); first[10:25, 10:25] = True
    second = empty.copy(); second[10:25, 11:26] = True
    assert boundary_f1(first, second, 1) == 1
    assert boundary_f1(first, second, 0) < 1


def test_raw_probability_mae_and_no_minmax_scaling():
    target = np.zeros((8, 9), dtype=bool)
    probability = np.full(target.shape, 0.8)
    result = metrics_for_pair(probability, target)
    assert result['mae'] == pytest.approx(0.8)
    assert result['s_measure'] == pytest.approx(0.2)
    with pytest.raises(ValueError, match='boolean'):
        metrics_for_pair(probability, target.astype(np.uint8))


def test_ground_truth_dimensions_and_binarization(tmp_path):
    path = tmp_path/'mask.png'
    values = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    Image.fromarray(values).save(path)
    np.testing.assert_array_equal(read_ground_truth(path, (2, 2)), [[False, False], [True, True]])
    with pytest.raises(ValueError, match='dimensions'):
        read_ground_truth(path, (3, 4))
