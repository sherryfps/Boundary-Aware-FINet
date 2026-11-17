"""Explicit native-resolution evaluation without prediction min-max scaling."""

import math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from src.postprocessing import validate_probability, validate_threshold


def read_ground_truth(path, shape):
    with Image.open(Path(path)) as image:
        if image.format != 'PNG':
            raise ValueError('Ground truth must be an original PNG mask.')
        rgb = np.asarray(image.convert('RGB'))
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
            raise ValueError('Use a grayscale object mask, not colored instance labels.')
        target = np.asarray(image.convert('L')) >= 128
    if target.shape != shape:
        raise ValueError('Ground truth must match the original photograph dimensions exactly.')
    return target


def object_boundary(mask):
    """One-pixel inner contour with 8-neighbour erosion, including frame edges."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Boundary mask must be a nonempty 2D array.')
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def boundary_f1(prediction, target, tolerance_pixels=2):
    """Symmetric contour F1 with an explicit Euclidean distance tolerance.

    Project protocol: two native-resolution pixels by default; both empty
    contours score 1 and only one empty contour scores 0. Not a threshold sweep.
    """
    if not math.isfinite(tolerance_pixels) or tolerance_pixels < 0:
        raise ValueError('Boundary tolerance must be finite and nonnegative.')
    if prediction.shape != target.shape:
        raise ValueError('Prediction and target dimensions differ.')
    predicted_edge, target_edge = object_boundary(prediction), object_boundary(target)
    predicted_count, target_count = predicted_edge.sum(), target_edge.sum()
    if not predicted_count or not target_count:
        return float(predicted_count == target_count)
    matched_predicted = (ndimage.distance_transform_edt(~target_edge)[predicted_edge] <= tolerance_pixels).sum()
    matched_target = (ndimage.distance_transform_edt(~predicted_edge)[target_edge] <= tolerance_pixels).sum()
    precision, recall = matched_predicted / predicted_count, matched_target / target_count
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def metrics_for_pair(probability, target, threshold=0.5, tolerance_pixels=2):
    probability = validate_probability(probability).astype(np.float64)
    validate_threshold(threshold)
    target = np.asarray(target)
    if target.shape != probability.shape or target.dtype != np.bool_:
        raise ValueError('Metrics require a matching boolean ground-truth mask.')
    from py_sod_metrics import Smeasure
    metric = Smeasure()
    # The package default min-max scales predictions. Disable it explicitly.
    metric.step(pred=probability, gt=target, normalize=False)
    return {'s_measure': float(metric.get_results()['sm']),
            'mae': float(np.abs(probability - target.astype(np.float64)).mean()),
            'boundary_f1': boundary_f1(probability >= threshold, target, tolerance_pixels),
            'threshold': float(threshold), 'boundary_tolerance_pixels': float(tolerance_pixels)}
