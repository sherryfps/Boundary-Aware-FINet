"""Restore official logits and create display artifacts without altering predictions."""

from pathlib import Path
import json
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def validate_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability)
    if (probability.ndim != 2 or probability.size == 0
            or not np.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1):
        raise ValueError("Probability mask must be a nonempty finite H×W array in [0, 1].")
    return probability


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1.")


def restore_probability(logits: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    """Pillow size is (width, height); torch interpolation expects (height, width)."""
    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or min(logits.shape[2:]) < 1:
        raise ValueError("Expected single-image segmentation logits shaped [1, 1, H, W].")
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits.")
    if len(original_size) != 2 or any(x <= 0 for x in original_size):
        raise ValueError("Original image dimensions must be positive.")
    width, height = original_size
    # This order and align_corners setting reproduce original inference.py.
    resized = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
    probability = resized.sigmoid()[0, 0].detach().cpu().numpy().astype(np.float32)
    return validate_probability(probability)


def make_visualizations(original: Image.Image, probability: np.ndarray,
                        threshold: float = 0.5, opacity: float = 0.45) -> dict[str, Image.Image]:
    probability = validate_probability(probability)
    validate_threshold(threshold)
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1.")
    original = original.convert("RGB")
    if probability.shape != (original.height, original.width):
        raise ValueError("Probability mask and original image dimensions do not match.")
    foreground = probability >= threshold
    binary = foreground.astype(np.uint8) * 255
    # Deliberately truncate as upstream does, rather than min-max normalizing.
    grayscale = (probability * 255).astype(np.uint8)
    overlay = np.asarray(original).copy()
    color = np.array([0, 220, 170], dtype=np.float32)
    overlay[foreground] = (
        (1 - opacity) * overlay[foreground].astype(np.float32) + opacity * color
    ).astype(np.uint8)
    return {"probability": Image.fromarray(grayscale), "binary": Image.fromarray(binary),
            "overlay": Image.fromarray(overlay)}


def save_prediction(original: Image.Image, probability: np.ndarray, output_dir: str | Path,
                    metadata: dict, threshold: float = 0.5) -> dict[str, str]:
    """Save float probabilities as well as the three requested PNG visualizations."""
    visuals = make_visualizations(original, probability, threshold)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, image in visuals.items():
        path = output_dir / f"{name}.png"
        image.save(path)
        paths[name] = str(path)
    float_path = output_dir / "probability.npy"
    np.save(float_path, probability.astype(np.float32), allow_pickle=False)
    paths["probability_float"] = str(float_path)
    paths["report"] = str(output_dir / "prediction.json")
    report = {**metadata, "threshold": threshold,
              "probability_range": [float(probability.min()), float(probability.max())],
              "output_size_wh": list(original.size), "outputs": paths}
    Path(paths["report"]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
