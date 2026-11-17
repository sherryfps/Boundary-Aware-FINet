"""Training-only morphological boundary supervision; FINet layers stay unchanged."""

import math
import torch
from torch.nn import functional as F


def morphological_boundary(probability: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """MaxPool(P) - MinPool(P), ignoring pixels outside the image.

    Pooling is piecewise differentiable. Constant foreground/background maps have
    zero boundary, including at image borders; padding creates no artificial edge.
    """
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError("Boundary kernel must be an odd integer >=3.")
    if probability.ndim != 4 or probability.shape[1] != 1 or probability.numel() == 0:
        raise ValueError("Boundary input must be a nonempty [B,1,H,W] tensor.")
    if not probability.is_floating_point() or not torch.isfinite(probability).all():
        raise ValueError("Boundary probabilities must be finite floating point values.")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Boundary probabilities must lie within [0,1].")
    padding = kernel_size // 2
    maximum = F.max_pool2d(probability, kernel_size, 1, padding)
    minimum = -F.max_pool2d(-probability, kernel_size, 1, padding)
    return maximum - minimum


def boundary_loss(logits: torch.Tensor, mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Mean per-image BCE + smoothed Dice on matching morphological boundaries.

    The original bilinear-resized mask remains unchanged for structure loss.
    Only the boundary target is binarized at 0.5 and detached. All four heads
    receive this loss when enabled. FP32 probability BCE avoids log(0).
    """
    if logits.shape != mask.shape or not torch.isfinite(logits).all() or not torch.isfinite(mask).all():
        raise ValueError("Boundary logits and target must have matching finite tensors.")
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("Boundary target mask must lie within [0,1].")
    prediction = morphological_boundary(logits.float().sigmoid(), kernel_size)
    target = morphological_boundary((mask.detach() >= 0.5).float(), kernel_size)
    epsilon = torch.finfo(prediction.dtype).eps
    bce = F.binary_cross_entropy(prediction.clamp(epsilon, 1 - epsilon), target, reduction="none")
    dimensions = (1, 2, 3)
    intersection = (prediction * target).sum(dimensions)
    dice = 1 - (2 * intersection + 1) / (prediction.sum(dimensions) + target.sum(dimensions) + 1)
    return (bce.mean(dimensions) + dice).mean()


def finet_objective(outputs, mask, boundary_lambda=0.0, kernel_size=3):
    """Return total, original structure sum, boundary sum, and per-head structure."""
    from train import structure_loss
    if not math.isfinite(boundary_lambda) or boundary_lambda < 0:
        raise ValueError("Boundary lambda must be finite and nonnegative.")
    if len(outputs) != 4 or any(output.shape != mask.shape for output in outputs):
        raise ValueError("Expected FINet's four same-resolution segmentation heads.")
    heads = [structure_loss(output, mask) for output in outputs]
    structure = sum(heads)
    # Preserve exact baseline arithmetic and avoid boundary operations for lambda=0.
    boundary = (sum(boundary_loss(output, mask, kernel_size) for output in outputs)
                if boundary_lambda else structure.new_zeros(()))
    total = structure + boundary_lambda * boundary if boundary_lambda else structure
    return total, structure, boundary, heads
