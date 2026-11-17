"""Verify the original torchvision initialization; this is not a COD quality test."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.models import EfficientNet_B0_Weights
from Model.FINet import FINet


def verify() -> dict:
    torch.set_num_threads(2)
    weights = EfficientNet_B0_Weights.DEFAULT
    reference = weights.get_state_dict(progress=True, check_hash=True)
    model = FINet(backbone="efficientb0", channels=(8, 24, 32, 64), pretrained_backbone=True).eval()
    encoder = model.encoder.state_dict()
    matched = 0
    for key, value in encoder.items():
        layer, rest = key.split(".", 1)
        reference_key = f"features.{int(layer.removeprefix('layer')) - 1}.{rest}"
        torch.testing.assert_close(value, reference[reference_key], rtol=0, atol=0)
        matched += 1
    with torch.inference_mode():
        rgb = torch.zeros(1, 3, 384, 384)
        high = torch.zeros(1, 96, 48, 48)
        low = torch.zeros_like(high)
        features = model.encoder(rgb)
        outputs = model(rgb, high, low)
    assert [value.shape[1] for value in features] == [16, 24, 40, 112, 320]
    assert all(value.shape == (1, 1, 384, 384) and torch.isfinite(value).all() for value in outputs)
    cache = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    with cache.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    pretrained_parameters = sum(p.numel() for p in model.encoder.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    return {"status": "passed", "torch_version": torch.__version__, "device": "cpu",
            "cuda_available": torch.cuda.is_available(), "weights": str(weights),
            "weights_url": weights.url, "cached_file": str(cache), "sha256": sha256,
            "encoder_tensors_equal_to_torchvision": matched,
            "feature_shapes": [list(value.shape) for value in features],
            "output_shapes": [list(value.shape) for value in outputs],
            "total_parameters": total_parameters, "imagenet_pretrained_parameters": pretrained_parameters,
            "randomly_initialized_parameters": total_parameters - pretrained_parameters,
            "random_components": ["re_conv1..4", "ffm1..4 (HFA/LFA)", "deconv1..3", "out_conv1..4"],
            "limitation": "Encoder initialization and tensor interfaces only; segmentation heads have not been trained."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/training_verification/backbone.json"))
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
