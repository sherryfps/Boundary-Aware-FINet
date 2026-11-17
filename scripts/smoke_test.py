"""Require real weights + a real image and validate saved FINet artifacts."""

import argparse
import json
from pathlib import Path
import sys

# Permit python scripts/smoke_test.py from either this folder or another cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    import predict
    status = predict.main(["--image", str(args.image), "--checkpoint", str(args.checkpoint),
                           "--output-dir", str(args.output_dir), "--device", args.device])
    if status:
        return status
    import numpy as np
    from PIL import Image
    from src.postprocessing import make_visualizations, validate_probability
    from src.preprocessing import read_image

    original = read_image(args.image)
    probability = validate_probability(np.load(args.output_dir / "probability.npy", allow_pickle=False))
    assert probability.shape == (original.height, original.width)
    expected = make_visualizations(original, probability)
    for name, reference in expected.items():
        with Image.open(args.output_dir / f"{name}.png") as saved:
            assert saved.size == original.size, f"Incorrect restored dimensions: {name}"
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(reference))
    report = json.loads((args.output_dir / "prediction.json").read_text(encoding="utf-8"))
    assert report["strict_checkpoint_loading"] is True
    assert report["parameter_count"] > 0 and report["inference_ms"] > 0
    assert len(report["checkpoint_sha256"]) == 64
    evidence = {"status": "passed", "image": report["image"], "checkpoint_sha256": report["checkpoint_sha256"],
                "checks": ["strict full checkpoint load", "real forward pass", "finite [0,1] probabilities",
                           "original dimensions restored", "saved PNGs match float prediction",
                           "binary and overlay align pixelwise"],
                "limitation": "Artifact integrity is checked; segmentation accuracy requires ground truth and evaluation."}
    (args.output_dir / "smoke_report.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
