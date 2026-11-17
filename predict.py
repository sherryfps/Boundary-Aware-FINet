"""Run a genuine full-checkpoint FINet prediction and save PNG/NumPy artifacts."""

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input JPG, JPEG or PNG")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Official or project-trained full FINet EfficientNet-B0 checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary/overlay threshold in [0,1]")
    args = parser.parse_args(argv)
    try:
        from src.postprocessing import save_prediction, validate_threshold
        from src.preprocessing import read_image
        from src.finet_adapter import FINetPredictor

        validate_threshold(args.threshold)
        # Validate inputs before allocating a model. Never overwrite the input image.
        read_image(args.image)
        output_dir = args.output_dir.expanduser().resolve()
        targets = [output_dir / name for name in
                   ("probability.png", "binary.png", "overlay.png", "probability.npy", "prediction.json")]
        if args.image.expanduser().resolve() in targets:
            raise ValueError("Output directory would overwrite the input image. Choose a different --output-dir.")
        predictor = FINetPredictor(args.checkpoint, device=args.device)
        result = predictor.predict(args.image)
        paths = save_prediction(result.original, result.probability, output_dir,
                                result.metadata, threshold=args.threshold)
        print(json.dumps({**result.metadata, "outputs": paths}, indent=2, allow_nan=False))
        return 0
    except ImportError as exc:
        print(f"Missing or incompatible dependency: {exc}. Install requirements.txt in your Python environment.", file=sys.stderr)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"FINet error: {exc}", file=sys.stderr)
    except MemoryError:
        print("Not enough memory. Close other applications and retry one image on CPU.", file=sys.stderr)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("Inference ran out of memory. Retry with --device cpu and close other applications.", file=sys.stderr)
        else:
            print(f"PyTorch could not run FINet: {exc}. Check the device and compatible torch/torchvision installation.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
