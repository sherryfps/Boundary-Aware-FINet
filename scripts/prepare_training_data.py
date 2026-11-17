"""Audit official Imgs/GT folders and save a fixed, disjoint training/validation split."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_data import create_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--allow-small", action="store_true", help="Smoke tests only, not the 4040-pair baseline")
    parser.add_argument("--allow-soft-masks", action="store_true", help="Preserve original grayscale CAMO edge labels; record explicit policy")
    parser.add_argument("--group-duplicates", action="store_true", help="Keep official duplicate images together to prevent split leakage")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Manifest already exists; keep it fixed or choose a new output path.")
    result = create_manifest(args.data_root, args.validation_fraction, args.seed, args.allow_small,
                             args.allow_soft_masks, args.group_duplicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["source_counts"], "train": len(result["train_ids"]),
                      "validation": len(result["validation_ids"]), "fingerprint": result["fingerprint"],
                      "smoke_only": result["smoke_only"]}, indent=2))
