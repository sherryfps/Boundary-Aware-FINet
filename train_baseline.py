"""Smoke-test or explicitly launch original FINet baseline training."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--steps", type=int, default=2, help="Smoke steps, plus one resume-verification update")
    for name in ("data-root", "split-manifest", "output-dir", "initial-checkpoint", "device"):
        parser.add_argument(f"--{name}")
    for name in ("batch-size", "num-workers"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--resume", help="Epoch-boundary full-run checkpoint (last.pth)")
    parser.add_argument("--start-full-training", action="store_true", help="Explicit opt-in; requires CUDA and matching smoke proof")
    parser.add_argument("--verified-smoke-report", help="GPU smoke_report.json with identical config and dataset split")
    args = parser.parse_args(argv)
    if args.mode == "train" and not args.start_full_training:
        parser.error("Full training is disabled without --start-full-training. Inspect the GPU smoke report first.")
    if args.mode == "smoke" and args.resume:
        parser.error("Smoke mode verifies save/reload internally; --resume is only for an actual baseline run.")
    try:
        from src.training import TrainingConfig, run_baseline, run_smoke
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        for name in ("data_root", "split_manifest", "output_dir", "initial_checkpoint", "device", "batch_size", "num_workers"):
            value = getattr(args, name)
            if value is not None:
                settings[name] = value
        config = TrainingConfig(**settings)
        config.validate()
        print(json.dumps({"mode": args.mode, "settings": asdict(config)}, indent=2), flush=True)
        if args.mode == "smoke":
            report = run_smoke(config, args.steps)
            print(json.dumps({"status": report["status"], "report": str(Path(config.output_dir) / "smoke_report.json"),
                              "full_training_started": False}), flush=True)
        else:
            run_baseline(config, args.resume, args.verified_smoke_report)
        return 0
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, ImportError, OSError) as exc:
        print(f"Training setup error: {exc}", file=sys.stderr)
    except (RuntimeError, MemoryError) as exc:
        print(f"Training execution error: {exc}. If out of memory, lower batch size to >=2 and re-run smoke verification; "
              "do not silently change an existing experiment.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
