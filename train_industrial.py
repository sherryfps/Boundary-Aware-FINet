"""Smoke-test or explicitly train a separate custom-supervised MVTec FINet model."""

import argparse
import json
from pathlib import Path

from src.industrial_training import IndustrialConfig, run_industrial_smoke, run_industrial_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/industrial_baseline.json'))
    parser.add_argument('--mode', choices=('smoke', 'train'), default='smoke')
    parser.add_argument('--steps', type=int, default=2)
    for key in ('data-root', 'split-manifest', 'output-dir', 'initial-checkpoint', 'device'):
        parser.add_argument(f'--{key}')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--verified-smoke-report')
    parser.add_argument('--resume')
    parser.add_argument('--max-hours', type=float, default=7.5)
    parser.add_argument('--max-new-epochs', type=int)
    parser.add_argument('--start-full-training', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'train' and not args.start_full_training:
        parser.error('Inspect the GPU smoke result before explicitly enabling long training.')
    if args.mode == 'smoke' and args.resume:
        parser.error('Smoke mode tests its own recovery; --resume is for full training.')
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('data_root', 'split_manifest', 'output_dir', 'initial_checkpoint', 'device', 'num_workers', 'batch_size'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    config = IndustrialConfig(**settings)
    if args.mode == 'smoke':
        result = run_industrial_smoke(config, args.steps)
    else:
        result = run_industrial_training(config, args.verified_smoke_report, args.resume,
                                         args.max_hours, args.max_new_epochs)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
