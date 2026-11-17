"""Prepare the explicit custom supervised MVTec AD protocol."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.industrial_data import CATEGORIES, prepare_supervised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, default=Path('data/MVTecAD_original'))
    parser.add_argument('--output', type=Path, default=Path('data/MVTecAD_supervised_v1'))
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    args = parser.parse_args()
    manifest = prepare_supervised(args.original_root, args.output, args.categories)
    print(json.dumps({'fingerprint': manifest['fingerprint'], 'categories': manifest['categories'],
                      'protocol': manifest['protocol'], 'source_counts': manifest['source_counts'],
                      'splits': {key: len(manifest[key]) for key in ('train_ids','validation_ids','test_ids')},
                      'defects_lost_at_384': sum(r['is_defective'] and r['foreground_pixels_384'] == 0 for r in manifest['records'])}), flush=True)


if __name__ == '__main__':
    main()
