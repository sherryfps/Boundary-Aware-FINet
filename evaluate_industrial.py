"""Evaluate completed matched industrial experiments on the untouched custom test split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.finet_adapter import FINetPredictor
from src.industrial_data import DOMAIN, load_industrial_manifest
from src.industrial_training import aggregate_metrics, segmentation_metrics
from src.postprocessing import save_prediction
from src.training import write_json
from src.training_data import file_hash
from PIL import Image


def verify_matched_checkpoints(paths, manifest):
    payloads = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    for payload in payloads:
        if (payload.get('application_domain') != DOMAIN
                or payload.get('dataset_protocol') != manifest['protocol']
                or payload.get('split_fingerprint') != manifest['fingerprint']
                or payload.get('completed_epochs') != payload.get('config', {}).get('epochs')
                or payload.get('completed_epochs', 0) < 1
                or payload.get('training_status') not in ('baseline_training', 'boundary_training')):
            raise ValueError('Primary evaluation needs real completed industrial checkpoints on this exact split.')
    a, b = payloads
    if a['training_status'] != 'baseline_training' or b['training_status'] != 'boundary_training':
        raise ValueError('Supply original baseline first and boundary-trained model second.')
    if not a.get('initial_model_sha256') or a['initial_model_sha256'] != b.get('initial_model_sha256'):
        raise ValueError('Industrial A/B experiments do not share the exact full-model initialization.')
    configs = []
    for payload in payloads:
        config = dict(payload['config'])
        for key in ('output_dir', 'initial_checkpoint', 'data_root', 'split_manifest', 'device', 'boundary_lambda'):
            config.pop(key, None)
        configs.append(config)
    if configs[0] != configs[1] or a['config']['boundary_lambda'] != 0 or b['config']['boundary_lambda'] != .1:
        raise ValueError('Industrial A/B settings differ beyond the registered boundary coefficient/output paths.')
    return [{'checkpoint_sha256': file_hash(Path(path)), 'completed_epochs': payload['completed_epochs'],
             'initial_model_sha256': payload['initial_model_sha256'], 'split_fingerprint': payload['split_fingerprint'],
             'config': payload['config']} for path, payload in zip(paths, payloads)]


def evaluate(data_root, manifest_path, baseline, boundary, output, device='auto'):
    manifest = load_industrial_manifest(manifest_path, data_root)
    paths = [Path(baseline), Path(boundary)]
    provenance = verify_matched_checkpoints(paths, manifest)
    output, data_root = Path(output), Path(data_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Preserve existing industrial evaluation results; choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    lookup = {row['id']: row for row in manifest['records']}
    summaries = {}
    for name, checkpoint, proof in zip(('original', 'boundary'), paths, provenance):
        predictor = FINetPredictor(checkpoint, device)
        warmup_id = manifest['validation_ids'][0]
        for _ in range(3):
            predictor.predict(data_root / lookup[warmup_id]['image'])
        rows, timings = [], []
        saved_counts = {True: 0, False: 0}
        for sample_id in manifest['test_ids']:
            record = lookup[sample_id]
            prediction = predictor.predict(data_root / record['image'])
            with Image.open(data_root / record['mask']) as mask:
                target = np.asarray(mask.convert('L')) > 0
            row = {'id': sample_id, 'category': record['category'], 'is_defective': record['is_defective'],
                   'original_split': record['original_split'],
                   **segmentation_metrics(prediction.probability, target, .5),
                   'inference_ms': prediction.metadata['inference_ms']}
            rows.append(row)
            timings.append(row['inference_ms'])
            if saved_counts[record['is_defective']] < 3:
                save_prediction(prediction.original, prediction.probability, output/name/'examples'/sample_id,
                                {**prediction.metadata, 'evaluation_split': 'custom held-out test', **row}, .5)
                Image.fromarray(target.astype(np.uint8)*255).save(output/name/'examples'/sample_id/'ground_truth.png')
                saved_counts[record['is_defective']] += 1
        summary = {'protocol': manifest['protocol'], 'official_benchmark_eligible': False,
                   'categories': manifest['categories'], 'threshold': .5,
                   'postprocessing': 'Raw sigmoid probability thresholded at the preregistered 0.5. No min-max scaling, denoising or product acceptance decision.',
                   'boundary_tolerance_native_pixels': 2, **proof, **aggregate_metrics(rows),
                   'latency_ms': {'mean': float(np.mean(timings)), 'median': float(np.median(timings)),
                                  'p95': float(np.percentile(timings,95)), 'samples': len(timings),
                                  'warmup_validation_images': 3, 'device': str(predictor.device)}}
        summaries[name] = summary
        write_json(output/name/'summary.json', summary)
        with (output/name/'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    delta = {key: summaries['boundary']['mean_per_image'][key] - summaries['original']['mean_per_image'][key]
             for key in ('dice','iou','mae','boundary_f1')}
    report = {'domain': DOMAIN, 'protocol': manifest['protocol'], 'official_benchmark_eligible': False,
              'test_images': len(manifest['test_ids']), 'models': summaries, 'boundary_minus_original': delta,
              'interpretation': 'Differences measured on this single-seed custom supervised split; not official MVTec anomaly benchmark or certified factory performance.'}
    write_json(output/'comparison.json', report)
    lines = ['# Industrial FINet comparison', '', 'Custom supervised MVTec AD split; fixed threshold 0.5.', '',
             '| Model | Dice | IoU | MAE | Boundary F1 | Forward median ms |',
             '|---|---:|---:|---:|---:|---:|']
    for name, summary in summaries.items():
        metrics = summary['mean_per_image']
        lines.append(f"| {name} | {metrics['dice']:.6f} | {metrics['iou']:.6f} | {metrics['mae']:.6f} | {metrics['boundary_f1']:.6f} | {summary['latency_ms']['median']:.2f} |")
    lines += ['', 'See per-category and defective-only results in summary.json. Empty normal masks can inflate aggregate overlap scores.',
              'This academic prototype makes no product acceptance/rejection decision. No industrial production certification is claimed.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--boundary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(args.data_root, args.manifest, args.baseline, args.boundary, args.output, args.device), indent=2))


if __name__ == '__main__':
    main()
