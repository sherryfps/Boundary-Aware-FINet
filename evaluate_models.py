"""Evaluate real FINet checkpoints on labelled unseen data, excluding development overlap."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch

from src.evaluation_metrics import metrics_for_pair, read_ground_truth
from src.evaluation_data import audit_evaluation_data
from src.finet_adapter import FINetPredictor
from src.postprocessing import save_prediction, validate_threshold
from src.training import hardware_report, write_json
from src.training_data import audit_pairs


def evaluate(checkpoint, dataset, training_manifest, destination, device='auto', threshold=0.5,
             tolerance=2, sample_count=3, warmup=2):
    validate_threshold(threshold)
    predictor = FINetPredictor(checkpoint, device)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('training_status') in {'baseline_training', 'boundary_training'} and payload.get('completed_epochs', 0) < 1:
        raise ValueError('A completed training epoch is required; initialization is not an evaluation model.')
    development = json.loads(Path(training_manifest).read_text(encoding='utf-8'))
    development_names = {record['id'] for record in development['records']}
    development_hashes = {record['image_sha256'] for record in development['records']}
    records, invalid_geometry, released_pairs = audit_evaluation_data(dataset)
    excluded = [record['id'] for record in records if record['id'] in development_names
                or record['image_sha256'] in development_hashes]
    included = [record for record in records if record['id'] not in set(excluded)]
    if not included:
        raise ValueError('No unseen labelled images remain after development-overlap exclusion.')
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Evaluation output already contains evidence; choose a new output directory.')
    for _ in range(warmup):
        predictor.predict(Path(dataset)/included[0]['image'])
    rows = []
    for index, record in enumerate(included):
        prediction = predictor.predict(Path(dataset)/record['image'])
        target = read_ground_truth(Path(dataset)/record['mask'], prediction.probability.shape)
        scores = metrics_for_pair(prediction.probability, target, threshold, tolerance)
        row = {'sample_id': record['id'], 'image_sha256': record['image_sha256'], **scores,
               'inference_ms': prediction.metadata['inference_ms'],
               'processing_ms': prediction.metadata['processing_ms']}
        rows.append(row)
        if index < sample_count:
            save_prediction(prediction.original, prediction.probability, destination/'samples'/record['id'],
                            {**prediction.metadata, 'ground_truth_metrics': scores}, threshold)
        if (index + 1) % 100 == 0:
            print(json.dumps({'evaluated': index + 1, 'total': len(included)}), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'per_image.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {'model': predictor.model_name, 'checkpoint_sha256': predictor.checkpoint_sha256,
              'checkpoint_training_status': predictor.training_status,
              'completed_epochs': payload.get('completed_epochs'), 'training_config': payload.get('config'),
              'initial_model_sha256': payload.get('initial_model_sha256'),
              'training_split_fingerprint': payload.get('split_fingerprint'),
              'evaluated_ids': [row['sample_id'] for row in rows],
              'evaluated_image_hashes': [row['image_sha256'] for row in rows],
              'excluded_development_overlap_ids': excluded,
              'excluded_invalid_geometry': invalid_geometry, 'released_pairs': released_pairs,
              'sample_count': len(rows), 'parameter_count': predictor.parameter_count,
              'metrics': {name: float(np.mean([row[name] for row in rows])) for name in ('s_measure', 'mae', 'boundary_f1')},
              'inference_latency_ms': {'median': float(np.median([row['inference_ms'] for row in rows])),
                                       'mean': float(np.mean([row['inference_ms'] for row in rows])),
                                       'p95': float(np.percentile([row['inference_ms'] for row in rows], 95))},
              'protocol': {'prediction': 'native-resolution raw sigmoid probabilities; no min-max scaling',
                           'ground_truth': 'grayscale PNG thresholded at 128; original files unchanged',
                           's_measure': 'PySODMetrics 1.6.2, alpha=0.5, normalize=False',
                           'mae': 'native-resolution raw probability vs binary GT, averaged per image',
                           'boundary_f1': '8-neighbour one-pixel inner contour, Euclidean many-to-many matching',
                           'boundary_tolerance_pixels': tolerance, 'threshold': threshold,
                           'warmup_forwards_excluded': warmup, 'inference_batch_size': 1,
                           'latency': 'synchronized FINet forward only; model load, preprocessing and saving excluded',
                           'development_manifest_fingerprint': development['fingerprint']},
              'hardware': hardware_report(predictor.device),
              'created_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(destination/'summary.json', report)
    return report


def comparison_audit(left, right):
    ignored = {'output_dir', 'initial_checkpoint', 'boundary_lambda'}
    configs = [report.get('training_config') or {} for report in (left, right)]
    keys = (configs[0].keys() | configs[1].keys()) - ignored
    differences = [key for key in sorted(keys) if configs[0].get(key) != configs[1].get(key)]
    checks = {'same_initial_weights': bool(left.get('initial_model_sha256')) and left.get('initial_model_sha256') == right.get('initial_model_sha256'),
              'baseline_vs_boundary_objectives': configs[0].get('boundary_lambda', 0) == 0 and configs[1].get('boundary_lambda', 0) > 0,
              'same_training_split': bool(left.get('training_split_fingerprint')) and left.get('training_split_fingerprint') == right.get('training_split_fingerprint'),
              'same_completed_epochs': left.get('completed_epochs') is not None and left.get('completed_epochs') == right.get('completed_epochs'),
              'same_settings_except_boundary_loss': bool(configs[0]) and bool(configs[1]) and not differences,
              'same_evaluated_images': left['evaluated_image_hashes'] == right['evaluated_image_hashes'],
              'same_metric_protocol': left['protocol'] == right['protocol'],
              'same_parameter_count': left['parameter_count'] == right['parameter_count']}
    return {'checks': checks, 'configuration_differences': differences,
            'controlled_protocol_verified': all(checks.values()),
            'interpretation': 'Measured differences are descriptive for these runs; no statistical significance or improvement is assumed.'}


def write_comparison(left, right, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = comparison_audit(left, right)
    write_json(output/'comparison.json', {'original': left, 'boundary': right, 'audit': audit})
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 's_measure', 'mae', 'boundary_f1', 'parameters', 'median_forward_ms'])
        writer.writeheader()
        for report in (left, right):
            writer.writerow({'model': report['model'], **report['metrics'], 'parameters': report['parameter_count'],
                             'median_forward_ms': report['inference_latency_ms']['median']})
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout='constrained')
    for axis, metric, title in zip(axes, ('s_measure', 'mae', 'boundary_f1'), ('S-measure (higher better)', 'MAE (lower better)', 'Boundary F1 (higher better)')):
        values = [report['metrics'][metric] for report in (left, right)]
        axis.bar(['Original', 'Boundary'], values, color=['#626c77', '#a35535'])
        axis.set_title(title, fontsize=10); axis.set_ylim(0, 1)
        for index, value in enumerate(values):
            axis.text(index, min(value + 0.02, 0.98), f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    figure.suptitle('Measured FINet comparison' + ('' if audit['controlled_protocol_verified'] else ' (protocol differences; see audit)'), fontsize=11)
    figure.savefig(output/'comparison.png', dpi=180)
    figure.savefig(output/'comparison.pdf')
    plt.close(figure)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--boundary', type=Path)
    parser.add_argument('--dataset', type=Path, required=True, help='One test dataset containing Imgs and GT')
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--boundary-tolerance', type=float, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    left = evaluate(args.baseline, args.dataset, args.training_manifest, args.output/'original', args.device,
                    args.threshold, args.boundary_tolerance)
    print(json.dumps({key: left[key] for key in ('model', 'sample_count', 'metrics', 'inference_latency_ms')}, indent=2))
    if args.boundary:
        right = evaluate(args.boundary, args.dataset, args.training_manifest, args.output/'boundary', args.device,
                         args.threshold, args.boundary_tolerance)
        print(json.dumps(write_comparison(left, right, args.output), indent=2))


if __name__ == '__main__':
    main()
