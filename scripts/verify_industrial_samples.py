"""Measure three predetermined held-out examples through the actual running app."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gradio_client import Client, handle_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.industrial_training import segmentation_metrics


CHECKPOINT_SHA = 'fd92df468740e977fb16e9bcc6f44dfcf424be8822d155f36ea4cc209d4db94a'
SELECTION = [('bottle', True, 'Broken bottle'), ('metal_nut', True, 'Bent metal nut'),
             ('carpet', False, 'Normal carpet')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7860')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/industrial_three_samples_20261116')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Choose a new output directory to preserve prior evidence.')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    checkpoint = ROOT/'checkpoints/industrial/baseline/last.pth'
    assert sha(checkpoint) == CHECKPOINT_SHA
    trained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert trained['completed_epochs'] == trained['config']['epochs'] == 200
    assert trained['global_step'] == 186400 and trained['config']['boundary_lambda'] == 0
    manifest = json.loads((ROOT/'data/MVTecAD_supervised_v1/split_manifest.json').read_text())
    canonical = json.loads((ROOT/'outputs/industrial_setup/version004/FINetIndustrial/split_manifest.json').read_text())
    assert canonical['fingerprint'] == trained['split_fingerprint']
    lookup = {r['id']: r for r in manifest['records']}
    # Predetermine every sample before any inference; no output-based selection.
    selected = [(next(lookup[i] for i in manifest['test_ids'] if lookup[i]['category'] == category
                      and lookup[i]['is_defective'] == defective), title)
                for category, defective, title in SELECTION]
    for record, _ in selected:
        assert record['id'] in canonical['test_ids']
        assert record['id'] not in canonical['train_ids'] + canonical['validation_ids']
    client = Client(args.url, verbose=False, download_files=out/'downloads', analytics_enabled=False)
    figure, axes = plt.subplots(3, 4, figsize=(12, 10.5), layout='constrained')
    columns = ['Input photograph', 'Ground-truth mask', 'Predicted mask (0.5)', 'Prediction overlay']
    samples = []
    for row, (record, title) in enumerate(selected):
        folder = out/record['id']
        folder.mkdir()
        image = ROOT/'data/MVTecAD_supervised_v1'/record['image']
        mask = ROOT/'data/MVTecAD_supervised_v1'/record['mask']
        assert sha(image) == record['image_sha256'] and sha(mask) == record['mask_sha256']
        result = client.predict(handle_file(str(image)), 'Original FINet', .5,
                                handle_file(str(mask)), 'Factory Quality Inspection', api_name='/segment')
        assert len(result) == 7 and 'Unable' not in result[5], result[5]
        with ZipFile(result[6]) as zipped:
            assert zipped.testzip() is None
            assert set(zipped.namelist()) == {'original.png', 'probability.png', 'probability.npy',
                                              'binary.png', 'boundary.png', 'overlay.png', 'prediction.json'}
            metadata = json.loads(zipped.read('prediction.json'))
            assert metadata['checkpoint_sha256'] == CHECKPOINT_SHA
            assert metadata['application_domain'] == 'Factory Quality Inspection'
            assert metadata['dataset_protocol'] == 'custom_supervised_mvtec_v1'
            assert metadata['training_progress'] == {'completed_epochs': 200, 'planned_epochs': 200,
                                                       'training_complete': True}
            assert metadata['strict_checkpoint_loading'] and metadata['checkpoint_state_tensors'] == 965
            assert metadata['threshold'] == .5 and metadata['output_size_wh'] == record['size_wh']
            probability = np.load(io.BytesIO(zipped.read('probability.npy')), allow_pickle=False)
            for name in zipped.namelist():
                (folder/name).write_bytes(zipped.read(name))
        shutil.copy2(mask, folder/'ground_truth.png')
        with Image.open(mask) as source:
            target = np.asarray(source.convert('L')) >= 128
        assert probability.shape == target.shape == tuple(reversed(record['size_wh']))
        assert np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1
        measured = segmentation_metrics(probability, target, .5)
        prediction = probability >= .5
        intersection = int((prediction & target).sum())
        total, union = int(prediction.sum() + target.sum()), int((prediction | target).sum())
        assert measured['dice'] == (2*intersection/total if total else 1.)
        assert measured['iou'] == (intersection/union if union else 1.)
        for name, value in measured.items():
            assert np.isclose(value, metadata['ground_truth_metrics'][name], rtol=0, atol=1e-12), name
        with Image.open(folder/'binary.png') as binary:
            assert np.array_equal(np.asarray(binary) > 0, prediction)
        sample = {'id': record['id'], 'title': title, 'category': record['category'],
                  'defect_type': record['defect_type'], 'is_defective': record['is_defective'],
                  'split': 'test', 'native_size_wh': record['size_wh'], 'image_sha256': sha(image),
                  'ground_truth_sha256': sha(mask), 'threshold': .5, **measured,
                  'predicted_foreground_percent': float(100*prediction.mean()),
                  'inference_ms': metadata['inference_ms'], 'processing_ms': metadata['processing_ms'],
                  'artifacts': str(folder), 'app_results_zip': result[6]}
        samples.append(sample)
        (folder/'measured_results.json').write_text(json.dumps(sample, indent=2)+'\n')
        for col, filename in enumerate(('original.png', 'ground_truth.png', 'binary.png', 'overlay.png')):
            with Image.open(folder/filename) as displayed:
                axes[row, col].imshow(displayed, cmap='gray' if col in (1, 2) else None, vmin=0, vmax=255)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(columns[col], fontsize=12, pad=10)
        axes[row, 0].set_ylabel(title, fontsize=12)
        axes[row, 2].set_xlabel(f"Dice {measured['dice']:.3f} | IoU {measured['iou']:.3f}", fontsize=10)
        axes[row, 3].set_xlabel(f"Boundary F1 {measured['boundary_f1']:.3f} | {sample['inference_ms']:.0f} ms", fontsize=10)
        print(json.dumps(sample), flush=True)
    figure.suptitle('Industrial FINet baseline | epoch 200 | three held-out examples', fontsize=15)
    figure.supxlabel('White = defect-region foreground. Normal-image scores are 1 only when both masks are empty.\n'
                       'Threshold 0.5; native-resolution metrics; boundary tolerance 2 pixels. This is a small spot check.', fontsize=10)
    figure.savefig(out/'comparison.png', dpi=160)
    figure.savefig(out/'comparison.pdf')
    plt.close(figure)
    report = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'app_url': args.url, 'checkpoint': str(checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA,
              'completed_epochs': 200, 'training_updates': 186400,
              'trained_split_fingerprint': canonical['fingerprint'],
              'selection': 'First test ID matching each predeclared category/defect condition; all selected before inference.',
              'threshold': .5, 'boundary_tolerance_native_pixels': 2,
              'empty_mask_convention': 'Dice/IoU/BF1: both empty=1; only one empty=0.',
              'scope': 'Three individual held-out examples through the actual app; no aggregate accuracy or product-verdict claim.',
              'samples': samples}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['id','is_defective','dice','iou','mae','boundary_f1','predicted_pixels',
                  'target_pixels','predicted_foreground_percent','inference_ms','processing_ms']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    with ZipFile(out/'industrial_test_results.zip', 'w', ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'industrial_test_results.zip' and 'downloads' not in path.relative_to(out).parts:
                zipped.write(path, path.relative_to(out).as_posix())
    assert sha(checkpoint) == CHECKPOINT_SHA
    print(json.dumps({'status': 'passed', 'samples': len(samples), 'report': str(out/'report.json'),
                      'figure': str(out/'comparison.png'), 'download': str(out/'industrial_test_results.zip')}), flush=True)


if __name__ == '__main__':
    main()
