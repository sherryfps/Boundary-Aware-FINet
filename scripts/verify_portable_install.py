"""Verify relocated final models and a real image without modifying checkpoints."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.finet_adapter import FINetPredictor


def main():
    torch.set_num_threads(2)
    expected = {'baseline': '660a43bc1f5d57cbfd58d5cec7dd452f592932f34e1b8931567a30784af62810',
                'boundary': 'c7859e2b2445dbc195dfc1f94908c862f678871e286eb6a1dd99ef45f3caee48'}
    sample = ROOT / 'data/TestDataset/CAMO/Imgs/camourflage_00012.jpg'
    with Image.open(sample) as image:
        width, height = image.size
    results = {}
    started = perf_counter()
    for name, digest in expected.items():
        path = ROOT / f'checkpoints/{name}/last.pth'
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, f'{name}: checkpoint changed'
        predictor = FINetPredictor(path, 'cpu')
        assert predictor.training_progress['completed_epochs'] == 200
        assert predictor.training_progress['training_complete']
        prediction = predictor.predict(sample)
        assert prediction.probability.shape == (height, width)
        assert np.isfinite(prediction.probability).all()
        assert 0 <= prediction.probability.min() <= prediction.probability.max() <= 1
        results[name] = {'completed_epochs': 200, 'checkpoint_sha256': digest,
                         'inference_ms': prediction.metadata['inference_ms'],
                         'output_shape': list(prediction.probability.shape)}
        del predictor, prediction
    report = {'status': 'passed', 'project_root': str(ROOT), 'python': sys.version,
              'device': 'cpu', 'scope': 'Relocated full models and real-image inference; not a benchmark.',
              'models': results, 'seconds': perf_counter() - started}
    destination = ROOT / 'outputs/new_laptop_verification.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
