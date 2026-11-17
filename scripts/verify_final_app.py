"""Exercise the app with both genuine, matched epoch-200 camouflage models."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch

from src.app_service import AppService, ModelRegistry, BASELINE, BOUNDARY
from src.training import write_json


def main():
    torch.set_num_threads(2)
    verification = json.loads((ROOT / 'outputs/kaggle_migration/kaggle_version005_completion_verification.json').read_text())
    registry = ModelRegistry(ROOT / 'checkpoints/baseline/last.pth', ROOT / 'checkpoints/boundary/last.pth', 'cpu')
    audit = json.loads((ROOT / 'outputs/test_data_audit.json').read_text())
    dataset = audit['datasets']['CAMO']
    excluded = set(dataset['development_overlap_ids'])
    record = next(row for row in dataset['records'] if row['id'] not in excluded)
    image = ROOT / 'data/TestDataset/CAMO' / record['image']
    target = ROOT / 'data/TestDataset/CAMO' / record['mask']
    expected_size = Image.open(image).size
    service = AppService(registry, ROOT / 'outputs/final_camouflage_app')
    left, right = service.compare(str(image), 0.5, str(target))
    reports = {}
    for key, name, result in (('baseline', BASELINE, left), ('boundary', BOUNDARY, right)):
        predictor = registry.get(name)
        assert predictor.training_progress['training_complete']
        assert predictor.training_progress['completed_epochs'] == 200
        assert predictor.checkpoint_sha256 == verification[key]['checkpoint_sha256']
        assert all(item.size == expected_size for item in result[:5])
        archive = Path(result[-1])
        with ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            names = zipped.namelist()
            assert {'probability.png', 'binary.png', 'overlay.png', 'boundary.png', 'prediction.json'} <= set(names)
            metadata = json.loads(zipped.read('prediction.json'))
        reports[key] = {'checkpoint_sha256': predictor.checkpoint_sha256,
                        'training_progress': predictor.training_progress,
                        'archive': str(archive), 'artifacts': names,
                        'native_size': list(expected_size),
                        'inference_ms': metadata['inference_ms'],
                        'processing_ms': metadata['processing_ms'],
                        'ground_truth_metrics': metadata['ground_truth_metrics']}
    assert registry.get(BASELINE).comparison_provenance == registry.get(BOUNDARY).comparison_provenance
    report = {'status': 'passed', 'scope': 'One real held-out app comparison; not aggregate benchmark evidence.',
              'device': 'cpu', 'sample_id': record['id'], 'threshold': 0.5, **reports}
    destination = ROOT / 'outputs/final_camouflage_app/verification.json'
    write_json(destination, report)
    print(json.dumps({'status': 'passed', 'sample_id': record['id'], 'report': str(destination),
                      'inference_ms': {k: v['inference_ms'] for k, v in reports.items()}}), flush=True)


if __name__ == '__main__':
    main()
