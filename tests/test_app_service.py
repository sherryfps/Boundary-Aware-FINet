"""Artifact and error tests using explicitly synthetic unit fixtures only."""

import json
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pytest
from src.app_service import AppService, BASELINE, BOUNDARY, ModelRegistry


def test_missing_checkpoint_and_invalid_model_are_actionable(tmp_path):
    registry = ModelRegistry(tmp_path/'a.pth', tmp_path/'b.pth')
    assert 'pending' in registry.describe()
    with pytest.raises(FileNotFoundError, match='trained checkpoint'):
        registry.get(BASELINE)
    with pytest.raises(ValueError, match='Choose'):
        registry.get('invalid')
    with pytest.raises(ValueError, match='Upload'):
        AppService(registry, tmp_path/'output').predict(None, BASELINE)


def test_render_download_threshold_and_metadata(tmp_path):
    registry = ModelRegistry(tmp_path/'a.pth', tmp_path/'b.pth')
    service = AppService(registry, tmp_path/'unit_output')
    probability = np.zeros((32, 40), dtype=np.float32)
    probability[8:20, 10:30] = 0.7
    state = {'original': Image.new('RGB', (40, 32), (40, 60, 80)),
             'probability': probability, 'model_name': 'Synthetic unit fixture',
             'metadata': {'inference_ms': 2.5, 'processing_ms': 3.0, 'ground_truth_metrics': None,
                          'purpose': 'synthetic unit-test data; not model output'}}
    rendered = service.render(state, 0.5)
    assert rendered[2].getextrema() == (0, 255)
    assert rendered[4].getextrema() == (0, 255)
    with ZipFile(rendered[6]) as zipped:
        assert {'probability.png', 'binary.png', 'overlay.png', 'boundary.png', 'probability.npy',
                'prediction.json', 'original.png'} == set(zipped.namelist())
        report = json.loads(zipped.read('prediction.json'))
        assert report['ground_truth_metrics'] is None
        assert report['threshold'] == 0.5
    assert service.render(state, 0.9)[2].getextrema() == (0, 0)


def test_application_builds_without_claiming_a_trained_model(tmp_path):
    pytest.importorskip('gradio')
    from app import build_app
    app = build_app(AppService(ModelRegistry(tmp_path/'a.pth', tmp_path/'b.pth'), tmp_path/'out'))
    configuration = app.get_config_file()
    labels = [component.get('props', {}).get('label') for component in configuration['components']]
    assert 'Upload a photograph' in labels and 'Predicted boundary' in labels
    assert 'Matching object mask (PNG)' in labels
