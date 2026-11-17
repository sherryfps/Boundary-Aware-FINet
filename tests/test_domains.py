"""Fail-closed application domain routing, using explicitly mocked unit fixtures."""

from types import SimpleNamespace
import pytest

from src.app_service import BASELINE, BOUNDARY, CAMOUFLAGE, INDUSTRIAL, ModelRegistry
from src.finet_adapter import CheckpointError


def mock_predictor(domain=CAMOUFLAGE, status='baseline_training', protocol=None, completed=200):
    return SimpleNamespace(application_domain=domain, training_status=status, dataset_protocol=protocol,
                           training_progress={'completed_epochs': completed, 'planned_epochs': 200, 'training_complete': completed==200},
                           parameter_count=3742988, device='cpu')


def test_camouflage_cannot_be_loaded_as_industrial_even_with_same_path(tmp_path, monkeypatch):
    path = tmp_path/'fixture.pth'; path.write_bytes(b'mocked unit fixture only')
    monkeypatch.setattr('src.app_service.FINetPredictor', lambda *_: mock_predictor())
    registry = ModelRegistry(path, tmp_path/'missing', industrial_baseline=path)
    assert registry.get(BASELINE, CAMOUFLAGE).application_domain == CAMOUFLAGE
    with pytest.raises(CheckpointError, match='domain'):
        registry.get(BASELINE, INDUSTRIAL)
    assert registry.available(INDUSTRIAL) == []


def test_industrial_requires_provenance_and_genuine_epoch(tmp_path, monkeypatch):
    path = tmp_path/'fixture.pth'; path.write_bytes(b'mocked unit fixture only')
    registry = ModelRegistry(tmp_path/'missing', tmp_path/'missing2', industrial_baseline=path)
    for protocol, completed in ((None,200), ('custom_supervised_mvtec_v1',0)):
        monkeypatch.setattr('src.app_service.FINetPredictor', lambda *_, p=protocol,c=completed: mock_predictor(INDUSTRIAL,protocol=p,completed=c))
        with pytest.raises(CheckpointError, match='provenance'):
            registry.get(BASELINE, INDUSTRIAL)
    monkeypatch.setattr('src.app_service.FINetPredictor', lambda *_: mock_predictor(INDUSTRIAL,protocol='custom_supervised_mvtec_v1',completed=1))
    assert registry.available(INDUSTRIAL) == [BASELINE]
    with pytest.raises(CheckpointError, match='domain'):
        ModelRegistry(path, tmp_path/'missing').get(BASELINE, CAMOUFLAGE)


def test_industrial_variant_mismatch_is_rejected(tmp_path, monkeypatch):
    path = tmp_path/'fixture.pth'; path.write_bytes(b'mocked unit fixture only')
    monkeypatch.setattr('src.app_service.FINetPredictor', lambda *_: mock_predictor(INDUSTRIAL,protocol='custom_supervised_mvtec_v1'))
    registry = ModelRegistry(tmp_path/'missing', tmp_path/'missing2', industrial_boundary=path)
    with pytest.raises(CheckpointError, match='boundary-aware'):
        registry.get(BOUNDARY, INDUSTRIAL)
