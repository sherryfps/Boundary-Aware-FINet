"""Check experiment identity and reject misleading comparisons."""

import copy
import pytest
import torch
from src.finet_adapter import state_dict_sha256


def test_state_hash_ignores_mapping_order_but_detects_changed_tensor():
    state = {'buffer': torch.tensor(3), 'weight': torch.arange(12).reshape(3, 4).float()}
    assert state_dict_sha256(state) == state_dict_sha256(dict(reversed(list(state.items()))))
    changed = {key: value.clone() for key, value in state.items()}
    changed['weight'][0, 0] += 1
    assert state_dict_sha256(state) != state_dict_sha256(changed)


def test_comparison_audit_requires_shared_initial_state_and_boundary_objective():
    pytest.importorskip('py_sod_metrics')
    from evaluate_models import comparison_audit
    report = {'initial_model_sha256': 'unit-initial', 'training_split_fingerprint': 'unit-split',
              'completed_epochs': 2, 'training_config': {'seed': 2, 'epochs': 2, 'boundary_lambda': 0},
              'evaluated_image_hashes': ['unit-image'], 'protocol': {'threshold': 0.5}, 'parameter_count': 10}
    right = copy.deepcopy(report)
    right['training_config']['boundary_lambda'] = 0.1
    assert comparison_audit(report, right)['controlled_protocol_verified']
    right['initial_model_sha256'] = 'different'
    assert not comparison_audit(report, right)['controlled_protocol_verified']
    right = copy.deepcopy(report)
    assert not comparison_audit(report, right)['checks']['baseline_vs_boundary_objectives']
