"""Protocol guards use unit fixtures; these are not research measurements."""
from dataclasses import asdict
import json
import math
import hashlib
from pathlib import Path

import pytest
import torch

from scripts.kaggle_migration import (INITIAL, SPLIT, equal_state, latest_checkpoint,
                                      relocated_config, validate_payload, verify_recovery_files,
                                      cache_training_pairs)
from src.training import TrainingConfig, run_signature


def fixture_payload():
    config = TrainingConfig()
    manifest = {'fingerprint': SPLIT, 'train_ids': list(range(3636))}
    epoch = 14
    lr = (config.learning_rate - config.min_lr) / 2 * math.cos(math.pi / 199 * epoch) + (config.learning_rate + config.min_lr) / 2
    payload = {'training_status': 'baseline_training', 'run_signature': run_signature(config, manifest),
               'split_fingerprint': SPLIT, 'initial_model_sha256': INITIAL,
               'completed_epochs': epoch, 'global_step': 12726, 'best_validation_mae': .1,
               'model_state_dict': {str(i): torch.ones(1) for i in range(965)},
               'optimizer_state_dict': {'state': {i: {'step': torch.tensor(12726.), 'exp_avg': torch.zeros(1)}
                                               for i in range(581)}, 'param_groups': [{'lr': lr}]},
               'scheduler_state_dict': {'cnt': epoch, 'max_epoch': 200, 'max_lr': config.learning_rate,
                                        'min_lr': config.min_lr, 'current_lr': lr},
               'rng_state': dict.fromkeys(('python', 'numpy', 'torch', 'cuda', 'loader'))}
    return payload, config, manifest


def test_relocation_preserves_every_training_setting():
    saved = asdict(TrainingConfig())
    migrated = relocated_config(saved, '/input/data', '/work/split', '/work/run', '/input/initial')
    for key in ('seed', 'batch_size', 'epochs', 'learning_rate', 'min_lr', 'weight_decay',
                'num_workers', 'cpu_threads', 'cudnn_enabled', 'image_size', 'boundary_lambda'):
        assert migrated[key] == saved[key]
    assert run_signature(TrainingConfig(**migrated), {'fingerprint': SPLIT}) == run_signature(TrainingConfig(**saved), {'fingerprint': SPLIT})


def test_saved_epoch_fourteen_continues_schedule():
    payload, config, manifest = fixture_payload()
    report = validate_payload(payload, config, manifest)
    assert report['next_epoch'] == 15
    assert report['global_step'] == 12726
    assert report['next_learning_rate'] == pytest.approx(.00025683897921066374)


@pytest.mark.parametrize('field,value', [('completed_epochs', 0), ('global_step', 12725),
    ('training_status', 'smoke_test'), ('split_fingerprint', 'changed'), ('initial_model_sha256', 'changed')])
def test_invalid_recovery_is_refused(field, value):
    payload, config, manifest = fixture_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_payload(payload, config, manifest)


def test_scheduler_restart_and_missing_optimizer_are_refused():
    payload, config, manifest = fixture_payload()
    payload['scheduler_state_dict']['cnt'] = 0
    with pytest.raises(ValueError, match='schedule'):
        validate_payload(payload, config, manifest)
    payload, config, manifest = fixture_payload()
    payload['optimizer_state_dict']['state'].pop(0)
    with pytest.raises(ValueError, match='Adam'):
        validate_payload(payload, config, manifest)


def test_exact_state_comparison_detects_changes():
    equal_state({'a': [torch.ones(2), 3]}, {'a': [torch.ones(2), 3]})
    with pytest.raises(AssertionError):
        equal_state({'a': torch.ones(2)}, {'a': torch.zeros(2)})


def test_latest_valid_checkpoint_wins_and_corrupt_candidate_is_recorded(tmp_path):
    payload, config, manifest = fixture_payload()
    newest = tmp_path / 'newest.pth'
    torch.save(payload, newest)
    old = tmp_path / 'old.pth'
    payload['completed_epochs'] = 13
    payload['global_step'] = 13 * 909
    payload['scheduler_state_dict']['cnt'] = 13
    lr = (config.learning_rate-config.min_lr)/2*math.cos(math.pi/199*13)+(config.learning_rate+config.min_lr)/2
    payload['scheduler_state_dict']['current_lr'] = lr
    payload['optimizer_state_dict']['param_groups'][0]['lr'] = lr
    for state in payload['optimizer_state_dict']['state'].values():
        state['step'] = torch.tensor(float(13*909))
    torch.save(payload, old)
    corrupt = tmp_path / 'corrupt.pth'
    corrupt.write_bytes(b'incomplete interrupted transfer')
    selected, summary = latest_checkpoint([old, corrupt, newest], config, manifest)
    assert selected == newest
    assert summary['completed_epochs'] == 14
    assert len(summary['invalid_candidates']) == 1


def test_kaggle_expanded_legacy_source_archive_is_optional_but_weights_are_required(tmp_path):
    entry = {'sha256': 'unused-for-expanded-legacy-archive', 'bytes': 12}
    assert verify_recovery_files(tmp_path, {'code/finet_colab_training.zip': entry}) == ['code/finet_colab_training.zip']
    with pytest.raises(FileNotFoundError):
        verify_recovery_files(tmp_path, {'experiments/baseline_seed123456_b4_v1/last.pth': entry})
    (tmp_path / 'code').mkdir()
    (tmp_path / 'code/finet_colab_training.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        verify_recovery_files(tmp_path, {'code/finet_colab_training.zip': entry})


def test_data_cache_retains_exact_bytes_and_refuses_corruption(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'cache'
    source.mkdir()
    image, mask = b'original image bytes', b'original mask bytes'
    (source / 'image.jpg').write_bytes(image)
    (source / 'mask.png').write_bytes(mask)
    manifest = {'records': [{'image': 'image.jpg', 'mask': 'mask.png',
                            'image_sha256': hashlib.sha256(image).hexdigest(),
                            'mask_sha256': hashlib.sha256(mask).hexdigest()}]}
    cache_training_pairs(source, destination, manifest, workers=2)
    assert (destination / 'image.jpg').read_bytes() == image
    assert (destination / 'mask.png').read_bytes() == mask
    cache_training_pairs(source, destination, manifest, workers=2)
    (destination / 'image.jpg').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='differs'):
        cache_training_pairs(source, destination, manifest, workers=2)
