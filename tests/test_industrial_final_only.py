"""Checkpoint-policy tests with synthetic unit fixtures, never research evidence."""

from pathlib import Path
import json
import pytest
import torch

import src.industrial_training as training
from src.training import run_signature
from src.training_data import file_hash


def prepare(tmp_path, monkeypatch, final_only, interrupt=False, deadline=None, clock=None):
    initial = tmp_path / 'initial.bin'
    initial.write_bytes(b'explicit synthetic unit fixture')
    config = training.IndustrialConfig(epochs=3, initial_checkpoint=str(initial),
                                       output_dir=str(tmp_path / 'run'), boundary_lambda=.1)
    manifest = {'fingerprint': 'synthetic-unit-split'}
    proof = tmp_path / 'smoke.json'
    proof.write_text(json.dumps({'status': 'passed', 'run_signature': run_signature(config, manifest),
                                'hardware': {'device': 'cuda:0', 'gpu': 'synthetic-unit-GPU'},
                                'initialization_sha256': file_hash(initial),
                                'measured_steady_step_seconds': .1}))
    monkeypatch.setattr(training, 'select_device', lambda _: torch.device('cuda:0'))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'synthetic-unit-GPU')
    monkeypatch.setattr(training, 'load_industrial_manifest', lambda *_: manifest)
    monkeypatch.setattr(training, 'seed_everything', lambda _: None)
    monkeypatch.setattr(training, 'hardware_report', lambda _: {'fixture': True})
    monkeypatch.setattr(training, 'make_model_optimizer', lambda *_args, **_kwargs: (
        object(), object(), type('Schedule', (), {'get_lr': lambda _: .00026, 'step': lambda _: None})()))
    monkeypatch.setattr(training, 'make_industrial_loaders', lambda *_: ([None], []))
    calls, saved, previews = [], [], []

    def step(*args, **kwargs):
        calls.append(True)
        if clock is not None:
            clock[0] += 300
        if interrupt and len(calls) == 2:
            raise RuntimeError('synthetic unit interruption')
        return {'loss': 1.0}

    def validate(_model, _loader, _device, output):
        previews.append(output)
        return {'macro_defective_category_dice': .5, 'mean_per_image': {'mae': .1}}

    monkeypatch.setattr(training, 'train_step', step)
    monkeypatch.setattr(training, 'validate_industrial', validate)
    monkeypatch.setattr(training, 'industrial_save', lambda path, *args: saved.append((Path(path).name, args[6])))
    return lambda: training.run_industrial_training(config, proof, final_only=final_only,
                    session_checkpoint_deadline=deadline), saved, previews


def test_final_only_saves_only_at_full_budget(tmp_path, monkeypatch):
    run, saved, previews = prepare(tmp_path, monkeypatch, True)
    result = run()
    assert result['training_complete'] and result['completed_epochs'] == 3
    assert saved == [('last.pth', 3)]
    assert previews == [None, None, None]


def test_interruption_before_final_epoch_writes_no_weights(tmp_path, monkeypatch):
    run, saved, _ = prepare(tmp_path, monkeypatch, True, interrupt=True)
    with pytest.raises(RuntimeError, match='synthetic unit interruption'):
        run()
    assert saved == []


def test_default_recovery_policy_is_preserved(tmp_path, monkeypatch):
    run, saved, previews = prepare(tmp_path, monkeypatch, False)
    assert run()['training_complete']
    assert saved == [('last.pth', 1), ('best.pth', 1), ('last.pth', 2), ('last.pth', 3)]
    assert all(path is not None for path in previews)


def test_session_deadline_saves_once_at_last_completed_epoch(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(training, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(training, 'perf_counter', lambda: clock[0])
    run, saved, previews = prepare(tmp_path, monkeypatch, True, deadline=1000., clock=clock)
    result = run()
    assert result['completed_epochs'] == 2 and not result['training_complete']
    assert result['session_checkpoint_saved']
    assert saved == [('last.pth', 2)]
    assert previews == [None, None]
    history = [json.loads(line) for line in (tmp_path/'run/epochs.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == [1, 2]


def test_final_epoch_before_deadline_saves_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'monotonic', lambda: 0.)
    run, saved, _ = prepare(tmp_path, monkeypatch, True, deadline=3600.)
    assert run()['training_complete']
    assert saved == [('last.pth', 3)]


def test_expired_deadline_does_not_overwrite_imported_state(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'monotonic', lambda: 1001.)
    run, saved, _ = prepare(tmp_path, monkeypatch, True, deadline=1000.)
    result = run()
    assert result['completed_epochs'] == 0 and not result['session_checkpoint_saved']
    assert saved == []
