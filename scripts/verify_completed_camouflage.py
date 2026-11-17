"""Independently verify a saved, full-budget camouflage recovery archive on CPU."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.kaggle_migration import equal_state, validate_payload
from src.training import (TrainingConfig, capture_rng, make_model_optimizer,
                          restore_checkpoint, write_json)
from src.training_data import file_hash


def verify(archive, destination, report):
    torch.set_num_threads(2)
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    backup = destination.parent / 'kaggle_version005_recovery.zip'
    digest = file_hash(archive)
    if backup.exists():
        assert file_hash(backup) == digest, 'Existing backup differs; preserve both.'
    else:
        shutil.copy2(archive, backup)
    with ZipFile(backup) as zipped:
        assert zipped.testzip() is None, 'Archive CRC failure.'
        members = zipped.infolist()
        for member in members:
            parts = PurePosixPath(member.filename).parts
            assert parts and parts[0] == 'FINet'
            assert '\\' not in member.filename and all(p not in {'.', '..'} and ':' not in p for p in parts)
            target = destination.joinpath(*parts)
            assert target.resolve().is_relative_to(destination.resolve())
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = zipped.read(member)
            if target.exists():
                assert target.read_bytes() == data, f'Existing evidence differs: {target}'
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    work = destination / 'FINet'
    manifest = json.loads((work / 'splits/cod_camo_seed123456.json').read_text())
    results, configs = {}, []
    for name in ('baseline', 'boundary'):
        run = work / f'experiments/{name}_seed123456_b4_v1'
        path = run / 'last.pth'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config = TrainingConfig(**payload['config'])
        summary = validate_payload(payload, config, manifest)
        assert summary['completed_epochs'] == config.epochs == 200
        model, optimizer, scheduler = make_model_optimizer(config, torch.device('cpu'), load_path=path)
        generator = torch.Generator().manual_seed(config.seed)
        restore_checkpoint(path, model, optimizer, scheduler, generator, config, manifest)
        equal_state(model.state_dict(), payload['model_state_dict'])
        equal_state(optimizer.state_dict(), payload['optimizer_state_dict'])
        equal_state(scheduler.state_dict(), payload['scheduler_state_dict'])
        rng = capture_rng(generator)
        expected_rng = {**payload['rng_state'], 'cuda': []}
        equal_state(rng, expected_rng)
        history = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines() if line.strip()]
        history = [row for row in history if 'epoch' in row]
        assert [row['epoch'] for row in history] == list(range(1, 201))
        best = torch.load(run / 'best.pth', map_location='cpu', weights_only=True)
        results[name] = {**summary, 'checkpoint_sha256': file_hash(path),
                         'strict_cpu_restore': 'exact model/Adam/scheduler/Python/NumPy/torch/loader RNG',
                         'cuda_rng_states_retained': len(payload['rng_state']['cuda']),
                         'epoch_history_contiguous': True, 'epoch_history_count': len(history),
                         'last_epoch_log': history[-1], 'best_checkpoint_epoch': best['completed_epochs'],
                         'best_checkpoint_sha256': file_hash(run / 'best.pth'),
                         'boundary_lambda': config.boundary_lambda}
        configs.append(asdict(config))
        del model, optimizer, scheduler, payload, best
    differences = sorted(k for k in configs[0] if configs[0][k] != configs[1][k])
    assert set(differences) <= {'boundary_lambda', 'initial_checkpoint', 'output_dir'}
    assert results['baseline']['checkpoint_sha256'] == '660a43bc1f5d57cbfd58d5cec7dd452f592932f34e1b8931567a30784af62810'
    evidence = {'script_version_id': 369255109, 'verified_date': '2026-11-14',
                'source': 'Actual Kaggle saved Version 5 downloaded through Chrome',
                'platform_result': 'Successfully ran in 15394.4 seconds',
                'archive_bytes': backup.stat().st_size, 'archive_members': len(members),
                'archive_sha256': digest, 'all_member_crc_checks_passed': True,
                **results, 'matched_configuration_differences': differences,
                'remaining_camouflage_training_epochs': 0,
                'scope': 'Training completion verified; held-out evaluation and app comparison are separate.'}
    write_json(Path(report), evidence)
    print(json.dumps({'archive_sha256': digest, 'epochs': {k: v['completed_epochs'] for k, v in results.items()},
                      'checkpoint_sha256': {k: v['checkpoint_sha256'] for k, v in results.items()},
                      'report': str(report)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=ROOT / 'outputs/kaggle_migration/version005')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/kaggle_migration/kaggle_version005_completion_verification.json')
    args = parser.parse_args()
    verify(args.archive, args.destination, args.report)
