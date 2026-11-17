"""Resume industrial boundary; save at epoch 200 or once before the session limit."""

import argparse
import io
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.kaggle_migration import cache_training_pairs, equal_state
from scripts.run_industrial_kaggle import checkpoint_rank, recovery_zip, restore_latest
from src.industrial_data import load_industrial_manifest
from src.industrial_training import IndustrialConfig, make_industrial_loaders, run_industrial_training
from src.training import (capture_rng, make_model_optimizer, restore_checkpoint,
                          seed_everything, train_step, write_json)
from src.training_data import file_hash, manifest_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--session-start-monotonic', type=float,
                        help='Notebook first-cell monotonic start; includes setup in the 12-hour allowance.')
    args = parser.parse_args()
    session_started = args.session_start_monotonic if args.session_start_monotonic is not None else monotonic()
    if not 0 <= monotonic() - session_started < 12*3600:
        raise ValueError('Session start must be a valid monotonic time in the last 12 hours.')
    deadline = session_started + 12*3600 - 15*60
    if not torch.cuda.is_available():
        raise RuntimeError('A genuine CUDA GPU is required; no CPU long training.')
    state = Path('/kaggle/working/FINetIndustrial')
    selected = restore_latest('/kaggle/input', state)
    print(json.dumps({'event': 'final_only_recovery_selected', **selected}), flush=True)
    paths = {v: state / f'experiments/{v}/last.pth' for v in ('baseline', 'boundary')}
    digests = {v: file_hash(p) for v, p in paths.items()}
    payloads = {v: torch.load(p, map_location='cpu', weights_only=True) for v, p in paths.items()}
    a, b = payloads['baseline'], payloads['boundary']
    assert checkpoint_rank(a) == 200 and a['config']['epochs'] == 200
    assert digests['baseline'] == 'fd92df468740e977fb16e9bcc6f44dfcf424be8822d155f36ea4cc209d4db94a'
    assert checkpoint_rank(b) >= 6 and b['config']['epochs'] == 200
    if b['completed_epochs'] == 6:
        assert digests['boundary'] == 'ffd2527504cabc50de0260a1e2cd01fc136527948e80295bbbcd177726359954'
    assert a['initial_model_sha256'] == b['initial_model_sha256']
    assert a['split_fingerprint'] == b['split_fingerprint'] == '219361ad92613f34bb36fe9beee42998dfe03f49aa436aedc0c168bafb0980fa'
    manifest = json.loads((state / 'split_manifest.json').read_text())
    assert manifest_hash(manifest) == manifest['fingerprint'] == b['split_fingerprint']
    caches = sorted(Path('/kaggle/input').rglob('MVTecAD_supervised_v1/split_manifest.json'))
    assert caches, 'Attach the separate frozen canonical prepared cache.'
    original = next(p.parent for p in caches if json.loads(p.read_text())['fingerprint'] == manifest['fingerprint'])
    data = cache_training_pairs(original, '/kaggle/temp/industrial_final_data', manifest)
    write_json(data / 'split_manifest.json', manifest)
    load_industrial_manifest(data / 'split_manifest.json', data)
    config = IndustrialConfig(**{**b['config'], 'data_root': str(data),
                                'split_manifest': str(data / 'split_manifest.json'),
                                'output_dir': str(paths['boundary'].parent), 'device': 'cuda:0',
                                'initial_checkpoint': str(state / 'smoke/boundary/initialization.pth')})
    config.validate()
    device = torch.device(config.device)
    seed_everything(config)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=paths['boundary'])
    generator = torch.Generator()
    restore_checkpoint(paths['boundary'], model, optimizer, scheduler, generator, config, manifest)
    equal_state(model.state_dict(), b['model_state_dict'])
    equal_state(optimizer.state_dict(), b['optimizer_state_dict'])
    equal_state(scheduler.state_dict(), b['scheduler_state_dict'])
    equal_state(capture_rng(generator), b['rng_state'])
    train, _ = make_industrial_loaders(config, manifest, generator)
    batch = next(iter(train))
    step = train_step(model, optimizer, batch, device, verify_updates=True,
                      boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    # Test serialization in memory; no engineering/intermediate checkpoint file.
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu', weights_only=True)
    equal_state(restored['model'], model.state_dict())
    equal_state(restored['optimizer'], optimizer.state_dict())
    equal_state(restored['scheduler'], scheduler.state_dict())
    assert all(file_hash(paths[v]) == digests[v] for v in paths)
    print(json.dumps({'event': 'boundary_final_only_gpu_probe_passed', 'gpu': torch.cuda.get_device_name(0),
                      'restored_epoch': b['completed_epochs'], 'next_epoch': b['completed_epochs'] + 1,
                      'step': step, 'baseline_sha256_unchanged': digests['baseline'],
                      'boundary_recovery_sha256_unchanged': digests['boundary'],
                      'probe_storage': 'memory only; zero checkpoint files written'}), flush=True)
    del model, optimizer, scheduler, restored, buffer, train, batch, payloads, a, b
    torch.cuda.empty_cache()
    if args.probe_only:
        return
    proof = state / 'smoke/boundary/smoke_report.json'
    print(json.dumps({'event': 'SESSION_END_CHECKPOINT_POLICY',
                      'policy': 'No per-epoch saves; one full-state save at session end or final epoch 200.',
                      'session_limit_hours': 12, 'save_margin_minutes': 15,
                      'seconds_until_save_deadline': deadline-monotonic()}), flush=True)
    result = run_industrial_training(config, proof, str(paths['boundary']), final_only=True,
                                     session_checkpoint_deadline=deadline)
    assert result['training_complete'] or result['session_checkpoint_saved']
    assert file_hash(paths['baseline']) == digests['baseline']
    saved = torch.load(paths['boundary'], map_location='cpu', weights_only=True)
    assert checkpoint_rank(saved) == result['completed_epochs']
    print(json.dumps({'event': 'industrial_boundary_final_complete' if result['training_complete']
                      else 'industrial_boundary_session_saved', **result,
                      'checkpoint_sha256': file_hash(paths['boundary'])}), flush=True)
    archive = recovery_zip(state, '/kaggle/working/industrial_recovery.zip')
    print(json.dumps({'event': 'industrial_recovery_saved', **archive}), flush=True)


if __name__ == '__main__':
    main()
