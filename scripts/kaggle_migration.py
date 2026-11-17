"""Verify and resume the existing FINet experiment without changing its protocol."""

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.finet_adapter import extract_state_dict, state_dict_sha256
from src.training import (TrainingConfig, capture_rng, hardware_report, make_loaders,
                          make_model_optimizer, restore_checkpoint, run_signature,
                          save_checkpoint, seed_everything, train_step)
from src.training_data import file_hash, load_manifest
from scripts.run_colab_project import read_json, record

SPLIT = '2100aa40c52f01442d59b28973fee653eec655e370c8b1c1ff3c8d40cf61d724'
INITIAL = 'e7c3b4edd324ad589dc4404b952caaa7aefd17151d0844bf1b9219f7e2b578e9'
PATH_FIELDS = {'data_root', 'split_manifest', 'output_dir', 'initial_checkpoint', 'device'}


def relocated_config(saved, data, manifest, output, initial):
    result = {**saved, 'data_root': str(data), 'split_manifest': str(manifest),
              'output_dir': str(output), 'initial_checkpoint': str(initial), 'device': 'cuda:0'}
    differences = {k for k in saved.keys() | result.keys() if saved.get(k) != result.get(k)}
    if not differences <= PATH_FIELDS:
        raise ValueError(f'Migration changed training settings: {differences - PATH_FIELDS}')
    TrainingConfig(**result).validate()
    return result


def equal_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_state(a, b)
    else:
        assert actual == expected


def validate_payload(payload, config, manifest):
    if (payload.get('training_status') != ('boundary_training' if config.boundary_lambda else 'baseline_training')
            or payload.get('run_signature') != run_signature(config, manifest)
            or payload.get('split_fingerprint') != SPLIT
            or payload.get('initial_model_sha256') != INITIAL):
        raise ValueError('Checkpoint protocol, split, status or shared initialization does not match.')
    epoch, step = payload['completed_epochs'], payload['global_step']
    per_epoch = len(manifest['train_ids']) // config.batch_size
    if not isinstance(epoch, int) or not 1 <= epoch <= config.epochs or step != epoch * per_epoch:
        raise ValueError('Checkpoint is not a complete saved epoch of this experiment.')
    state = extract_state_dict(payload)
    if len(state) != 965 or any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Missing or non-finite FINet model tensors.')
    optimizer = payload['optimizer_state_dict']
    if len(optimizer['state']) != 581:
        raise ValueError('Missing Adam parameter states.')
    for value in optimizer['state'].values():
        if int(value['step']) != step or any(not torch.isfinite(v).all() for v in value.values() if isinstance(v, torch.Tensor)):
            raise ValueError('Invalid Adam update counter or non-finite state.')
    scheduler = payload['scheduler_state_dict']
    expected_lr = ((config.learning_rate - config.min_lr) / 2 * math.cos(math.pi / (config.epochs - 1) * epoch)
                   + (config.learning_rate + config.min_lr) / 2)
    if (scheduler['cnt'] != epoch or scheduler['max_epoch'] != config.epochs
            or scheduler['max_lr'] != config.learning_rate or scheduler['min_lr'] != config.min_lr
            or not math.isclose(scheduler['current_lr'], expected_lr, rel_tol=1e-13)
            or any(group['lr'] != scheduler['current_lr'] for group in optimizer['param_groups'])):
        raise ValueError('Saved learning-rate schedule does not continue at this epoch.')
    if set(payload['rng_state']) != {'python', 'numpy', 'torch', 'cuda', 'loader'}:
        raise ValueError('Incomplete random-number / data-loader recovery state.')
    return {'completed_epochs': epoch, 'global_step': step, 'next_epoch': epoch + 1,
            'next_learning_rate': scheduler['current_lr'], 'model_tensors': len(state),
            'adam_parameter_states': len(optimizer['state']), 'scheduler_count': scheduler['cnt'],
            'model_sha256': state_dict_sha256(state), 'initial_model_sha256': INITIAL,
            'split_fingerprint': SPLIT, 'best_validation_mae': payload['best_validation_mae']}


def latest_checkpoint(candidates, config, manifest):
    valid, invalid = [], []
    for path in sorted(set(map(Path, candidates))):
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            summary = validate_payload(payload, config, manifest)
            valid.append((path, summary))
        except Exception as exc:
            invalid.append({'path': str(path), 'error': str(exc)})
    if not valid:
        raise ValueError(f'No valid full-state checkpoint found. {invalid}')
    newest = max(s['completed_epochs'] for _, s in valid)
    winners = [(p, s) for p, s in valid if s['completed_epochs'] == newest]
    if len({s['model_sha256'] for _, s in winners}) != 1:
        raise ValueError('Conflicting model states at the latest epoch; preserve and inspect both.')
    return winners[0][0], {**winners[0][1], 'invalid_candidates': invalid}


def verify_recovery_files(recovery, entries):
    omitted = []
    for relative, expected in entries.items():
        source = Path(recovery) / relative
        # Kaggle expands nested .zip inputs. This historical source backup is
        # redundant: current project files and checkpoint training hashes are
        # verified separately. Model, data, split and logs are never optional.
        if relative == 'code/finet_colab_training.zip' and not source.is_file():
            omitted.append(relative)
            continue
        if file_hash(source) != expected['sha256'] or source.stat().st_size != expected['bytes']:
            raise ValueError(f'Transferred recovery file failed integrity verification: {relative}')
    return omitted


def cache_training_pairs(source, destination, manifest, workers=16):
    """Copy original bytes off Kaggle's cold input mount, verifying every hash."""
    source, destination = Path(source), Path(destination)
    jobs = [(record[key], record[key + '_sha256']) for record in manifest['records']
            for key in ('image', 'mask')]
    started = perf_counter()

    def copy_verified(job):
        relative, expected = job
        target = destination / relative
        if target.is_file():
            if file_hash(target) != expected:
                raise ValueError(f'Existing cached dataset differs: {relative}')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.partial')
        shutil.copyfile(source / relative, temporary)
        if file_hash(temporary) != expected:
            raise ValueError(f'Transferred dataset bytes differ: {relative}')
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(as_completed([executor.submit(copy_verified, job) for job in jobs]), 1):
            result.result()
            if index % 1000 == 0 or index == len(jobs):
                print('VERIFIED_LOCAL_DATA_CACHE', json.dumps({'files': index, 'total': len(jobs),
                      'seconds': perf_counter() - started}), flush=True)
    return destination


def prepare(transfer, work):
    transfer, work = Path(transfer), Path(work)
    inventory = read_json(transfer / 'transfer_manifest.json')
    omitted = verify_recovery_files(transfer / 'recovery', inventory['recovery_files'])
    recovery = transfer / 'recovery'
    for source in recovery.rglob('*'):
        if source.is_file():
            target = work / source.relative_to(recovery)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    manifest_path = work / 'splits/cod_camo_seed123456.json'
    portable_manifest = read_json(manifest_path)
    data = cache_training_pairs(transfer / 'data/TrainDataset',
                                Path('/kaggle/temp/finet_data/TrainDataset'), portable_manifest)
    manifest = load_manifest(manifest_path, data)
    if manifest['fingerprint'] != SPLIT or manifest['source_counts'] != {'COD10K': 3040, 'CAMO': 1000}:
        raise ValueError('The original official dataset / split was not retained.')
    output = work / 'experiments/baseline_seed123456_b4_v1'
    saved = read_json(recovery / 'experiments/baseline_seed123456_b4_v1/config.json')
    initial = work / 'experiments/baseline_seed123456_b4_v1_smoke_v1/initialization.pth'
    if state_dict_sha256(extract_state_dict(torch.load(initial, map_location='cpu', weights_only=True))) != INITIAL:
        raise ValueError('Transferred initial full-model weights differ.')
    config = TrainingConfig(**relocated_config(saved, data, manifest_path, output, initial))
    torch.set_num_threads(config.cpu_threads)
    candidates = [output / 'last.pth', recovery / 'experiments/baseline_seed123456_b4_v1/last.pth']
    # Saved notebook outputs may be attached as a newer read-only recovery input.
    candidates += list(Path('/kaggle/input').glob('**/baseline_seed123456_b4_v1/last.pth'))
    latest, summary = latest_checkpoint(candidates, config, manifest)
    payload = torch.load(latest, map_location='cpu', weights_only=True)
    for name, expected in payload['environment']['source_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f'Training source changed since the saved run: {name}')
    if latest.resolve() != (output / 'last.pth').resolve():
        if (output / 'last.pth').exists():
            shutil.copy2(output / 'last.pth', output / 'previous_import.pth')
        shutil.copy2(latest, output / 'last.pth')
        # Keep the selected validation state and history with its corresponding last state.
        for name in ('best.pth', 'epochs.jsonl', 'steps.jsonl'):
            if latest.with_name(name).is_file():
                shutil.copy2(latest.with_name(name), output / name)
    config_path = work / 'projects/finet_matched_v1/kaggle_baseline_config.json'
    record(config_path, asdict(config))
    summary.update(checkpoint=str(output / 'last.pth'), checkpoint_sha256=file_hash(output / 'last.pth'),
                   original_config=saved, migrated_config=asdict(config),
                   legacy_archives_expanded_by_kaggle=omitted,
                   scope='Existing epoch-boundary recovery. No training has run in this preparation step.')
    record(work / 'migration/checkpoint_verification.json', summary)
    # A later saved Kaggle version may also contain an interrupted boundary run.
    boundary_output = work / 'experiments/boundary_seed123456_b4_v1'
    boundary_candidates = [boundary_output / 'last.pth'] + list(
        Path('/kaggle/input').glob('**/boundary_seed123456_b4_v1/last.pth'))
    if any(path.is_file() for path in boundary_candidates):
        boundary_config = TrainingConfig(**{**asdict(config), 'boundary_lambda': 0.1,
                                             'output_dir': str(boundary_output)})
        boundary_latest, boundary_summary = latest_checkpoint(boundary_candidates, boundary_config, manifest)
        boundary_output.mkdir(parents=True, exist_ok=True)
        if boundary_latest.resolve() != (boundary_output / 'last.pth').resolve():
            if (boundary_output / 'last.pth').exists():
                shutil.copy2(boundary_output / 'last.pth', boundary_output / 'previous_import.pth')
            for name in ('last.pth', 'best.pth', 'epochs.jsonl', 'steps.jsonl'):
                if boundary_latest.with_name(name).is_file():
                    shutil.copy2(boundary_latest.with_name(name), boundary_output / name)
        record(work / 'migration/boundary_checkpoint_verification.json', boundary_summary)
    print('VERIFIED_RECOVERY', json.dumps(summary), flush=True)
    return config_path


def resume_probe(config_path):
    if not torch.cuda.is_available():
        raise RuntimeError('A verified Kaggle CUDA GPU is required.')
    config = TrainingConfig(**read_json(config_path))
    seed_everything(config)
    manifest = load_manifest(config.split_manifest, config.data_root)
    checkpoint = Path(config.output_dir) / 'last.pth'
    before_hash = file_hash(checkpoint)
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    summary = validate_payload(source, config, manifest)
    device = torch.device(config.device)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=checkpoint)
    generator = torch.Generator().manual_seed(config.seed)
    restore_checkpoint(checkpoint, model, optimizer, scheduler, generator, config, manifest)
    equal_state(model.state_dict(), source['model_state_dict'])
    equal_state(optimizer.state_dict(), source['optimizer_state_dict'])
    equal_state(scheduler.state_dict(), source['scheduler_state_dict'])
    recovered_rng = capture_rng(generator)
    # An extra idle GPU is available; only cuda:0 participates in this single-GPU run.
    recovered_rng['cuda'] = recovered_rng['cuda'][:len(source['rng_state']['cuda'])]
    equal_state(recovered_rng, source['rng_state'])
    loader, _ = make_loaders(config, manifest, generator)
    batch = next(iter(loader))
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = perf_counter()
    row = train_step(model, optimizer, batch, device, verify_updates=True,
                     boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    torch.cuda.synchronize(device)
    seconds = perf_counter() - started
    output = Path(config_path).parents[2] / 'migration' / f'resume_probe_epoch_{summary["completed_epochs"]:03d}'
    output.mkdir(parents=True, exist_ok=True)
    test_checkpoint = output / 'migration_smoke.pth'
    save_checkpoint(test_checkpoint, model, optimizer, scheduler, generator, config, manifest,
                    summary['completed_epochs'], summary['global_step'] + 1,
                    summary['best_validation_mae'], status='migration_smoke')
    model.eval()
    with torch.inference_mode():
        expected = [v.cpu() for v in model(batch[0].to(device), batch[2].to(device), batch[3].to(device))]
    other, other_optimizer, other_scheduler = make_model_optimizer(config, device, load_path=test_checkpoint)
    restore_checkpoint(test_checkpoint, other, other_optimizer, other_scheduler, generator, config, manifest, allow_smoke=True)
    equal_state(other.state_dict(), model.state_dict())
    equal_state(other_optimizer.state_dict(), optimizer.state_dict())
    equal_state(other_scheduler.state_dict(), scheduler.state_dict())
    other.eval()
    with torch.inference_mode():
        actual = other(batch[0].to(device), batch[2].to(device), batch[3].to(device))
    equal_state(actual, expected)
    assert file_hash(checkpoint) == before_hash, 'Smoke modified the official recovery checkpoint.'
    report = {'status': 'passed', 'scope': 'Isolated migrated-checkpoint GPU update; excluded from official training history.',
              'config': asdict(config), 'run_signature': run_signature(config, manifest),
              'hardware': hardware_report(device), 'initialization_sha256': file_hash(Path(config.initial_checkpoint)),
              'initial_model_sha256': INITIAL, 'split_fingerprint': SPLIT,
              'restored_checkpoint': summary, 'checkpoint_sha256_unchanged': before_hash,
              'input_shapes': [list(x.shape) for x in batch[:4]], 'output_shapes': [list(x.shape) for x in actual],
              'step': row, 'seconds': seconds, 'model_optimizer_scheduler_rng_restored_exactly': True,
              'test_checkpoint_reloaded_exactly': True, 'reloaded_output_equal': True,
              'initialization_note': 'The original pre-smoke full-model checkpoint is retained without reinitialization.'}
    report_path = output / 'smoke_report.json'
    record(report_path, report)
    print('KAGGLE_RESUME_SMOKE', json.dumps(report), flush=True)
    del model, other, optimizer, other_optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    return report_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--work', type=Path, default=Path('/kaggle/working/FINet'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    path = prepare(args.transfer, args.work)
    if args.smoke:
        resume_probe(path)
