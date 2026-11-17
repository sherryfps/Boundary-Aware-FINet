"""One real-batch recovery probe; never modifies the official epoch checkpoint."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.kaggle_migration import equal_state
from scripts.run_industrial_kaggle import checkpoint_rank
from src.industrial_training import IndustrialConfig, make_industrial_loaders
from src.training import (capture_rng, hardware_report, make_model_optimizer, restore_checkpoint,
                          save_checkpoint, seed_everything, train_step, write_json)
from src.training_data import file_hash, manifest_hash


def probe(state, device='cuda:0', data_root=None, output=None, minimum_epoch=96,
          expected_sha='242b7cbf7200d41554087bc9b5a31a68eec081dc4f9bedf56fa15374af37c1a1'):
    state, device = Path(state), torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('A real CUDA GPU is required for the Kaggle resume probe.')
    original = state/'experiments/baseline/last.pth'
    before = file_hash(original)
    payload = torch.load(original, map_location='cpu', weights_only=True)
    torch.set_num_threads(payload['config']['cpu_threads'])
    completed = checkpoint_rank(payload)
    if completed < minimum_epoch or (completed == minimum_epoch and before != expected_sha):
        raise ValueError('Attach the independently verified epoch-96 state or a newer valid state.')
    manifest = json.loads((state/'split_manifest.json').read_text())
    if manifest_hash(manifest) != manifest['fingerprint']:
        raise ValueError('Recovery manifest fingerprint failed.')
    active_data = data_root or json.loads((state/'data_location.json').read_text())['data_root']
    config = IndustrialConfig(**{**payload['config'], 'device':str(device),
                                'data_root':str(active_data), 'split_manifest':str(state/'split_manifest.json')})
    config.validate()
    seed_everything(config)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=original)
    generator = torch.Generator().manual_seed(config.seed)
    restore_checkpoint(original, model, optimizer, scheduler, generator, config, manifest)
    equal_state(model.state_dict(), payload['model_state_dict'])
    equal_state(optimizer.state_dict(), payload['optimizer_state_dict'])
    equal_state(scheduler.state_dict(), payload['scheduler_state_dict'])
    rng_keys = ('python','numpy','torch','loader','cuda') if device.type == 'cuda' else ('python','numpy','torch','loader')
    equal_state({k:capture_rng(generator)[k] for k in rng_keys}, {k:payload['rng_state'][k] for k in rng_keys})
    loader, _ = make_industrial_loaders(config, manifest, generator)
    batch = next(iter(loader))
    step = train_step(model, optimizer, batch, device, verify_updates=True,
                      boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    model.eval()
    inputs = [batch[i].to(device) for i in (0,2,3)]
    with torch.no_grad():
        expected = tuple(value.cpu() for value in model(*inputs))
    if len(expected) != 4 or any(tuple(v.shape) != (config.batch_size,1,384,384) or not torch.isfinite(v).all() for v in expected):
        raise ValueError('Invalid four-head FINet recovery output.')
    output = Path(output or state/'migration'/f'resume_probe_epoch{completed:03d}')
    test_path = output/'engineering_checkpoint.pth'
    save_checkpoint(test_path, model, optimizer, scheduler, generator, config, manifest,
                    completed, payload['global_step']+1, payload['best_validation_mae'], status='smoke_test')
    other, other_optimizer, other_scheduler = make_model_optimizer(config, device, load_path=test_path)
    other_generator = torch.Generator()
    test_state = restore_checkpoint(test_path, other, other_optimizer, other_scheduler,
                                    other_generator, config, manifest, allow_smoke=True)
    equal_state(other.state_dict(), model.state_dict())
    equal_state(other_optimizer.state_dict(), optimizer.state_dict())
    equal_state(other_scheduler.state_dict(), scheduler.state_dict())
    equal_state({k:capture_rng(other_generator)[k] for k in rng_keys}, {k:test_state['rng_state'][k] for k in rng_keys})
    other.eval()
    with torch.no_grad():
        actual = tuple(value.cpu() for value in other(*inputs))
    equal_state(actual, expected)
    if file_hash(original) != before:
        raise RuntimeError('The isolated probe modified the official checkpoint.')
    report = {'event':'industrial_resume_probe_passed', 'status':'passed',
              'scope':'isolated engineering update, excluded from official epoch history',
              'restored_completed_epochs':completed, 'restored_global_step':payload['global_step'],
              'next_epoch':completed+1, 'next_learning_rate':scheduler.get_lr(),
              'official_checkpoint_sha256_unchanged':before, 'step':step,
              'exact_model_optimizer_scheduler_rng_restore':True,
              'exact_test_checkpoint_and_four_output_reload':True,
              'cuda_rng_restored':device.type == 'cuda',
              'saved_cuda_rng_states':len(payload['rng_state']['cuda']),
              'hardware':hardware_report(device), 'test_checkpoint_sha256':file_hash(test_path)}
    write_json(output/'report.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', default='/kaggle/working/FINetIndustrial')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-root')
    parser.add_argument('--output')
    args = parser.parse_args()
    probe(args.state, args.device, args.data_root, args.output)
