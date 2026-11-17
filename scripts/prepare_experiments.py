"""Prepare matched A/B configurations from one complete shared starting state."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from Model.FINet import FINet
from src.finet_adapter import extract_state_dict, state_dict_sha256
from src.training import TrainingConfig, write_json
from src.training_data import file_hash


def prepare(config_path, initial_checkpoint, destination, boundary_lambda=0.1):
    config = TrainingConfig(**json.loads(Path(config_path).read_text(encoding='utf-8')))
    config.boundary_lambda = 0
    initial = Path(initial_checkpoint).resolve()
    payload = torch.load(initial, map_location='cpu', weights_only=True)
    state = extract_state_dict(payload)
    model = FINet(backbone='efficientb0', channels=(8, 24, 32, 64), pretrained_backbone=False)
    if state.keys() != model.state_dict().keys():
        raise ValueError('Shared initialization must contain every original FINet tensor.')
    model.load_state_dict(state, strict=True)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Experiment directory already contains files; preserve the original protocol.')
    config.initial_checkpoint = str(initial)
    a = asdict(config)
    a['output_dir'] = str(destination/'model_a')
    b = {**a, 'boundary_lambda': boundary_lambda, 'output_dir': str(destination/'model_b')}
    TrainingConfig(**a).validate(); TrainingConfig(**b).validate()
    if boundary_lambda <= 0:
        raise ValueError('Model B needs a positive boundary lambda.')
    differences = [key for key in a if a[key] != b[key]]
    assert set(differences) == {'boundary_lambda', 'output_dir'}
    write_json(destination/'model_a.json', a)
    write_json(destination/'model_b.json', b)
    protocol = {'status': 'configuration only; no training or improvement claimed',
                'shared_initial_checkpoint': str(initial), 'shared_file_sha256': file_hash(initial),
                'shared_model_state_sha256': state_dict_sha256(state),
                'configuration_differences': differences,
                'next': 'Measure separate GPU smoke runs for A and B, keeping identical seed, data and schedule.'}
    write_json(destination/'protocol.json', protocol)
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.json')
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--output', default='outputs/paired_experiments')
    parser.add_argument('--lambda', dest='boundary_lambda', type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.initial_checkpoint, args.output, args.boundary_lambda), indent=2))
