"""Separate industrial experiment using unchanged FINet and original structure loss."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.evaluation_metrics import boundary_f1
from src.finet_adapter import select_device, state_dict_sha256
from src.industrial_data import DOMAIN, PROTOCOL, IndustrialDataset, load_industrial_manifest, sampling_weights
from src.postprocessing import restore_probability, save_prediction
from src.training import (TrainingConfig, append_log, atomic_torch_save, hardware_report,
                          make_model_optimizer, restore_checkpoint, run_signature, save_checkpoint,
                          seed_everything, synchronize, train_step, write_json)
from src.training_data import file_hash
from train import structure_loss


@dataclass
class IndustrialConfig(TrainingConfig):
    data_root: str = 'data/MVTecAD_supervised_v1'
    split_manifest: str = 'data/MVTecAD_supervised_v1/split_manifest.json'
    output_dir: str = 'outputs/industrial/baseline_seed123456_v1'
    application_domain: str = DOMAIN
    dataset_protocol: str = PROTOCOL
    sampler: str = 'equal_category_normal_abnormal_replacement_v1'
    augmentation: str = 'horizontal_flip_only_nearest_mask_v1'
    selection: str = 'macro_defective_category_dice_at_0.5'

    def validate(self):
        super().validate()
        if (self.application_domain != DOMAIN or self.dataset_protocol != PROTOCOL
                or self.sampler != 'equal_category_normal_abnormal_replacement_v1'
                or self.augmentation != 'horizontal_flip_only_nearest_mask_v1'
                or self.selection != 'macro_defective_category_dice_at_0.5'):
            raise ValueError('Unknown industrial protocol; do not silently change a registered experiment.')


def make_industrial_loaders(config, manifest, generator):
    train = IndustrialDataset(config.data_root, manifest['records'], manifest['train_ids'], augment=True)
    validation = IndustrialDataset(config.data_root, manifest['records'], manifest['validation_ids'])
    if len(train) < config.batch_size:
        raise ValueError('Industrial training split is smaller than the batch size.')
    sampler = WeightedRandomSampler(sampling_weights(train.records), len(train), replacement=True, generator=generator)
    return (DataLoader(train, batch_size=config.batch_size, sampler=sampler, drop_last=True,
                       num_workers=config.num_workers, generator=generator, persistent_workers=False,
                       pin_memory=config.device.startswith('cuda')),
            DataLoader(validation, batch_size=1, num_workers=config.num_workers, shuffle=False))


def segmentation_metrics(probability, target, threshold=.5, measure_boundary=True):
    prediction = probability >= threshold
    target = np.asarray(target, dtype=bool)
    if probability.shape != target.shape or not np.isfinite(probability).all():
        raise ValueError('Industrial prediction/target must have matching finite dimensions.')
    intersection = int((prediction & target).sum())
    total = int(prediction.sum() + target.sum())
    union = int((prediction | target).sum())
    return {'dice': 2 * intersection / total if total else 1.,
            'iou': intersection / union if union else 1.,
            'mae': float(np.abs(probability.astype(np.float64) - target).mean()),
            'boundary_f1': boundary_f1(prediction, target, 2) if measure_boundary else None,
            'predicted_pixels': int(prediction.sum()), 'target_pixels': int(target.sum())}


@torch.inference_mode()
def validate_industrial(model, loader, device, output=None, max_samples=None):
    started = perf_counter()
    model.eval()
    rows, losses = [], []
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        rgb, mask, high, low = [value.to(device) for value in batch]
        outputs = model(rgb, high, low)
        losses.append(sum(structure_loss(value, mask) for value in outputs).item())
        record = loader.dataset.records[index]
        with Image.open(loader.dataset.imgs[index]) as source:
            image = source.convert('RGB')
        with Image.open(loader.dataset.gts[index]) as source:
            target = np.asarray(source.convert('L')) > 0
        probability = restore_probability(outputs[0], image.size)
        row = {'id': record['id'], 'category': record['category'], 'is_defective': record['is_defective'],
               **segmentation_metrics(probability, target, measure_boundary=False)}
        rows.append(row)
        if output is not None and index < 3:
            save_prediction(image, probability, Path(output) / record['id'],
                            {'purpose': 'Industrial validation diagnostic, not held-out test results', **row})
    if not rows:
        raise ValueError('Industrial validation is empty.')
    metrics = aggregate_metrics(rows)
    synchronize(device)
    return {'structure_loss': float(np.mean(losses)), 'seconds': perf_counter() - started, **metrics}


def aggregate_metrics(rows):
    names = ('dice', 'iou', 'mae', 'boundary_f1')
    def means(group):
        return {key: float(np.mean([r[key] for r in group if r[key] is not None]))
                if any(r[key] is not None for r in group) else None for key in names}
    categories = {}
    for category in sorted({row['category'] for row in rows}):
        group = [r for r in rows if r['category'] == category]
        abnormal = [r for r in group if r['is_defective']]
        normal = [r for r in group if not r['is_defective']]
        categories[category] = {'samples': len(group),
                                'all_images': means(group),
                                'defective_images': means(abnormal) if abnormal else None,
                                'normal_images_with_any_foreground': sum(r['predicted_pixels'] > 0 for r in normal),
                                'normal_images': len(normal)}
    abnormal_scores = [value['defective_images']['dice'] for value in categories.values() if value['defective_images']]
    return {'samples': len(rows), 'per_category': categories,
            'mean_per_image': means(rows),
            'macro_defective_category_dice': float(np.mean(abnormal_scores)) if abnormal_scores else None,
            'empty_mask_convention': 'Dice/IoU/BF1: both empty=1, only one empty=0. MAE uses raw probabilities.',
            'product_classification': 'Any-foreground counts are diagnostics, not a validated defective-product decision.'}


def industrial_save(path, model, optimizer, scheduler, generator, config, manifest,
                    completed=0, step=0, best_mae=None, best_score=None, status='baseline_training'):
    save_checkpoint(path, model, optimizer, scheduler, generator, config, manifest,
                    completed, step, best_mae, status)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    payload.update(application_domain=DOMAIN, dataset_name='MVTec AD (custom supervised split)',
                   dataset_protocol=PROTOCOL, categories=manifest['categories'],
                   official_benchmark_eligible=False, best_selection_score=best_score,
                   industrial_sources={name: file_hash(Path(__file__).resolve().parents[1] / name)
                                       for name in ('src/industrial_data.py', 'src/industrial_training.py')})
    atomic_torch_save(payload, Path(path))


def run_industrial_smoke(config, steps=2):
    config.validate()
    if steps < 2:
        raise ValueError('At least two measured smoke steps are required.')
    seed_everything(config)
    device = select_device(config.device)
    manifest = load_industrial_manifest(config.split_manifest, config.data_root)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Industrial smoke evidence already exists; choose a new directory.')
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    train, validation = make_industrial_loaders(config, manifest, generator)
    model, optimizer, scheduler = make_model_optimizer(config, device)
    initialization = output / 'initialization.pth'
    industrial_save(initialization, model, optimizer, scheduler, generator, config, manifest, status='initialization')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # Explicit real normal + defective examples check empty masks and small defects.
    lookup = {r['id']: r for r in manifest['records']}
    chosen = [next(name for name in manifest['train_ids'] if lookup[name]['is_defective']),
              next(name for name in manifest['train_ids'] if not lookup[name]['is_defective'])]
    dataset = IndustrialDataset(config.data_root, manifest['records'], chosen)
    pair = next(iter(DataLoader(dataset, batch_size=2)))
    if not (pair[1][0].any() and not pair[1][1].any()):
        raise ValueError('Real defect/normal smoke masks did not retain correct labels at 384.')
    diagnostic = train_step(model, optimizer, pair, device, verify_updates=True,
                            boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    # Restore the exact initialization before timing normal sampled training steps.
    model.load_state_dict(torch.load(initialization, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=initialization)
    iterator = iter(train)
    rows = []
    for index in range(steps):
        synchronize(device)
        started = perf_counter()
        batch = next(iterator)
        row = train_step(model, optimizer, batch, device, verify_updates=True,
                         boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
        synchronize(device)
        row.update(step=index+1, seconds=perf_counter()-started)
        rows.append(row)
        print(json.dumps({'event': 'industrial_smoke_step', **row}), flush=True)
    checkpoint = output / 'smoke_checkpoint.pth'
    industrial_save(checkpoint, model, optimizer, scheduler, generator, config, manifest,
                    step=steps, status='smoke_test')
    reloaded, restored_optimizer, restored_scheduler = make_model_optimizer(config, device, load_path=checkpoint)
    restore_checkpoint(checkpoint, reloaded, restored_optimizer, restored_scheduler, generator,
                       config, manifest, allow_smoke=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored_optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        expected = model(batch[0].to(device), batch[2].to(device), batch[3].to(device))
        actual = reloaded(batch[0].to(device), batch[2].to(device), batch[3].to(device))
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    continuation = train_step(reloaded, restored_optimizer, batch, device, verify_updates=True,
                              boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    validation_result = validate_industrial(reloaded, validation, device, max_samples=2)
    report = {'status': 'passed', 'scope': 'engineering smoke only, no trained-quality claim',
              'hardware': hardware_report(device), 'config': asdict(config),
              'run_signature': run_signature(config, manifest), 'split_fingerprint': manifest['fingerprint'],
              'initialization_sha256': file_hash(initialization), 'initial_model_sha256': model._finet_initial_model_sha256,
              'real_normal_and_defect_ids': chosen, 'normal_defect_diagnostic': diagnostic,
              'steps': rows, 'checkpoint_sha256': file_hash(checkpoint),
              'all_model_tensors_reloaded_exactly': True, 'optimizer_scheduler_reloaded_exactly': True,
              'all_four_reloaded_outputs_equal': True, 'resumed_optimizer_step': continuation,
              'validation_diagnostic': validation_result,
              'measured_steady_step_seconds': float(np.mean([r['seconds'] for r in rows[1:]])),
              'training_steps_per_epoch': len(train), 'full_training_started': False}
    write_json(output / 'smoke_report.json', report)
    return report


def run_industrial_training(config, smoke_report, resume=None, max_hours=7.5, max_new_epochs=None,
                            final_only=False, session_checkpoint_deadline=None):
    config.validate()
    device = select_device(config.device)
    if device.type != 'cuda':
        raise ValueError('Long industrial training requires confirmed CUDA; CPU supports smoke only.')
    if not final_only and not 0 < max_hours <= 8:
        raise ValueError('Use a bounded session of at most 8 hours.')
    if final_only and max_new_epochs is not None:
        raise ValueError('Final-only training must target the full configured epoch budget.')
    if session_checkpoint_deadline is not None:
        if not final_only or not math.isfinite(session_checkpoint_deadline) or session_checkpoint_deadline <= 0:
            raise ValueError('A session-end checkpoint deadline requires final-only mode and a positive monotonic time.')
    manifest = load_industrial_manifest(config.split_manifest, config.data_root)
    evidence = json.loads(Path(smoke_report).read_text(encoding='utf-8'))
    if (evidence.get('status') != 'passed' or evidence.get('run_signature') != run_signature(config, manifest)
            or not evidence.get('hardware', {}).get('device', '').startswith('cuda')
            or evidence['hardware'].get('gpu') != torch.cuda.get_device_name(device)):
        raise ValueError('A passed GPU smoke report for this exact industrial configuration is required.')
    if not config.initial_checkpoint or file_hash(Path(config.initial_checkpoint)) != evidence['initialization_sha256']:
        raise ValueError('Use the unchanged initialization saved before the matching industrial GPU smoke.')
    output = Path(config.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing industrial run must be resumed, never overwritten.')
    seed_everything(config)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=resume)
    completed, global_step, best_score, best_mae = 0, 0, None, None
    if resume:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, generator, config, manifest)
        if saved.get('application_domain') != DOMAIN:
            raise ValueError('Refusing to resume camouflage weights as industrial training.')
        completed, global_step = saved['completed_epochs'], saved['global_step']
        best_score, best_mae = saved['best_selection_score'], saved['best_validation_mae']
        marker = {'event': 'resume', 'completed_epochs': completed, 'global_step': global_step,
                  'next_epoch': completed+1, 'learning_rate': scheduler.get_lr()}
        append_log(output / 'epochs.jsonl', marker)
        print(json.dumps(marker), flush=True)
    train, validation = make_industrial_loaders(config, manifest, generator)
    write_json(output / 'config.json', asdict(config))
    write_json(output / 'split_manifest.json', manifest)
    write_json(output / 'hardware.json', hardware_report(device))
    # A user-requested session-end save is distinct from per-epoch recovery.
    # Its absolute monotonic deadline includes notebook setup/probe time.
    deadline = session_checkpoint_deadline if final_only else monotonic() + max_hours*3600 - 120
    resumed_epochs = completed
    epoch_seconds = evidence['measured_steady_step_seconds'] * len(train) + 180
    stop = min(config.epochs, completed + max_new_epochs) if max_new_epochs else config.epochs
    for epoch in range(completed, stop):
        if deadline is not None and deadline - monotonic() < epoch_seconds*1.2 + 120:
            break
        started = perf_counter()
        losses = []
        lr = scheduler.get_lr()
        for batch in train:
            row = train_step(model, optimizer, batch, device, boundary_lambda=config.boundary_lambda,
                             boundary_kernel=config.boundary_kernel)
            global_step += 1
            losses.append(row['loss'])
            row.update(epoch=epoch+1, global_step=global_step, learning_rate=lr)
            append_log(output / 'steps.jsonl', row)
            if global_step % 50 == 0:
                print(json.dumps({'event': 'industrial_training_progress', **row}), flush=True)
        samples = None if final_only else output / 'validation_samples' / f'epoch_{epoch+1:03d}'
        metrics = validate_industrial(model, validation, device, samples)
        scheduler.step()
        score = metrics['macro_defective_category_dice']
        if score is None:
            raise ValueError('Validation must contain real defective images.')
        improved = best_score is None or score > best_score
        best_score = score if improved else best_score
        best_mae = min(best_mae, metrics['mean_per_image']['mae']) if best_mae is not None else metrics['mean_per_image']['mae']
        if not final_only or epoch + 1 == config.epochs:
            industrial_save(output / 'last.pth', model, optimizer, scheduler, generator, config, manifest,
                            epoch+1, global_step, best_mae, best_score)
        if improved and not final_only:
            industrial_save(output / 'best.pth', model, optimizer, scheduler, generator, config, manifest,
                            epoch+1, global_step, best_mae, best_score)
        epoch_seconds = perf_counter() - started
        completed = epoch+1
        summary = {'epoch': completed, 'global_step': global_step, 'training_loss': float(np.mean(losses)),
                   'learning_rate': lr, 'validation': metrics, 'best_selection_score': best_score,
                   'seconds': epoch_seconds}
        append_log(output / 'epochs.jsonl', summary)
        print(json.dumps(summary), flush=True)
    session_checkpoint_saved = False
    if final_only and deadline is not None and resumed_epochs < completed < config.epochs:
        industrial_save(output / 'last.pth', model, optimizer, scheduler, generator, config, manifest,
                        completed, global_step, best_mae, best_score)
        session_checkpoint_saved = True
        print(json.dumps({'event': 'industrial_session_end_checkpoint', 'completed_epochs': completed,
                          'global_step': global_step, 'next_epoch': completed+1,
                          'remaining_until_deadline_seconds': deadline-monotonic()}), flush=True)
    result = {'completed_epochs': completed, 'planned_epochs': config.epochs,
              'training_complete': completed == config.epochs, 'global_step': global_step,
              'checkpoint': str(output / 'last.pth'), 'remaining_epochs': config.epochs-completed,
              'session_checkpoint_saved': session_checkpoint_saved}
    write_json(output / 'session_result.json', result)
    return result
