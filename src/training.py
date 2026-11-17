"""Resumable original FINet training with optional training-only boundary loss."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from Model.FINet import FINet
from src.boundary_loss import finet_objective
from src.finet_adapter import extract_state_dict, select_device, state_dict_sha256
from src.postprocessing import restore_probability, save_prediction
from src.training_data import FINetTrainingDataset, file_hash, load_manifest
from train import structure_loss
from utils.LRScheduler import CosineDecay


@dataclass
class TrainingConfig:
    seed: int = 123456
    image_size: int = 384
    batch_size: int = 4
    epochs: int = 200
    learning_rate: float = 2.6e-4
    min_lr: float = 1e-7
    weight_decay: float = 4e-8
    num_workers: int = 2
    cpu_threads: int = 2
    save_every_epochs: int = 1
    cudnn_enabled: bool = True
    data_root: str = "data/TrainDataset"
    split_manifest: str = "outputs/splits/cod_camo_seed123456.json"
    output_dir: str = "outputs/baseline"
    device: str = "cuda"
    initial_checkpoint: str | None = None
    boundary_lambda: float = 0.0
    boundary_kernel: int = 3

    def validate(self):
        if not math.isfinite(self.boundary_lambda) or self.boundary_lambda < 0:
            raise ValueError("Boundary lambda must be finite and nonnegative.")
        if not isinstance(self.boundary_kernel, int) or self.boundary_kernel < 3 or self.boundary_kernel % 2 != 1:
            raise ValueError("Boundary kernel must be an odd integer >=3.")
        if self.image_size != 384:
            raise ValueError("This baseline preserves the official 384×384 preprocessing.")
        if self.batch_size < 2:
            raise ValueError("Training batch size must be >=2 for pooled BatchNorm branches.")
        if self.epochs < 1 or self.num_workers < 0 or self.cpu_threads < 1 or self.save_every_epochs < 1:
            raise ValueError("Invalid epochs, workers, thread count or checkpoint interval.")
        if not all(math.isfinite(value) for value in (self.learning_rate, self.min_lr, self.weight_decay)):
            raise ValueError("Optimizer settings must be finite.")
        if not 0 <= self.min_lr <= self.learning_rate or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")


def seed_everything(config: TrainingConfig):
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hardware_report(device: torch.device) -> dict:
    report = {"device": str(device), "cuda_available": torch.cuda.is_available(),
              "torch_version": torch.__version__, "environment": environment_report()}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        report.update(gpu=torch.cuda.get_device_name(device), gpu_free_bytes=free,
                      gpu_total_bytes=total, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return report


def environment_report() -> dict:
    root = Path(__file__).resolve().parents[1]
    sources = ("Model/FINet.py", "Model/EfficientNet.py", "Model/Modules.py", "train.py", "utils/dct.py",
               "utils/dataloader_freq.py", "utils/freq_mean_std.pkl", "utils/LRScheduler.py",
               "src/training.py", "src/training_data.py", "src/boundary_loss.py")
    return {"python": sys.version, "packages": {name: version(name) for name in
            ("torch", "torchvision", "numpy", "Pillow", "timm", "torch-dct")},
            "cuda_runtime": torch.version.cuda,
            "source_sha256": {name: file_hash(root / name) for name in sources}}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def capture_rng(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state: dict, generator: torch.Generator):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def run_signature(config: TrainingConfig, manifest: dict) -> str:
    settings = asdict(config)
    # Keep already verified baseline smoke signatures compatible with lambda=0.
    if not config.boundary_lambda:
        settings.pop("boundary_lambda")
        settings.pop("boundary_kernel")
    for name in ("data_root", "split_manifest", "output_dir", "device", "initial_checkpoint", "cpu_threads"):
        settings.pop(name)
    settings["split_fingerprint"] = manifest["fingerprint"]
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path: Path, model, optimizer, scheduler, generator, config, manifest,
                    completed_epochs=0, global_step=0, best_mae=None, status="baseline_training"):
    if status == "baseline_training" and config.boundary_lambda:
        status = "boundary_training"
    payload = {"format_version": 1, "training_status": status,
               "model_name": "Boundary-Aware FINet" if config.boundary_lambda else "Original FINet",
               "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
               "rng_state": capture_rng(generator), "completed_epochs": completed_epochs,
               "global_step": global_step, "best_validation_mae": best_mae,
               "config": asdict(config), "run_signature": run_signature(config, manifest),
               "split_fingerprint": manifest["fingerprint"], "smoke_data": manifest["smoke_only"],
               "initial_model_sha256": getattr(model, '_finet_initial_model_sha256', None),
               "loss": {"structure": "exact train.structure_loss summed over all four outputs",
                        "boundary": "sum of four morphological boundary BCE+Dice terms",
                        "lambda": config.boundary_lambda, "kernel": config.boundary_kernel},
               "environment": environment_report(),
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_torch_save(payload, path)


def restore_checkpoint(path, model, optimizer, scheduler, generator, config, manifest,
                       allow_smoke=False) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["run_signature"] != run_signature(config, manifest):
        raise ValueError("Resume configuration or dataset split differs from the saved run.")
    expected_status = "boundary_training" if config.boundary_lambda else "baseline_training"
    if payload["training_status"] != expected_status and not allow_smoke:
        raise ValueError("Initialization/smoke checkpoints cannot resume a full baseline run; use initialization as initial_checkpoint.")
    state = extract_state_dict(payload)
    if state.keys() != model.state_dict().keys():
        raise ValueError("Checkpoint does not contain every FINet tensor.")
    model.load_state_dict(state, strict=True)
    model._finet_initial_model_sha256 = payload.get('initial_model_sha256')
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng(payload["rng_state"], generator)
    return payload


def make_model_optimizer(config: TrainingConfig, device, load_path=None):
    path = load_path or config.initial_checkpoint
    model = FINet(backbone="efficientb0", channels=(8, 24, 32, 64), pretrained_backbone=not bool(path))
    if path:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = extract_state_dict(payload)
        if state.keys() != model.state_dict().keys():
            raise ValueError("Initial checkpoint must contain all FINet tensors.")
        model.load_state_dict(state, strict=True)
    model._finet_initial_model_sha256 = state_dict_sha256(model.state_dict())
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineDecay(optimizer, config.learning_rate, config.min_lr, config.epochs)
    return model, optimizer, scheduler


def make_loaders(config: TrainingConfig, manifest: dict, generator):
    train = FINetTrainingDataset(config.data_root, manifest["records"], manifest["train_ids"], augment=True)
    validation = FINetTrainingDataset(config.data_root, manifest["records"], manifest["validation_ids"], augment=False)
    if len(train) < config.batch_size:
        raise ValueError("Training split is smaller than the batch size; reduce batch size to >=2.")
    training_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, drop_last=True,
                                 num_workers=config.num_workers, pin_memory=config.device.startswith("cuda"),
                                 generator=generator, persistent_workers=False)
    validation_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=config.num_workers)
    return training_loader, validation_loader


def train_step(model, optimizer, batch, device, verify_updates=False, boundary_lambda=0.0, boundary_kernel=3) -> dict:
    if len(batch) != 4:
        raise ValueError("Expected RGB, segmentation mask, high frequency and low frequency tensors.")
    rgb, mask, high, low = [value.to(device) for value in batch]
    count = rgb.shape[0]
    if (count < 2 or rgb.shape != (count, 3, 384, 384) or mask.shape != (count, 1, 384, 384)
            or high.shape != (count, 96, 48, 48) or low.shape != high.shape):
        raise ValueError("Invalid original FINet training tensor dimensions (batch must be >=2).")
    if not all(torch.isfinite(value).all() for value in (rgb, mask, high, low)) or mask.min() < 0 or mask.max() > 1:
        raise ValueError("Training inputs must be finite and masks within [0,1].")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    probes = {"encoder": next(model.encoder.parameters()), "head": model.out_conv1.weight}
    before = {key: value.detach().clone() for key, value in probes.items()} if verify_updates else {}
    outputs = model(rgb, high, low)
    if len(outputs) != 4 or any(output.shape != mask.shape for output in outputs):
        raise ValueError("FINet must return four same-resolution logits maps.")
    loss, structure, boundary, losses = finet_objective(outputs, mask, boundary_lambda, boundary_kernel)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite original structure loss.")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if any(gradient is None for gradient in gradients):
        raise ValueError("A trainable parameter did not receive gradients.")
    if not torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all():
        raise ValueError("NaN/infinite gradients; optimizer update refused.")
    gradient_norm = torch.stack([gradient.detach().float().norm().square() for gradient in gradients]).sum().sqrt().item()
    optimizer.step()
    result = {"loss": loss.item(), "structure_loss": structure.item(), "boundary_loss": boundary.item(),
              "boundary_lambda": boundary_lambda, "head_losses": [value.item() for value in losses],
              "gradient_norm": gradient_norm, "gradient_tensors": len(gradients), "finite_gradients": True}
    if verify_updates:
        result["updated"] = {key: not torch.equal(before[key], value.detach()) for key, value in probes.items()}
        if not all(result["updated"].values()):
            raise ValueError("Optimizer did not update both encoder and segmentation head probes.")
        result["input_shapes"] = [list(value.shape) for value in (rgb, mask, high, low)]
        result["output_shapes"] = [list(value.shape) for value in outputs]
    return result


@torch.inference_mode()
def validate(model, loader, device, output_dir: Path | None = None, max_samples=None) -> dict:
    model.eval()
    started = perf_counter()
    losses, maes = [], []
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        rgb, mask, high, low = [value.to(device) for value in batch]
        outputs = model(rgb, high, low)
        losses.append(sum(structure_loss(output, mask) for output in outputs).item())
        record = loader.dataset.records[index]
        with Image.open(loader.dataset.imgs[index]) as image:
            original = image.convert("RGB")
        with Image.open(loader.dataset.gts[index]) as mask_image:
            target = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
        probability = restore_probability(outputs[0], original.size)
        maes.append(float(np.abs(probability - target).mean()))
        if output_dir is not None and index < 3:
            save_prediction(original, probability, output_dir / record["id"],
                            {"purpose": "validation diagnostic after baseline epoch; not final test performance",
                             "sample_id": record["id"]})
    if not losses:
        raise ValueError("Validation split is empty.")
    synchronize(device)
    return {"structure_loss": float(np.mean(losses)), "mae": float(np.mean(maes)),
            "samples": len(losses), "seconds": perf_counter() - started,
            "mae_definition": "Per-image native-resolution mean absolute probability error, averaged over validation images; no min-max normalization."}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_log(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def run_smoke(config: TrainingConfig, steps: int = 2) -> dict:
    config.validate()
    if steps < 2:
        raise ValueError("Use at least two measured smoke steps before estimating steady-step time.")
    seed_everything(config)
    device = select_device(config.device)
    manifest = load_manifest(config.split_manifest, config.data_root)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Smoke output directory is not empty; choose a new directory to retain previous evidence.")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader, val_loader = make_loaders(config, manifest, generator)
    model, optimizer, scheduler = make_model_optimizer(config, device)
    save_checkpoint(output / "initialization.pth", model, optimizer, scheduler, generator, config, manifest,
                    status="initialization")
    write_json(output / "config.json", asdict(config))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(train_loader)
    rows, epochs = [], 0
    for step in range(steps):
        synchronize(device)
        started = perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            scheduler.step()
            epochs += 1
            iterator = iter(train_loader)
            batch = next(iterator)
        row = train_step(model, optimizer, batch, device, verify_updates=True,
                         boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
        synchronize(device)
        row.update(step=step + 1, seconds=perf_counter() - started, learning_rate=scheduler.get_lr())
        rows.append(row)
        append_log(output / "steps.jsonl", row)
        print(json.dumps({key: row[key] for key in ("step", "loss", "head_losses", "seconds", "updated")}), flush=True)
    checkpoint = output / "smoke_checkpoint.pth"
    save_checkpoint(checkpoint, model, optimizer, scheduler, generator, config, manifest,
                    completed_epochs=epochs, global_step=steps, status="smoke_test")
    model.eval()
    with torch.inference_mode():
        expected = model(batch[0].to(device), batch[2].to(device), batch[3].to(device))[0].cpu()
    reloaded, reloaded_optimizer, reloaded_scheduler = make_model_optimizer(config, device, load_path=checkpoint)
    payload = restore_checkpoint(checkpoint, reloaded, reloaded_optimizer, reloaded_scheduler,
                                 generator, config, manifest, allow_smoke=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)
    if reloaded_scheduler.state_dict() != scheduler.state_dict():
        raise ValueError("Reloaded scheduler state differs.")
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(reloaded_optimizer.state_dict()["state"][key][name], value, rtol=0, atol=0)
    reloaded.eval()
    with torch.inference_mode():
        actual = reloaded(batch[0].to(device), batch[2].to(device), batch[3].to(device))[0].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    continuation = train_step(reloaded, reloaded_optimizer, batch, device, verify_updates=True,
                              boundary_lambda=config.boundary_lambda, boundary_kernel=config.boundary_kernel)
    validation = validate(reloaded, val_loader, device, max_samples=2)
    report = {"status": "passed", "scope": "engineering smoke test only; no trained-quality segmentation claim",
              "hardware": hardware_report(device), "config": asdict(config),
              "run_signature": run_signature(config, manifest),
              "initialization_sha256": file_hash(output / "initialization.pth"),
              "initial_model_sha256": model._finet_initial_model_sha256,
              "split_fingerprint": manifest["fingerprint"], "dataset_counts": manifest["source_counts"],
              "train_samples": len(manifest["train_ids"]), "validation_samples": len(manifest["validation_ids"]),
              "steps": rows, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
              "reloaded_global_step": payload["global_step"], "all_model_tensors_reloaded_exactly": True,
              "optimizer_scheduler_reloaded_exactly": True, "reloaded_output_equal": True,
              "resumed_optimizer_step": continuation, "validation_diagnostic": validation,
              "measured_steady_step_seconds": float(np.mean([row["seconds"] for row in rows[1:]])),
              "estimate_note": "Includes data loading/augmentation and optimizer step, excludes initial model loading, checkpoint I/O and validation. CPU timing cannot estimate a Colab GPU duration.",
              "full_training_started": False}
    write_json(output / "smoke_report.json", report)
    return report


def run_baseline(config: TrainingConfig, resume: str | None = None, verified_smoke_report: str | None = None):
    """Explicit CLI opt-in only. The notebook never calls this automatically."""
    config.validate()
    device = select_device(config.device)
    if device.type != "cuda":
        raise ValueError("Full training requires a confirmed CUDA GPU. CPU is supported for smoke tests only.")
    manifest = load_manifest(config.split_manifest, config.data_root)
    if manifest["smoke_only"] or manifest["source_counts"] != {"COD10K": 3040, "CAMO": 1000}:
        raise ValueError("Full baseline requires the audited official 4040-pair training collection.")
    if not verified_smoke_report:
        raise ValueError("A passed GPU smoke report for this exact configuration/split is required.")
    evidence = json.loads(Path(verified_smoke_report).read_text(encoding="utf-8"))
    if (evidence.get("status") != "passed" or evidence.get("run_signature") != run_signature(config, manifest)
            or not evidence.get("hardware", {}).get("device", "").startswith("cuda")
            or evidence.get("hardware", {}).get("gpu") != torch.cuda.get_device_name(device)):
        raise ValueError("GPU smoke evidence does not match the current configuration/split/GPU.")
    if not config.initial_checkpoint or file_hash(Path(config.initial_checkpoint)) != evidence["initialization_sha256"]:
        raise ValueError("Use the exact initialization.pth saved before the verified GPU smoke run.")
    output = Path(config.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty. Resume the existing run or use a new directory.")
    seed_everything(config)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    model, optimizer, scheduler = make_model_optimizer(config, device, load_path=resume)
    completed, global_step, best_mae = 0, 0, None
    if resume:
        saved = restore_checkpoint(resume, model, optimizer, scheduler, generator, config, manifest)
        completed, global_step, best_mae = saved["completed_epochs"], saved["global_step"], saved["best_validation_mae"]
        marker = {"event": "resume", "checkpoint": str(resume), "completed_epochs": completed,
                  "restored_global_step": global_step,
                  "note": "Any earlier log tail beyond this saved epoch is interrupted work and will be replayed."}
        append_log(output / "steps.jsonl", marker)
        append_log(output / "epochs.jsonl", marker)
    else:
        save_checkpoint(output / "initialization.pth", model, optimizer, scheduler, generator, config, manifest,
                        status="initialization")
    train_loader, val_loader = make_loaders(config, manifest, generator)
    write_json(output / "config.json", asdict(config))
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "hardware.json", hardware_report(device))
    for epoch in range(completed, config.epochs):
        losses = []
        learning_rate = scheduler.get_lr()
        for batch in train_loader:
            started = perf_counter()
            row = train_step(model, optimizer, batch, device, boundary_lambda=config.boundary_lambda,
                             boundary_kernel=config.boundary_kernel)
            global_step += 1
            losses.append(row["loss"])
            row.update(epoch=epoch + 1, global_step=global_step, learning_rate=learning_rate,
                       seconds=perf_counter() - started)
            append_log(output / "steps.jsonl", row)
        metrics = validate(model, val_loader, device, output / "validation_samples" / f"epoch_{epoch + 1:03d}")
        scheduler.step()
        improved = best_mae is None or metrics["mae"] < best_mae
        if improved:
            best_mae = metrics["mae"]
        if (epoch + 1) % config.save_every_epochs == 0 or epoch + 1 == config.epochs:
            save_checkpoint(output / "last.pth", model, optimizer, scheduler, generator, config, manifest,
                            epoch + 1, global_step, best_mae)
        if improved:
            save_checkpoint(output / "best.pth", model, optimizer, scheduler, generator, config, manifest,
                            epoch + 1, global_step, best_mae)
        summary = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "learning_rate": learning_rate,
                   "validation": metrics, "best_validation_mae": best_mae, "selected_checkpoint_updated": improved}
        append_log(output / "epochs.jsonl", summary)
        print(json.dumps(summary), flush=True)
