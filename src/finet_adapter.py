"""Strict, device-aware inference using the original FINet architecture."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
from PIL import Image
import torch

from Model.FINet import FINet
from src.preprocessing import FINetPreprocessor, read_image
from src.postprocessing import restore_probability


class CheckpointError(ValueError):
    """The supplied file is not a complete compatible FINet state dict."""


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Device must be auto, cpu, cuda or cuda:N.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This pipeline supports CPU or CUDA devices.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Use --device cpu or auto.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist.")
    return device


def extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if (not isinstance(payload, Mapping) or not payload
            or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                       for key, value in payload.items())):
        raise CheckpointError("Expected a full FINet tensor state_dict, not a model object or ImageNet-only weights.")
    state = dict(payload)
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise CheckpointError("Checkpoint contains NaN or infinite values.")
    return state


def state_dict_sha256(state) -> str:
    """Hash exact named tensor contents independently of torch.save metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{name}|{value.dtype}|{tuple(value.shape)}\n'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Prediction:
    original: Image.Image
    probability: np.ndarray
    metadata: dict


class FINetPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"FINet checkpoint not found: {self.checkpoint}. Obtain the official full "
                "EfficientNet-B0 FINet weights or a project-trained full checkpoint; see docs/CHECKPOINTS.md."
            )
        self.device = select_device(device)
        # Hash and deserialize the same open file; never enable arbitrary pickle loading.
        try:
            with self.checkpoint.open("rb") as handle:
                self.checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
                handle.seek(0)
                payload = torch.load(handle, map_location="cpu", weights_only=True)
                status = payload.get("training_status") if isinstance(payload, Mapping) else None
                if status is not None and not isinstance(status, str):
                    raise CheckpointError("Checkpoint training_status metadata must be a string.")
                if status in {"initialization", "smoke_test"}:
                    raise CheckpointError("This is an initialization/engineering smoke checkpoint, not a trained segmentation model. "
                                          "Finish baseline training before using it for detection or the application.")
                state = extract_state_dict(payload)
        except CheckpointError:
            raise
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            raise CheckpointError(
                f"Cannot safely read checkpoint '{self.checkpoint.name}'. Use the original tensor "
                "state_dict file, not an HTML download page or untrusted serialized model."
            ) from exc
        # No ImageNet download. The strict full checkpoint must replace every model tensor.
        self.model = FINet(backbone="efficientb0", channels=(8, 24, 32, 64), pretrained_backbone=False)
        expected_keys = self.model.state_dict().keys()
        if state.keys() != expected_keys:
            missing = sorted(set(expected_keys) - state.keys())
            unexpected = sorted(state.keys() - set(expected_keys))
            # PyTorch can silently supply old BatchNorm tracking buffers even in strict
            # mode. Explicit key equality makes our full-checkpoint requirement stronger.
            raise CheckpointError(
                "Checkpoint does not contain the complete FINet state. No partial weights were accepted. "
                f"Missing {len(missing)} tensors (first 5: {missing[:5]}); "
                f"unexpected {len(unexpected)} tensors (first 5: {unexpected[:5]})."
            )
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise CheckpointError(
                "Checkpoint does not match full FINet EfficientNet-B0 with channels (8,24,32,64). "
                f"No partial weights were accepted. Details: {exc}"
            ) from exc
        self.model.to(self.device).eval()
        self.preprocessor = FINetPreprocessor(size=384)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.state_tensor_count = len(state)
        self.training_status = payload.get("training_status", "unknown") if isinstance(payload, Mapping) else "unknown"
        settings = payload.get('config', {}) if isinstance(payload, Mapping) else {}
        self.application_domain = payload.get('application_domain', settings.get('application_domain', 'Camouflaged Objects')) if isinstance(payload, Mapping) else 'Camouflaged Objects'
        self.dataset_name = payload.get('dataset_name', 'COD10K-Train + CAMO-Train') if isinstance(payload, Mapping) else 'Camouflaged object segmentation'
        self.dataset_protocol = payload.get('dataset_protocol') if isinstance(payload, Mapping) else None
        self.comparison_provenance = {key: payload.get(key) for key in ('initial_model_sha256', 'split_fingerprint')} if isinstance(payload, Mapping) else {}
        self.training_progress = None
        if self.training_status in {"baseline_training", "boundary_training"}:
            settings = payload.get("config", {})
            completed = payload.get("completed_epochs")
            planned = settings.get("epochs") if isinstance(settings, Mapping) else None
            if type(completed) is int and type(planned) is int and 0 <= completed <= planned and planned > 0:
                self.training_progress = {"completed_epochs": completed, "planned_epochs": planned,
                                          "training_complete": completed == planned}
        self.model_name = ("Boundary-Aware FINet (EfficientNet-B0; project-trained)"
                           if self.training_status == "boundary_training" else
                           "FINet (EfficientNet-B0; project-trained baseline)"
                           if self.training_status == "baseline_training" else "Original FINet (EfficientNet-B0)")

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, image_path: str | Path) -> Prediction:
        started = perf_counter()
        image = read_image(image_path)
        prepared = self.preprocessor(image)
        rgb, high, low = [value.to(self.device) for value in (prepared.rgb, prepared.high, prepared.low)]
        self._synchronize()
        forward_started = perf_counter()
        outputs = self.model(rgb, high, low)
        self._synchronize()
        inference_ms = (perf_counter() - forward_started) * 1000
        if (not isinstance(outputs, (tuple, list)) or len(outputs) != 4
                or any(output.shape != (1, 1, 384, 384) for output in outputs)):
            raise ValueError("Unexpected FINet output shapes; expected four [1,1,384,384] logits.")
        probability = restore_probability(outputs[0], image.size)
        processing_ms = (perf_counter() - started) * 1000
        metadata = {
            "model": self.model_name, "checkpoint_training_status": self.training_status,
            "application_domain": self.application_domain, "training_dataset": self.dataset_name,
            "dataset_protocol": self.dataset_protocol,
            "training_progress": self.training_progress,
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
            "strict_checkpoint_loading": True, "checkpoint_state_tensors": self.state_tensor_count,
            "parameter_count": self.parameter_count, "device": str(self.device),
            "torch_version": torch.__version__, "input_size_wh": [384, 384],
            "image": str(Path(image_path).expanduser().resolve()),
            "inference_ms": inference_ms, "processing_ms": processing_ms,
            "timing_definition": "One synchronized forward pass; no internal warm-up. Processing includes decode, "
                                 "preprocessing, transfer and output restoration, excluding model loading and saving.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ground_truth_metrics": None,
        }
        return Prediction(original=image, probability=probability, metadata=metadata)
