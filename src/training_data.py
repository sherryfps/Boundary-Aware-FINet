"""Audit real paired masks and reuse the original FINet training transforms."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from utils.dataloader_freq import TrainDataset


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manifest_hash(manifest: dict) -> str:
    content = {key: value for key, value in manifest.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def indexed_files(folder: Path, suffixes: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    result = {}
    # Path comparison is case-insensitive on Windows and case-sensitive on Linux.
    # Sort literal filenames explicitly so manifests are portable across both.
    for path in sorted(folder.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate filename stem: {path.stem}")
        result[path.stem] = path
    return result


def audit_pairs(root: str | Path, allow_soft_masks: bool = False, group_duplicates: bool = False,
                excluded_ids=()) -> list[dict]:
    root = Path(root).resolve()
    images = indexed_files(root / "Imgs", {".jpg", ".png"})
    masks = indexed_files(root / "GT", {".png"})
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Require matching nonempty Imgs/GT filename stems. "
                         f"Missing masks: {sorted(images.keys() - masks.keys())[:5]}; "
                         f"missing images: {sorted(masks.keys() - images.keys())[:5]}.")
    records = []
    excluded_ids = set(excluded_ids)
    if not excluded_ids.issubset(images):
        raise ValueError('Excluded sample IDs do not exist in the dataset.')
    seen_images = set()
    for name, path in images.items():
        if name in excluded_ids:
            continue
        with Image.open(path) as image, Image.open(masks[name]) as mask:
            image.load()
            mask.load()
            if image.size != mask.size:
                raise ValueError(f"Image/mask dimension mismatch: {name}")
            if min(image.size) <= 30:
                raise ValueError(f"Image too small for original random crop (border 30): {name}")
            if mask.format != "PNG":
                raise ValueError(f"Ground truth must be original PNG, not a renamed lossy preview: {name}")
            values = np.asarray(mask.convert("L"))
            soft_pixels = int(((values > 0) & (values < 255)).sum())
            if soft_pixels and not allow_soft_masks:
                raise ValueError(f"Expected binary 0/255 object ground truth, not scribbles/instances: {name}")
            if allow_soft_masks:
                rgb_mask = np.asarray(mask.convert("RGB"))
                if not (np.array_equal(rgb_mask[..., 0], rgb_mask[..., 1])
                        and np.array_equal(rgb_mask[..., 1], rgb_mask[..., 2])):
                    raise ValueError(f"Expected grayscale object mask, not color/instance labels: {name}")
            size = list(image.size)
        image_hash = file_hash(path)
        if image_hash in seen_images and not group_duplicates:
            raise ValueError(f"Duplicate image content risks split leakage: {name}")
        seen_images.add(image_hash)
        lower = name.lower()
        source = "COD10K" if lower.startswith("cod10k-") else (
            "CAMO" if lower.startswith(("camourflage", "camouflage", "camo_")) else "unknown")
        records.append({"id": name, "image": path.relative_to(root).as_posix(),
                        "mask": masks[name].relative_to(root).as_posix(), "source": source,
                        "size_wh": size, "image_sha256": image_hash, "mask_sha256": file_hash(masks[name])})
        if allow_soft_masks:
            records[-1]["mask_soft_pixels"] = soft_pixels
    return records


def create_manifest(root: str | Path, validation_fraction: float = 0.1,
                    seed: int = 123456, allow_small: bool = False, allow_soft_masks: bool = False,
                    group_duplicates: bool = False) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    records = audit_pairs(root, allow_soft_masks, group_duplicates)
    counts = dict(Counter(record["source"] for record in records))
    if not allow_small and counts != {"COD10K": 3040, "CAMO": 1000}:
        raise ValueError(f"Full baseline requires 3040 COD10K-Train + 1000 CAMO-Train pairs; actual: {counts}. "
                         "Do not use test images. --allow-small is only for a documented smoke test.")
    rng = random.Random(seed)
    train, validation = [], []
    hash_sources = {}
    for record in records:
        previous = hash_sources.setdefault(record["image_sha256"], record["source"])
        if previous != record["source"]:
            raise ValueError("Cross-dataset duplicate needs an explicit unified split policy.")
    for source in sorted(counts):
        names = sorted(record["id"] for record in records if record["source"] == source)
        if len(names) < 2:
            raise ValueError(f"Need at least two images per source for a separate validation split: {source}")
        count = max(1, min(len(names) - 1, round(len(names) * validation_fraction)))
        if group_duplicates:
            groups = {}
            lookup = {record["id"]: record for record in records}
            for name in names:
                groups.setdefault(lookup[name]["image_sha256"], []).append(name)
            grouped = list(groups.values())
            rng.shuffle(grouped)
            selected = 0
            for group in grouped:
                if selected + len(group) <= count:
                    validation.extend(group)
                    selected += len(group)
                else:
                    train.extend(group)
            if selected == 0 or selected == len(names):
                raise ValueError("Insufficient independent images for grouped validation.")
        else:
            rng.shuffle(names)
            validation.extend(names[:count])
            train.extend(names[count:])
    if len(train) < 2:
        raise ValueError("At least two training images are required by FINet's pooled BatchNorm.")
    manifest = {"schema": 1, "seed": seed, "validation_fraction": validation_fraction,
                "smoke_only": allow_small, "source_counts": counts, "records": records,
                "train_ids": sorted(train), "validation_ids": sorted(validation)}
    if allow_soft_masks:
        manifest["allow_soft_masks"] = True
        manifest["mask_policy"] = "Preserve original grayscale 0..255 targets; no binarization for structure loss."
    if group_duplicates:
        manifest["group_duplicates"] = True
        manifest["duplicate_policy"] = "Keep byte-identical images together in a single split."
    manifest["fingerprint"] = manifest_hash(manifest)
    return manifest


def load_manifest(path: str | Path, root: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != manifest_hash(manifest):
        raise ValueError("Split manifest fingerprint is invalid.")
    actual = audit_pairs(root, manifest.get("allow_soft_masks", False), manifest.get("group_duplicates", False))
    if actual != manifest["records"]:
        raise ValueError("Dataset contents changed since split creation; do not silently resume on different data.")
    train, validation = set(manifest["train_ids"]), set(manifest["validation_ids"])
    names = {record["id"] for record in actual}
    if (train & validation or train | validation != names
            or len(train) != len(manifest["train_ids"]) or len(validation) != len(manifest["validation_ids"])):
        raise ValueError("Invalid, incomplete or overlapping training/validation split.")
    train_hashes = {record["image_sha256"] for record in actual if record["id"] in train}
    val_hashes = {record["image_sha256"] for record in actual if record["id"] in validation}
    if train_hashes & val_hashes:
        raise ValueError("Duplicate image content overlaps training and validation.")
    return manifest


class FINetTrainingDataset(TrainDataset):
    """Reuse upstream augmentations, resizing, normalization, DCT and mask transform."""

    def __init__(self, root: str | Path, records: list[dict], ids: list[str], augment: bool):
        root = Path(root).resolve()
        super().__init__(str(root / "Imgs"), str(root / "GT"), trainsize=384,
                         rVFlip=augment, rCrop=augment, rRotate=augment,
                         colorEnhance=augment, rPeper=augment)
        lookup = {record["id"]: record for record in records}
        self.records = [lookup[name] for name in ids]
        self.imgs = [str(root / record["image"]) for record in self.records]
        self.gts = [str(root / record["mask"]) for record in self.records]
        self.size = len(ids)

    def filter_files(self):
        # Upstream sorted paths can silently associate different filename stems.
        images = {Path(path).stem: path for path in self.imgs}
        masks = {Path(path).stem: path for path in self.gts}
        if images.keys() != masks.keys():
            raise ValueError("Training images and masks must have matching filename stems.")
        self.imgs = [images[name] for name in sorted(images)]
        self.gts = [masks[name] for name in sorted(images)]
