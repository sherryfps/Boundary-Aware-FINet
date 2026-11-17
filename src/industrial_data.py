"""Audited custom supervised MVTec AD split, independent of the COD experiment."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.training_data import FINetTrainingDataset, file_hash, manifest_hash

CATEGORIES = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper')
DOMAIN = 'Factory Quality Inspection'
PROTOCOL = 'custom_supervised_mvtec_v1'


def audit_original(root):
    root = Path(root).resolve()
    records = []
    for category in CATEGORIES:
        if not (root / category / 'train' / 'good').is_dir():
            raise ValueError(f'Missing original MVTec category: {category}')
        used_masks = set()
        for original_split in ('train', 'test'):
            images = sorted((root / category / original_split).glob('*/*.png'))
            if not images:
                raise ValueError(f'Missing original {category}/{original_split} images.')
            for image_path in images:
                defect = image_path.parent.name
                if original_split == 'train' and defect != 'good':
                    raise ValueError('Unexpected defective image in the official normal-only training split.')
                mask_path = root / category / 'ground_truth' / defect / f'{image_path.stem}_mask.png'
                with Image.open(image_path) as source:
                    source.load()
                    rgb = np.asarray(source.convert('RGB'))
                    size = source.size
                target = np.zeros((size[1], size[0]), dtype=np.uint8)
                if defect != 'good':
                    if not mask_path.is_file():
                        raise ValueError(f'Missing defect mask: {mask_path}')
                    with Image.open(mask_path) as source:
                        source.load()
                        if source.size != size:
                            raise ValueError(f'Mismatched image/mask dimensions: {image_path}')
                        target = (np.asarray(source.convert('L')) > 0).astype(np.uint8) * 255
                    if not target.any():
                        raise ValueError(f'Annotated abnormal image has an empty mask: {mask_path}')
                    used_masks.add(mask_path.resolve())
                tiny = np.asarray(Image.fromarray(target).resize((384, 384), Image.Resampling.NEAREST))
                sample_id = f'mvtec_{category}_{original_split}_{defect}_{image_path.stem}'
                records.append({'id': sample_id, 'category': category, 'defect_type': defect,
                                'is_defective': defect != 'good', 'original_split': original_split,
                                'source': 'MVTec AD', 'original_image': image_path.relative_to(root).as_posix(),
                                'original_mask': mask_path.relative_to(root).as_posix() if defect != 'good' else None,
                                'original_image_sha256': file_hash(image_path),
                                'original_mask_sha256': file_hash(mask_path) if defect != 'good' else None,
                                'pixel_sha256': hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest(),
                                'size_wh': list(size), 'foreground_pixels_native': int((target > 0).sum()),
                                'foreground_pixels_384': int((tiny > 0).sum())})
        available_masks = {path.resolve() for path in (root / category / 'ground_truth').glob('*/*.png')}
        if available_masks != used_masks:
            raise ValueError(f'Unpaired ground-truth masks in {category}.')
    if len(records) != 5354:
        raise ValueError(f'Expected original full 5354-image MVTec AD, found {len(records)}.')
    return records


def supervised_partitions(records, seed=123456):
    """70/15/15 by category, original split and defect type; group decoded duplicates."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record['pixel_sha256']].append(record)
    strata = defaultdict(list)
    for group in grouped.values():
        labels = {(r['category'], r['original_split'], r['defect_type']) for r in group}
        if len(labels) != 1 or len({r.get('original_mask_sha256') for r in group}) != 1:
            raise ValueError('Decoded duplicate crosses categories or annotations; needs explicit review.')
        strata[next(iter(labels))].append(sorted(r['id'] for r in group))
    result = {'train_ids': [], 'validation_ids': [], 'test_ids': []}
    for label, groups in sorted(strata.items()):
        rng = random.Random(f'{seed}|{label}')
        groups.sort()
        rng.shuffle(groups)
        # A stratum with fewer than three independent samples stays entirely held out.
        if len(groups) < 3:
            result['test_ids'].extend(name for group in groups for name in group)
            continue
        held = max(1, min(round(len(groups) * .15), (len(groups) - 1) // 2))
        for split, subset in (('validation_ids', groups[:held]), ('test_ids', groups[held:2*held]),
                              ('train_ids', groups[2*held:])):
            result[split].extend(name for group in subset for name in group)
    return {key: sorted(value) for key, value in result.items()}


def prepare_supervised(original_root, destination, categories=CATEGORIES, seed=123456):
    original_root, destination = Path(original_root).resolve(), Path(destination).resolve()
    selected = tuple(sorted(categories))
    if not selected or not set(selected).issubset(CATEGORIES):
        raise ValueError('Choose a nonempty subset of the 15 original categories.')
    manifest_path = destination / 'split_manifest.json'
    if manifest_path.exists():
        saved = load_industrial_manifest(manifest_path, destination)
        if saved['categories'] != list(selected) or saved['seed'] != seed:
            raise ValueError('Existing industrial split uses different categories/seed; choose another directory.')
        return saved
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Preserve partially prepared data; use a new destination or inspect it before retrying.')
    all_records = audit_original(original_root)
    records = [r.copy() for r in all_records if r['category'] in selected]
    partitions = supervised_partitions(records, seed)
    for name in ('Imgs', 'GT'):
        (destination / name).mkdir(parents=True, exist_ok=True)
    for record in records:
        image = destination / 'Imgs' / f"{record['id']}.png"
        source = original_root / record['original_image']
        try:
            os.link(source, image)
        except OSError:
            shutil.copyfile(source, image)
        width, height = record['size_wh']
        if record['original_mask']:
            with Image.open(original_root / record['original_mask']) as mask:
                binary = (np.asarray(mask.convert('L')) > 0).astype(np.uint8) * 255
        else:
            binary = np.zeros((height, width), dtype=np.uint8)
        mask_path = destination / 'GT' / f"{record['id']}.png"
        Image.fromarray(binary).save(mask_path)
        record.update(image=image.relative_to(destination).as_posix(), mask=mask_path.relative_to(destination).as_posix(),
                      image_sha256=file_hash(image), mask_sha256=file_hash(mask_path))
    manifest = {'schema': 1, 'application_domain': DOMAIN, 'protocol': PROTOCOL, 'seed': seed,
                'categories': list(selected), 'scope': 'all 15 categories' if len(selected) == 15 else 'documented subset',
                'official_benchmark_eligible': False, 'smoke_only': False,
                'source_counts': {'MVTec AD': len(records)}, 'original_total_images_audited': len(all_records),
                'original_category_counts': dict(Counter(r['category'] for r in all_records)),
                'split_policy': '70/15/15 within category/original-split/defect-type; decoded duplicates grouped; strata <3 all held out.',
                'official_protocol_note': 'Official abnormal test annotations enter custom training. These models cannot claim official anomaly-benchmark results. Original archive remains immutable for independent normal-only experiments.',
                'license': 'CC BY-NC-SA 4.0; https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
                'records': records, **partitions}
    manifest['fingerprint'] = manifest_hash(manifest)
    validate_partitions(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def validate_partitions(manifest):
    if manifest.get('protocol') != PROTOCOL or manifest.get('application_domain') != DOMAIN:
        raise ValueError('Not a declared custom supervised industrial manifest.')
    records = manifest['records']
    ids = {r['id'] for r in records}
    if len(ids) != len(records):
        raise ValueError('Duplicate sample ID.')
    seen_ids, seen_hashes = set(), set()
    for key in ('train_ids', 'validation_ids', 'test_ids'):
        names = manifest[key]
        if not names or len(set(names)) != len(names) or set(names) & seen_ids or not set(names) <= ids:
            raise ValueError('Empty, duplicate, unknown or overlapping industrial split.')
        hashes = {r['pixel_sha256'] for r in records if r['id'] in set(names)}
        if hashes & seen_hashes:
            raise ValueError('Decoded image content leaks between industrial splits.')
        seen_ids.update(names)
        seen_hashes.update(hashes)
    if seen_ids != ids:
        raise ValueError('Split does not cover every audited image.')


def load_industrial_manifest(path, root):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('fingerprint') != manifest_hash(manifest):
        raise ValueError('Industrial split fingerprint is invalid.')
    validate_partitions(manifest)
    root = Path(root).resolve()
    def verify_record(record):
        for key in ('image', 'mask'):
            path = (root / record[key]).resolve()
            if not path.is_relative_to(root) or file_hash(path) != record[f'{key}_sha256']:
                raise ValueError(f'Industrial {key} changed: {record["id"]}')
    # Kaggle input files have high per-file latency. Keep every path/hash check,
    # complete all checks before returning, and leave sampling/RNG untouched.
    started = perf_counter()
    records = manifest['records']
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, _ in enumerate(pool.map(verify_record, records), 1):
            if len(records) >= 1000 and (count % 500 == 0 or count == len(records)):
                print(json.dumps({'event':'industrial_integrity_progress', 'records_verified':count,
                                  'total_records':len(records), 'seconds':perf_counter()-started}), flush=True)
    return manifest


class IndustrialDataset(FINetTrainingDataset):
    def __init__(self, root, records, ids, augment=False):
        super().__init__(root, records, ids, augment=False)
        # Preserve upstream RGB normalization, YCbCr DCT and frequency statistics.
        # Industrial-only: no label pepper noise, crop, rotation or color jitter.
        # Normal examples must stay empty and thin defects must not be cropped away.
        self.rVFlip = augment
        self.gt_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])


def sampling_weights(records):
    counts = Counter((r['category'], r['is_defective']) for r in records)
    # Equal total probability per category × normal/abnormal stratum, with replacement.
    # The loss itself is unchanged; the same sampler is used for A and B.
    return [1.0 / counts[(r['category'], r['is_defective'])] for r in records]
