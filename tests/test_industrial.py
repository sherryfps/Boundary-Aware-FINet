"""Synthetic protocol tests only; these fixtures are never model evidence."""

from collections import defaultdict
from copy import deepcopy
import json

import numpy as np
import pytest

from src.industrial_data import DOMAIN, PROTOCOL, sampling_weights, supervised_partitions, validate_partitions
from src.industrial_training import IndustrialConfig, aggregate_metrics, segmentation_metrics


def fixture_records():
    records = []
    for category in ('bottle', 'screw'):
        for defect in ('good', 'scratch'):
            for index in range(20):
                identity = f'{category}_{defect}_{index}'
                records.append({'id': identity, 'category': category, 'original_split': 'test',
                                'defect_type': defect, 'is_defective': defect != 'good',
                                'pixel_sha256': identity, 'original_mask_sha256': identity if defect != 'good' else None})
    return records


def test_supervised_split_reproducible_complete_and_stratified():
    records = fixture_records()
    partitions = supervised_partitions(records)
    assert partitions == supervised_partitions(list(reversed(records)))
    assert partitions != supervised_partitions(records, seed=4)
    manifest = {'application_domain': DOMAIN, 'protocol': PROTOCOL, 'records': records, **partitions}
    validate_partitions(manifest)
    assert [len(partitions[k]) for k in ('train_ids', 'validation_ids', 'test_ids')] == [56, 12, 12]
    for ids in partitions.values():
        assert {r['category'] for r in records if r['id'] in ids} == {'bottle', 'screw'}


def test_duplicate_pixels_grouped_and_leakage_rejected():
    records = fixture_records()
    duplicate = {**records[0], 'id': 'duplicate'}
    records.append(duplicate)
    partitions = supervised_partitions(records)
    assert next(k for k,v in partitions.items() if records[0]['id'] in v) == next(k for k,v in partitions.items() if 'duplicate' in v)
    manifest = {'application_domain': DOMAIN, 'protocol': PROTOCOL, 'records': records, **partitions}
    validate_partitions(manifest)
    broken = deepcopy(manifest)
    source = next(k for k in partitions if 'duplicate' in partitions[k])
    target = next(k for k in partitions if k != source)
    broken[source].remove('duplicate')
    broken[target].append('duplicate')
    with pytest.raises(ValueError, match='leaks'):
        validate_partitions(broken)


def test_balanced_sampler_does_not_modify_masks_or_loss():
    records = fixture_records() + fixture_records()[:10]
    total = defaultdict(float)
    for r, weight in zip(records, sampling_weights(records)):
        total[(r['category'], r['is_defective'])] += weight
    assert all(value == pytest.approx(1) for value in total.values())


def test_empty_masks_and_macro_defective_metrics():
    empty = np.zeros((32, 32), dtype=np.float32)
    assert segmentation_metrics(empty, empty.astype(bool))['dice'] == 1
    target = empty.astype(bool)
    target[4:8, 4:8] = True
    assert segmentation_metrics(empty, target)['dice'] == 0
    assert segmentation_metrics(target.astype(np.float32), target)['iou'] == 1
    rows = [{'category': 'bottle', 'is_defective': True, **segmentation_metrics(empty, target)},
            {'category': 'bottle', 'is_defective': False, **segmentation_metrics(empty, empty.astype(bool))}]
    summary = aggregate_metrics(rows)
    assert summary['macro_defective_category_dice'] == 0
    assert summary['mean_per_image']['dice'] == .5


def test_industrial_a_b_configs_are_matched():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    a = json.loads((root/'configs/industrial_baseline.json').read_text())
    b = json.loads((root/'configs/industrial_boundary.json').read_text())
    assert b.pop('boundary_lambda') == .1 and a.pop('boundary_lambda') == 0
    a.pop('output_dir'); b.pop('output_dir')
    assert a == b
    IndustrialConfig(**json.loads((root/'configs/industrial_baseline.json').read_text())).validate()
    with pytest.raises(ValueError, match='protocol'):
        IndustrialConfig(dataset_protocol='official_anomaly_benchmark').validate()


def saved_integrity_fixture(root):
    from src.training_data import file_hash, manifest_hash
    records = fixture_records()
    root.mkdir()
    for record in records:
        for kind in ('image','mask'):
            path = root/f'{record["id"]}_{kind}.bin'
            path.write_bytes((record['id']+kind).encode())
            record[kind] = path.name
            record[kind+'_sha256'] = file_hash(path)
    manifest = {'application_domain':DOMAIN, 'protocol':PROTOCOL, 'records':records,
                **supervised_partitions(records)}
    manifest['fingerprint'] = manifest_hash(manifest)
    path = root/'split_manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_integrity_verifies_every_file_and_preserves_sampling_rng(tmp_path, monkeypatch):
    from collections import Counter
    import random
    from threading import Lock
    import torch
    import src.industrial_data as module
    path, manifest = saved_integrity_fixture(tmp_path/'data')
    original = module.file_hash
    calls, lock = [], Lock()
    def tracked(file):
        result = original(file)
        with lock:
            calls.append(file.name)
        return result
    monkeypatch.setattr(module,'file_hash',tracked)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    numpy_rng = np.random.get_state()
    assert module.load_industrial_manifest(path,path.parent) == manifest
    assert Counter(calls) == Counter(r[k] for r in manifest['records'] for k in ('image','mask'))
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(),torch_rng)
    actual = np.random.get_state()
    assert actual[0] == numpy_rng[0] and np.array_equal(actual[1],numpy_rng[1]) and actual[2:] == numpy_rng[2:]


def test_integrity_rejects_a_corrupt_late_file(tmp_path):
    from src.industrial_data import load_industrial_manifest
    path, manifest = saved_integrity_fixture(tmp_path/'data')
    (path.parent/manifest['records'][-1]['mask']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Industrial mask changed'):
        load_industrial_manifest(path,path.parent)


def test_integrity_rejects_path_escape_even_with_matching_hash(tmp_path):
    from src.industrial_data import load_industrial_manifest
    from src.training_data import file_hash, manifest_hash
    path, manifest = saved_integrity_fixture(tmp_path/'data')
    outside = tmp_path/'outside.bin'
    outside.write_bytes(b'outside dataset')
    manifest['records'][-1].update(image='../outside.bin',image_sha256=file_hash(outside))
    manifest['fingerprint'] = manifest_hash(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Industrial image changed'):
        load_industrial_manifest(path,path.parent)
