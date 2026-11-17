"""Official-release compatibility and extraction safeguards."""

import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
import pytest
from scripts.prepare_official_data import extract_pairs
from src.training_data import audit_pairs, create_manifest, load_manifest, manifest_hash
from test_training import make_unit_pairs


def test_official_soft_values_remain_unchanged(tmp_path):
    make_unit_pairs(tmp_path)
    path = next((tmp_path / 'GT').glob('*.png'))
    with Image.open(path) as mask:
        values = np.asarray(mask).copy()
    values[8, 8] = 127
    Image.fromarray(values).save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='binary'):
        audit_pairs(tmp_path)
    manifest = create_manifest(tmp_path, allow_small=True, allow_soft_masks=True)
    assert sum(record['mask_soft_pixels'] for record in manifest['records']) == 1
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(manifest), encoding='utf-8')
    assert load_manifest(split, tmp_path) == manifest
    assert path.read_bytes() == before


def test_identical_images_stay_in_same_split_and_leakage_is_rejected(tmp_path):
    make_unit_pairs(tmp_path)
    images = sorted((tmp_path / 'Imgs').glob('*.png'))
    images[1].write_bytes(images[0].read_bytes())
    with pytest.raises(ValueError, match='Duplicate'):
        audit_pairs(tmp_path)
    manifest = create_manifest(tmp_path, allow_small=True, group_duplicates=True)
    first, second = images[0].stem, images[1].stem
    assert (first in manifest['train_ids']) == (second in manifest['train_ids'])
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(manifest), encoding='utf-8')
    load_manifest(split, tmp_path)
    # Maintain complete/disjoint names but deliberately put duplicate bytes across splits.
    for name in (first, second):
        for partition in ('train_ids', 'validation_ids'):
            if name in manifest[partition]:
                manifest[partition].remove(name)
    manifest['train_ids'].append(first)
    manifest['validation_ids'].append(second)
    manifest['fingerprint'] = manifest_hash(manifest)
    split.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate image content overlaps'):
        load_manifest(split, tmp_path)


def test_archive_rejects_traversal_without_writing(tmp_path):
    archive = tmp_path / 'invalid.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('COD-TrainDataset/Imgs/../../escape.jpg', b'invalid')
    with pytest.raises(ValueError, match='Unexpected archive member'):
        extract_pairs(archive, tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


def test_test_geometry_exclusions_are_explicit_and_source_files_unchanged(tmp_path):
    from src.evaluation_data import audit_evaluation_data
    make_unit_pairs(tmp_path)
    path = next((tmp_path/'GT').glob('*.png'))
    Image.new('L', (63, 80)).save(path)
    before = path.read_bytes()
    records, exclusions, total = audit_evaluation_data(tmp_path)
    assert total == 6 and len(records) == 5
    assert exclusions[0]['id'] == path.stem
    assert exclusions[0]['mask_size_wh'] == [63, 80]
    assert path.read_bytes() == before


def test_audit_order_is_literal_filename_order_across_platforms(tmp_path):
    make_unit_pairs(tmp_path)
    for folder in ('Imgs', 'GT'):
        old = tmp_path/folder/'COD10K-unit-000.png'
        old.rename(tmp_path/folder/'camourflage_unit.png')
    ids = [record['id'] for record in audit_pairs(tmp_path)]
    assert ids == sorted(ids)
    assert ids[-1] == 'camourflage_unit'
