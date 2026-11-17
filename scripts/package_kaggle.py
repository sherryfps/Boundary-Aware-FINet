"""Build a private transfer containing current source, verified recovery and original pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.package_colab import package
from scripts.prepare_official_data import EXPECTED_SHA256
from src.training_data import file_hash, load_manifest


def build(snapshot, destination):
    snapshot = Path(snapshot)
    inventory = json.loads((snapshot / 'migration_inventory.json').read_text())
    for name, expected in inventory['files'].items():
        if file_hash(snapshot / name) != expected['sha256']:
            raise ValueError(f'Recovery snapshot changed: {name}')
    data = ROOT / 'data/TrainDataset'
    manifest = load_manifest(snapshot / 'splits/cod_camo_seed123456.json', data)
    archive = ROOT / 'data/downloads/COD-TrainDataset.zip'
    if file_hash(archive) != EXPECTED_SHA256:
        raise ValueError('Original official training archive changed.')
    source_zip = package(ROOT / 'outputs/kaggle_migration/current_source.zip')
    metadata = {'format': 1, 'original_training_archive_sha256': EXPECTED_SHA256,
                'split_fingerprint': manifest['fingerprint'], 'source_counts': manifest['source_counts'],
                'train_samples': len(manifest['train_ids']), 'validation_samples': len(manifest['validation_ids']),
                'recovery_files': inventory['files'], 'source_files': {}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'w', ZIP_DEFLATED) as bundle:
        with ZipFile(source_zip) as source:
            for info in source.infolist():
                content = source.read(info)
                bundle.writestr('project/' + info.filename, content)
                metadata['source_files'][info.filename] = hashlib.sha256(content).hexdigest()
        for name in inventory['files']:
            bundle.write(snapshot / name, 'recovery/' + name)
        bundle.write(snapshot / 'migration_inventory.json', 'recovery/migration_inventory.json')
        for record in manifest['records']:
            for key in ('image', 'mask'):
                relative = record[key]
                bundle.write(data / relative, 'data/TrainDataset/' + relative, compress_type=ZIP_STORED)
        bundle.writestr('transfer_manifest.json', json.dumps(metadata, indent=2))
    with ZipFile(destination) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Transfer archive integrity check failed.')
        count = len(bundle.infolist())
    return {'path': str(destination.resolve()), 'bytes': destination.stat().st_size,
            'sha256': file_hash(destination), 'members': count,
            'split_fingerprint': manifest['fingerprint']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'outputs/kaggle_migration/drive_snapshot')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/kaggle_migration/finet_kaggle_transfer.zip')
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot, args.output), indent=2))
