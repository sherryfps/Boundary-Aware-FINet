"""Extract/audit the official test archive, retaining split-overlap evidence."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_data import audit_pairs, file_hash
from src.evaluation_data import audit_evaluation_data
from src.training import write_json

FILE_ID = '1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o'
EXPECTED = {'CAMO': 250, 'CHAMELEON': 76, 'COD10K': 2026, 'NC4K': 4121}


def prepare(archive, destination, training_manifest, report_path):
    archive, destination = Path(archive), Path(destination)
    selected = {}
    duplicates = 0
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] == '__MACOSX' or member.is_dir():
                continue
            if any(part in {'.', '..'} or ':' in part for part in parts) or '\\' in member.filename:
                raise ValueError('Unsafe test archive member.')
            if len(parts) != 4 or parts[0] != 'COD-TestDataset' or parts[1] not in EXPECTED:
                raise ValueError(f'Unexpected test archive layout: {member.filename}')
            if parts[2] not in {'Imgs', 'GT'}:
                continue  # Do not substitute supplied edges/instances for object masks.
            if PurePosixPath(parts[3]).suffix not in ({'.jpg', '.png'} if parts[2] == 'Imgs' else {'.png'}):
                raise ValueError('Unexpected image or mask suffix.')
            relative = Path(*parts[1:])
            if relative in selected:
                previous = selected[relative]
                if (previous.CRC, previous.file_size) != (member.CRC, member.file_size):
                    raise ValueError(f'Conflicting repeated archive entry: {relative}')
                duplicates += 1
            else:
                selected[relative] = member
        for relative, member in selected.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Preserve and verify existing decoded-release files rather than replace them.
            data = zipped.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise FileExistsError(f'Existing test file differs: {target}')
            else:
                temporary = target.with_suffix(target.suffix + '.partial')
                temporary.write_bytes(data)
                temporary.replace(target)
    development = json.loads(Path(training_manifest).read_text(encoding='utf-8'))
    development_ids = {record['id'] for record in development['records']}
    development_hashes = {record['image_sha256'] for record in development['records']}
    datasets = {}
    for name, expected in EXPECTED.items():
        records, invalid_geometry, released_pairs = audit_evaluation_data(destination / name)
        if released_pairs != expected:
            raise ValueError(f'{name} requires {expected} released pairs; found {released_pairs}.')
        overlap = [record['id'] for record in records if record['id'] in development_ids
                   or record['image_sha256'] in development_hashes]
        datasets[name] = {'released_pairs': released_pairs, 'valid_geometry_pairs': len(records), 'records': records,
                          'invalid_geometry_exclusions': invalid_geometry, 'development_overlap_ids': overlap,
                          'evaluation_pairs_after_exclusion': len(records) - len(overlap)}
    report = {'status': 'data audit only; no model scores',
              'source': f'https://drive.google.com/file/d/{FILE_ID}/view',
              'linked_by': 'https://github.com/GewelsJI/SINet-V2',
              'archive_sha256': file_hash(archive), 'archive_bytes': archive.stat().st_size,
              'identical_repeated_zip_entries': duplicates,
              'development_split_fingerprint': development['fingerprint'],
              'datasets': datasets, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'overlap_policy': 'Exclude matching names or byte-identical development images from final evaluation; keep original release files intact.'}
    write_json(Path(report_path), report)
    return {name: {key: value for key, value in data.items() if key != 'records'} for name, data in datasets.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/downloads/COD-TestDataset.zip'))
    parser.add_argument('--data-root', type=Path, default=Path('data/TestDataset'))
    parser.add_argument('--training-manifest', type=Path, default=Path('outputs/splits/cod_camo_seed123456.json'))
    parser.add_argument('--report', type=Path, default=Path('outputs/test_data_audit.json'))
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not args.archive.is_file() and args.download:
        import gdown
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        if not gdown.download(id=FILE_ID, output=str(args.archive), quiet=False, resume=True):
            raise RuntimeError('Official test download failed.')
    print(json.dumps(prepare(args.archive, args.data_root, args.training_manifest, args.report), indent=2))
