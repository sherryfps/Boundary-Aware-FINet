"""Download the SINet-V2-linked COD training archive and audit the original pairs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_data import create_manifest, file_hash

FILE_ID = '1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG'
SOURCE = f'https://drive.google.com/file/d/{FILE_ID}/view'
EXPECTED_SHA256 = '255146828426e144c54867879824bcef27075b9e2386899fa0220f77551dc7be'


def extract_pairs(archive: Path, destination: Path):
    """Extract only the observed image/mask layout; never use extractall."""
    with ZipFile(archive) as zipped:
        selected = []
        names = set()
        for member in zipped.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] == '__MACOSX' or member.is_dir():
                continue
            if (len(parts) != 3 or parts[0] != 'COD-TrainDataset'
                    or parts[1] not in {'Imgs', 'GT'} or '\\' in member.filename
                    or any(part in {'.', '..'} or ':' in part for part in parts)):
                raise ValueError(f'Unexpected archive member: {member.filename}')
            suffix = PurePosixPath(parts[2]).suffix
            if suffix not in ({'.jpg', '.png'} if parts[1] == 'Imgs' else {'.png'}):
                raise ValueError(f'Unexpected training file: {member.filename}')
            relative = Path(parts[1]) / parts[2]
            if str(relative).casefold() in names:
                raise ValueError(f'Duplicate archive path: {relative}')
            names.add(str(relative).casefold())
            selected.append((member, relative))
        if not selected or sum(member.file_size for member, _ in selected) > 10 * 1024**3:
            raise ValueError('Empty or unexpectedly large training archive.')
        for member, relative in selected:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # CRC verifies that a partial earlier extraction is not reused.
                import zlib
                crc = 0
                with target.open('rb') as existing:
                    for block in iter(lambda: existing.read(1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size == member.file_size and crc == member.CRC:
                    continue
                raise FileExistsError(f'Existing dataset file differs: {target}')
            temporary = target.with_suffix(target.suffix + '.partial')
            with zipped.open(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/downloads/COD-TrainDataset.zip'))
    parser.add_argument('--data-root', type=Path, default=Path('data/TrainDataset'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/splits/cod_camo_seed123456.json'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--canonicalize-order', action='store_true', help='Back up an untrained manifest and normalize record order without changing split membership')
    args = parser.parse_args()
    if not args.archive.is_file():
        if not args.download:
            parser.error('Archive missing; use --download to obtain the public research release.')
        import gdown
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=FILE_ID, output=str(args.archive), quiet=False, resume=True)
        if not result:
            raise RuntimeError('Official download failed; no dataset prepared.')
    archive_hash = file_hash(args.archive)
    if archive_hash != EXPECTED_SHA256:
        raise ValueError('Archive differs from the verified November 11 release; no extraction performed.')
    count = extract_pairs(args.archive, args.data_root)
    # This official release includes grayscale CAMO edge labels. Retain its bytes
    # and values, as upstream does; never threshold structure-loss supervision.
    manifest = create_manifest(args.data_root, allow_soft_masks=True, group_duplicates=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        old = json.loads(args.manifest.read_text(encoding='utf-8'))
        if old != manifest:
            normalized = {**old, 'records': sorted(old['records'], key=lambda record: record['id'])}
            normalized['fingerprint'] = manifest['fingerprint']
            if not args.canonicalize_order or normalized != manifest:
                raise FileExistsError('Existing split differs. Refusing to replace experiment data.')
            backup = args.manifest.with_suffix('.pre-portable-order.json')
            if backup.exists():
                raise FileExistsError('Manifest backup already exists; inspect it before another migration.')
            shutil.copy2(args.manifest, backup)
            args.manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    else:
        args.manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    provenance = {'source': SOURCE, 'linked_by': 'https://github.com/GewelsJI/SINet-V2',
                  'archive_sha256': archive_hash, 'archive_bytes': args.archive.stat().st_size,
                  'extracted_files': count, 'counts': manifest['source_counts'],
                  'masks_with_soft_values': sum(record['mask_soft_pixels'] > 0 for record in manifest['records']),
                  'unique_image_hashes': len({record['image_sha256'] for record in manifest['records']}),
                  'split_fingerprint': manifest['fingerprint'],
                  'obtained_at_utc': datetime.now(timezone.utc).isoformat(),
                  'note': 'Hash measured locally, not an author-published checksum. Original bytes retained.'}
    (args.data_root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**provenance, 'train': len(manifest['train_ids']),
                      'validation': len(manifest['validation_ids'])}), flush=True)


if __name__ == '__main__':
    main()
