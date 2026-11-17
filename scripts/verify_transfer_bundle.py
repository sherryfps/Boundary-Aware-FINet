"""Verify every file of a FINet migration bundle using only the standard library."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
from time import monotonic


def digest_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_bundle(root, workers=8, require_completion=True):
    root = Path(root).resolve()
    manifest = root / 'manifest.jsonl'
    completion = root / 'TRANSFER_COMPLETE.json'
    if require_completion and not completion.is_file():
        raise ValueError('No successful TRANSFER_COMPLETE.json; this may be an unfinished copy.')
    expected = json.loads(completion.read_text(encoding='utf-8')) if completion.exists() else None
    if expected and expected.get('status') != 'passed':
        raise ValueError('The bundle completion status is not passed.')
    if expected and expected['manifest_sha256'] != digest_file(manifest):
        raise ValueError('Manifest digest differs from the completion report.')
    entries = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    names = [entry['path'] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest path.')

    def check(entry):
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or any(p in {'.', '..'} or ':' in p for p in relative.parts):
            raise ValueError('Unsafe manifest path.')
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError('Manifest path escapes the bundle.')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest_file(path) != entry['sha256']:
            raise ValueError(f'Missing or changed file: {entry["path"]}')
        return entry['bytes']

    started, count, size = monotonic(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, entry) for entry in entries]
        for future in as_completed(futures):
            size += future.result()
            count += 1
            if count % 5000 == 0:
                print(json.dumps({'verified_files': count, 'total_files': len(entries),
                                  'verified_GiB': round(size / 2**30, 3)}), flush=True)
    if expected and (count != expected['files'] or size != expected['bytes']):
        raise ValueError('Manifest totals differ from the completion report.')
    result = {'status': 'passed', 'files': count, 'bytes': size,
              'GiB': round(size / 2**30, 3), 'seconds': round(monotonic() - started, 3),
              'manifest_sha256': digest_file(manifest)}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    verify_bundle(args.root, args.workers)
