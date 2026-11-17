"""Copy the entire local research project with full source/destination SHA proof."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_transfer_bundle import digest_file, verify_bundle

EXCLUDED_DIRS = {'.venv', 'venv', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
EXCLUDED_FILES = {'.env', 'kaggle.json', 'auth.json', '.git-credentials'}


def gather():
    selected, excluded = [], []
    for base, directories, files in os.walk(ROOT):
        base = Path(base)
        keep = []
        for name in directories:
            path = base / name
            if path.is_symlink():
                raise ValueError(f'Unexpected link: {path.relative_to(ROOT)}')
            if name in EXCLUDED_DIRS or name.startswith('.venv-'):
                excluded.append(str(path.relative_to(ROOT)))
            else:
                keep.append(name)
        directories[:] = keep
        for name in files:
            path = base / name
            relative = path.relative_to(ROOT)
            if path.is_symlink():
                raise ValueError(f'Unexpected link: {relative}')
            if name in EXCLUDED_FILES or name.startswith('.env.') or path.suffix.lower() in {'.pyc', '.pyo', '.pem', '.key'}:
                excluded.append(str(relative))
            else:
                selected.append((path, relative, path.stat()))
    return selected, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or destination.is_relative_to(ROOT) or ROOT.is_relative_to(destination):
        raise ValueError('The new backup must be separate from the existing project.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Preserve existing backup; choose an empty/new destination.')
    selected, excluded = gather()
    total_bytes = sum(stat.st_size for _, _, stat in selected)
    if shutil.disk_usage(destination.parent).free < total_bytes + 2**30:
        raise ValueError('Not enough free disk space for the full copy plus headroom.')
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / 'project'
    started, copied, copied_bytes, entries = monotonic(), 0, 0, []
    print(json.dumps({'stage': 'copy', 'source_files': len(selected), 'source_GiB': round(total_bytes / 2**30, 3),
                      'destination': str(destination)}), flush=True)

    def copy_one(record):
        source, relative, before = record
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open('rb') as reader, target.open('xb') as writer:
            while block := reader.read(4 * 1024 * 1024):
                digest.update(block)
                writer.write(block)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during copy: {relative}')
        shutil.copystat(source, target)
        return {'path': (Path('project') / relative).as_posix(), 'bytes': before.st_size,
                'sha256': digest.hexdigest()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(copy_one, record) for record in selected]
        for future in as_completed(futures):
            entry = future.result()
            entries.append(entry)
            copied += 1
            copied_bytes += entry['bytes']
            if copied % 5000 == 0:
                print(json.dumps({'copied_files': copied, 'total_files': len(selected),
                                  'copied_GiB': round(copied_bytes / 2**30, 3),
                                  'seconds': round(monotonic() - started, 1)}), flush=True)

    # These are generated migration artifacts, not modifications to copied research state.
    helpers = {
        'START_HERE.md': (ROOT / 'docs/TRANSFER_TO_NEW_LAPTOP.md').read_text(encoding='utf-8'),
        'CONTINUE_WITH_CODEX.txt': (
            'Continue this existing Boundary-Aware FINet / FINet Industrial project. Do not restart it.\n'
            'Use the project subfolder as the workspace. Read AGENTS.md and CODEX_HANDOFF.md, then the newest PROGRESS.md entries.\n'
            'Verify the backup, rebuild the local environment from requirements-transfer-cpu.txt, and test the installed epoch-200 models.\n'
            'Camouflage A and B are both complete at 200 epochs: never retrain them. Check existing Kaggle evaluation Version 6 for real saved results.\n'
            'Check the industrial job latest output before resuming: cloud progress may exceed the local epoch-96 recovery. Preserve all newer progress.\n'
            'Continue evaluation, fair comparison, industrial training/integration and final submission using measured evidence and authorized free compute.\n'
            'Ask only for real login/CAPTCHA/new permission/access blockers. Preserve original FINet/torchvision/DCT/structure loss and matched experiments.\n'),
        'AGENTS.md': (
            '# Existing FINet migration bundle\n\n'
            'The actual workspace is `project`. Read `START_HERE.md`, then `project/AGENTS.md` and `project/CODEX_HANDOFF.md`.\n'
            'This is a completed partial project handoff, not an instruction to initialize another repository or repeat trained epochs.\n'),
        'VERIFY_BACKUP.py': (ROOT / 'scripts/verify_transfer_bundle.py').read_text(encoding='utf-8'),
    }
    for name, value in helpers.items():
        path = destination / name
        path.write_text(value, encoding='utf-8')
        entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest_file(path)})
    manifest = destination / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in sorted(entries, key=lambda x: x['path'])),
                        encoding='utf-8')

    # Verify the relocated copy using the original installed interpreter; do not reinstall or train.
    import subprocess
    check = subprocess.run([sys.executable, '-B', 'scripts/verify_portable_install.py'], cwd=work,
                           capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError('Relocated-model check failed: ' + check.stderr[-2000:])
    # The real-image check produces one new evidence file; include its actual bytes in the manifest.
    new_report = work / 'outputs/new_laptop_verification.json'
    relative_report = new_report.relative_to(destination).as_posix()
    entries = [entry for entry in entries if entry['path'] != relative_report]
    entries.append({'path': relative_report, 'bytes': new_report.stat().st_size, 'sha256': digest_file(new_report)})
    manifest.write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in sorted(entries, key=lambda x: x['path'])),
                        encoding='utf-8')
    print(json.dumps({'stage': 'relocated_real_model_inference', 'status': 'passed'}), flush=True)
    for source, relative, before in selected:
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed after copy: {relative}; preserve this incomplete copy and take a new snapshot.')
    result = verify_bundle(destination, args.workers, require_completion=False)
    completion = {**result, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'source': str(ROOT), 'project_subfolder': 'project',
                  'source_files_copied': len(selected), 'excluded': sorted(excluded),
                  'preserved': ['source and uncommitted Git work', 'all local datasets', 'all local checkpoints',
                                'all local recovery archives and experimental evidence', 'portable dependency lock and written Codex context'],
                  'relocated_final_models_and_real_image': 'passed for both epoch-200 models',
                  'source_checkpoints_modified': False,
                  'cloud_jobs_modified': False,
                  'total_seconds': round(monotonic() - started, 3),
                  'note': 'manifest.jsonl and this final completion report are integrity-control files, not self-listed payload entries.'}
    (destination / 'TRANSFER_COMPLETE.json').write_text(json.dumps(completion, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'stage': 'complete', 'status': 'passed', 'destination': str(destination),
                      'files': result['files'], 'GiB': result['GiB'], 'seconds': completion['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
