"""Bounded Kaggle sessions supervising the existing FINet trainer and A/B protocol."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from time import monotonic
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.kaggle_migration import prepare, resume_probe
from scripts.run_colab_project import ProjectRunner, checkpoint_summary, read_json, record
from src.training_data import file_hash


class SessionBoundary(Exception):
    """An intentionally bounded session ended with recoverable epoch checkpoints."""


def latest_session_step(path, starting_offset, completed):
    """Read only newly appended steps; imported partial-epoch tails are historical."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= starting_offset:
        return None
    with path.open('rb') as handle:
        handle.seek(max(starting_offset, path.stat().st_size - 8192))
        tail = handle.read().decode('utf-8', errors='replace').splitlines()
    for value in reversed(tail):
        try:
            row = json.loads(value)
        except ValueError:
            continue
        if row.get('epoch', 0) > completed and 'loss' in row:
            return row
    return None


def snapshot(work):
    work = Path(work)
    target = work.parent / 'finet_kaggle_recovery_latest.zip'
    temporary = target.with_suffix('.tmp')
    allowed = {'.pth', '.json', '.jsonl', '.log', '.md'}
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as zipped:
        for path in work.rglob('*'):
            if path.is_file() and path.suffix in allowed and not path.name.endswith('.tmp'):
                zipped.write(path, Path('FINet') / path.relative_to(work))
    temporary.replace(target)
    record(work / 'migration/recovery_archive.json',
           {'path': str(target), 'bytes': target.stat().st_size, 'sha256': file_hash(target),
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'Raw epoch checkpoints are also in /kaggle/working/FINet; save notebook outputs to persist them.'})
    return target


class KaggleRunner(ProjectRunner):
    def __init__(self, config, smoke, destination, hours, epochs):
        super().__init__(config, smoke, destination)
        self.work = self.destination.parents[1]
        self.deadline = monotonic() + hours * 3600
        self.hard_deadline = min(self.deadline + 3600, monotonic() + 10.5 * 3600)
        self.max_new_epochs = epochs
        self.new_epochs = 0

    def evaluate(self, baseline, boundary, test_root=None):
        return super().evaluate(baseline, boundary, '/kaggle/temp/finet_test_data')

    def command(self, stage, arguments):
        if stage == 'evaluation_dependency_check':
            return super().command(stage, ['scripts/check_evaluation_environment.py', '--report',
                                          self.destination / 'evaluation_environment.json'])
        if stage not in {'baseline', 'boundary'}:
            return super().command(stage, arguments)
        config_path = Path(arguments[arguments.index('--config') + 1])
        config = read_json(config_path)
        run_dir = Path(config['output_dir'])
        last = run_dir / 'last.pth'
        completed = checkpoint_summary(last)['completed_epochs'] if last.is_file() else 0
        seen = completed
        self.status(stage, 'running', resumed_completed_epochs=completed,
                    session_max_new_epochs=self.max_new_epochs, gpu_index=0)
        log = self.destination / 'logs' / f'{stage}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        steps = run_dir / 'steps.jsonl'
        starting_offset = steps.stat().st_size if steps.is_file() else 0
        process = subprocess.Popen([sys.executable, '-u', *map(str, arguments)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        lines = queue.Queue()

        def reader():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=reader, daemon=True).start()
        progress_time = monotonic()
        try:
            with log.open('a', encoding='utf-8') as output:
                while True:
                    try:
                        line = lines.get(timeout=1)
                    except queue.Empty:
                        line = ''
                    if line is None:
                        break
                    if line:
                        output.write(line)
                        output.flush()
                        print(line, end='', flush=True)
                        try:
                            row = json.loads(line)
                        except (ValueError, TypeError):
                            row = {}
                        if 'epoch' in row and 'validation' in row:
                            epoch = row['epoch']
                            if epoch != seen + 1:
                                raise RuntimeError('Completed epoch history did not continue sequentially.')
                            self.new_epochs += epoch - seen
                            seen = epoch
                            saved = checkpoint_summary(last)
                            if saved['completed_epochs'] != epoch:
                                raise RuntimeError('Printed epoch does not match the saved recovery checkpoint.')
                            record(self.work / 'migration/live_training.json',
                                   {'stage': stage, 'completed_epoch': row, 'saved_checkpoint': str(last),
                                    'checkpoint_sha256': file_hash(last), 'new_epochs_this_session': self.new_epochs})
                            snapshot(self.work)
                            if (self.new_epochs >= self.max_new_epochs or monotonic() >= self.deadline):
                                process.terminate()
                                process.wait(timeout=60)
                                raise SessionBoundary(f'Saved through {stage} epoch {epoch}; session budget reached.')
                    if monotonic() - progress_time >= 30:
                        row = latest_session_step(steps, starting_offset, completed)
                        if row is not None:
                            print('KAGGLE_TRAINING_PROGRESS', json.dumps({k: row[k] for k in
                                  ('epoch', 'global_step', 'loss', 'learning_rate', 'finite_gradients')}), flush=True)
                        progress_time = monotonic()
                    if monotonic() >= self.hard_deadline:
                        process.terminate()
                        process.wait(timeout=60)
                        raise SessionBoundary('Safety time budget reached; resume the last completed epoch. Any partial epoch is replayed.')
                code = process.wait()
                if code:
                    raise RuntimeError(f'{stage} trainer exited {code}; inspect {log}.')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--work', type=Path, default=Path('/kaggle/working/FINet'))
    parser.add_argument('--run-project', action='store_true')
    parser.add_argument('--max-session-hours', type=float, default=8)
    parser.add_argument('--max-new-epochs', type=int, default=40)
    args = parser.parse_args()
    if not 0 < args.max_session_hours <= 8 or args.max_new_epochs < 1:
        parser.error('Use a positive session budget of at most eight hours and at least one new epoch.')
    config_path = prepare(args.transfer, args.work)
    smoke = resume_probe(config_path)
    if not args.run_project:
        print('GPU smoke completed; long training was not launched.', flush=True)
        snapshot(args.work)
        return
    runner = KaggleRunner(config_path, smoke, config_path.parent, args.max_session_hours, args.max_new_epochs)
    try:
        runner.run(include_evaluation=True)
    except SessionBoundary as exc:
        runner.status('session', 'checkpointed_session_complete', note=str(exc),
                      completed={name: checkpoint_summary(path / 'last.pth')['completed_epochs']
                                 for path in (args.work / 'experiments').iterdir()
                                 if (path / 'last.pth').is_file() for name in [path.name]},
                      next_action='Attach this saved version output as recovery input and run the same notebook again.')
    finally:
        archive = snapshot(args.work)
        print('RECOVERY_OUTPUT', str(archive), flush=True)


if __name__ == '__main__':
    main()
