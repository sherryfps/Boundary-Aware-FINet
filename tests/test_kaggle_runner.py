"""Session progress must never present imported Colab step tails as new work."""
import json

from scripts.run_kaggle_project import KaggleRunner, latest_session_step
from scripts.run_colab_project import ProjectRunner


def test_imported_partial_epoch_is_not_live_kaggle_progress(tmp_path):
    path = tmp_path / 'steps.jsonl'
    path.write_text(json.dumps({'epoch': 15, 'global_step': 12900, 'loss': 2.0}) + '\n')
    offset = path.stat().st_size
    assert latest_session_step(path, offset, 14) is None
    with path.open('a') as handle:
        handle.write(json.dumps({'event': 'resume', 'completed_epochs': 14}) + '\n')
    assert latest_session_step(path, offset, 14) is None
    with path.open('a') as handle:
        handle.write(json.dumps({'epoch': 15, 'global_step': 12727, 'loss': 1.8}) + '\n')
    assert latest_session_step(path, offset, 14)['global_step'] == 12727


def test_partial_trailing_line_and_absent_file_do_not_invent_progress(tmp_path):
    path = tmp_path / 'steps.jsonl'
    assert latest_session_step(path, 0, 14) is None
    path.write_text('{"epoch":15')
    assert latest_session_step(path, 0, 14) is None


def test_evaluation_checks_required_graph_and_preserves_report(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ProjectRunner, 'command', lambda self, stage, arguments: calls.append((stage, arguments)))
    runner = KaggleRunner.__new__(KaggleRunner)
    runner.destination = tmp_path
    runner.command('evaluation_dependency_check', ['-m', 'pip', 'check'])
    assert calls == [('evaluation_dependency_check', ['scripts/check_evaluation_environment.py',
                     '--report', tmp_path / 'evaluation_environment.json'])]
