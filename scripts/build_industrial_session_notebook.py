"""Build the portable boundary continuation notebook from the tested local source."""

import base64
import hashlib
import json
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[1]


def build():
    notebook = json.loads((ROOT/'notebooks/industrial_training_kaggle.ipynb').read_text(encoding='utf-8'))
    updates = {name: (ROOT/name).read_text(encoding='utf-8') for name in (
        'src/industrial_training.py', 'scripts/resume_industrial_boundary_final.py',
        'tests/test_industrial_final_only.py')}
    blob = base64.b64encode(zlib.compress(json.dumps(updates).encode(), 9)).decode()
    setup = ''.join(next(c for c in notebook['cells'] if c['cell_type'] == 'code')['source'])
    patch = f'''# Exact tested checkpoint-policy/source update; original model/loss/settings retained.
import base64, zlib
session_updates=json.loads(zlib.decompress(base64.b64decode({blob!r})))
for relative,content in session_updates.items():
    target=PROJECT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content,encoding='utf-8')
print('SESSION_CHECKPOINT_SOURCE',json.dumps({{p:hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in session_updates}}),flush=True)
'''
    setup = 'from time import monotonic\nSESSION_STARTED_AT = monotonic()\n' + setup.replace('os.chdir(PROJECT)', patch+'\nos.chdir(PROJECT)', 1)
    setup += "\nsubprocess.run([sys.executable,'-m','pytest','tests/test_industrial_final_only.py','-q'],cwd=PROJECT,check=True)\n"
    sources = [
        ('markdown', '''# Industrial boundary continuation: one checkpoint before session end
Resume the verified baseline 200 / boundary 6 recovery, retaining the complete original 200-epoch schedule.
No per-epoch last/best saves. Save full recovery once at the session boundary (target 11h45 from first cell, including setup; leave 15 minutes for output persistence), or at epoch 200 if reached sooner. Kaggle may interrupt earlier. Resume from the newest saved output in the next authorized session. The completed baseline is verified unchanged. Distribution defaults to probe only.
'''),
        ('code', setup),
        ('code', "subprocess.run([sys.executable,'-u','scripts/resume_industrial_boundary_final.py','--probe-only','--session-start-monotonic',str(SESSION_STARTED_AT)],cwd=PROJECT,check=True)\n"),
        ('code', "RUN_BOUNDARY_TRAINING = False\nif RUN_BOUNDARY_TRAINING:\n    subprocess.run([sys.executable,'-u','scripts/resume_industrial_boundary_final.py','--session-start-monotonic',str(SESSION_STARTED_AT)],cwd=PROJECT,check=True)\nelse:\n    print('Probe only; official training not started.')\n"),
        ('code', "p=Path('/kaggle/working/FINetIndustrial/experiments/boundary/last.pth')\nsaved=torch.load(p,map_location='cpu',weights_only=True)\nprint('BOUNDARY_SAVED_EPOCH',saved['completed_epochs'],'of',saved['config']['epochs'],flush=True)\nprint('CHECKPOINT_SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)\nprint('RECOVERY_ZIP_EXISTS',Path('/kaggle/working/industrial_recovery.zip').is_file(),flush=True)\n")]
    notebook['cells'] = []
    for kind, source in sources:
        cell = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            compile(source, '<notebook>', 'exec')
            cell.update(execution_count=None, outputs=[])
        notebook['cells'].append(cell)
    target = ROOT/'notebooks/industrial_boundary_session_kaggle.ipynb'
    target.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    print(json.dumps({'notebook': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n,s in updates.items()}}))


if __name__ == '__main__':
    build()
