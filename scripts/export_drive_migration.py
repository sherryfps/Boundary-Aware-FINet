"""Run in a mounted Colab CPU session to snapshot existing FINet recovery files."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from google.colab import drive, files

drive.mount('/content/drive')
root = Path('/content/drive/MyDrive/FINet')
destination = Path('/content/finet_drive_migration.zip')
names = ('last.pth', 'best.pth', 'initialization.pth', 'config.json',
         'split_manifest.json', 'epochs.jsonl', 'steps.jsonl', 'hardware.json',
         'smoke_report.json')
selected = [p for p in root.rglob('*') if p.is_file() and (
    (p.relative_to(root).parts[0] == 'experiments' and p.name in names)
    or (p.relative_to(root).parts[0] == 'splits' and p.suffix == '.json')
    or (p.relative_to(root).parts[0] == 'code')
    or (p.relative_to(root).parts[0] == 'projects' and p.suffix in {'.json', '.log'}))]
assert root.joinpath('experiments/baseline_seed123456_b4_v1/last.pth') in selected
inventory = {'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'files': {}}
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for path in sorted(selected):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        inventory['files'][relative] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        archive.writestr(relative, content)
    archive.writestr('migration_inventory.json', json.dumps(inventory, indent=2))
print('MIGRATION_EXPORT', json.dumps({'bytes': destination.stat().st_size,
      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
      'files': len(selected)}, indent=2), flush=True)
files.download(str(destination))
