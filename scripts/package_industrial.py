"""Generate a source-only private Kaggle upload, excluding all datasets and weights."""

import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.package_colab import package
from src.training_data import file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    archive = package(root/'outputs/industrial_setup/finet_industrial_source.zip')
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        names = zipped.namelist()
        assert all(not name.startswith(('data/','outputs/','.venv/')) and not name.endswith(('.pth','.pt')) for name in names)
    report = {'archive':str(archive),'sha256':file_hash(archive),'bytes':archive.stat().st_size,
              'members':len(names),'contains_datasets_or_weights':False}
    (archive.parent/'source_package_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
