"""Package current project code for Colab without datasets, model weights or environments."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package(destination=None):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination) if destination else root / "outputs" / "finet_colab_training.zip"
    folders = ("Model", "utils", "src", "scripts", "configs", "notebooks", "tests", "docs")
    files = [root / name for name in ("train.py", "config.py", "train_baseline.py", "train_industrial.py", "evaluate_industrial.py", "predict.py", "app.py", "evaluate_models.py", "pytest.ini",
              "requirements.txt", "requirements-training.txt", "requirements-smoke-data.txt", "requirements-data.txt",
              "requirements-app.txt", "requirements-evaluation.txt", "requirements-lock-cpu.txt",
              "README.md", "PROGRESS.md", "PRODUCT.md", "DESIGN.md")]
    for folder in folders:
        files.extend(path for path in (root / folder).rglob("*") if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.exists() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pth", ".pt", ".ckpt"}:
                archive.write(path, path.relative_to(root).as_posix())
    return destination


if __name__ == "__main__":
    print(package())
