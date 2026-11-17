# Google Drive upload and restore

**[Open the owner's Google Drive folder](https://drive.google.com/drive/folders/1dOmyEh9O8oKWYxZb__fVAMuBv6avcKbZ?usp=sharing).** The owner supplied this link on November 18, 2026. Contents, permissions and upload completeness have not been independently verified; compare downloaded folders with the file manifest before relying on this as a complete backup.

The following folders stay on this laptop and are excluded from Git. Upload them to one Drive folder, preserving their names and internal paths, to retain the full research data and recovery record.

| Folder | Files | Size (GiB) | Purpose |
|---|---:|---:|---|
| `data/` | 38,401 | 19.254 | Training/test datasets, prepared splits, native masks and source archives |
| `outputs/` | 18,130 | 6.908 | Training histories, recovery ZIPs, verification records and saved app results |
| `model_cache/` | 1 | 0.020 | Optional cached ImageNet encoder weights; can be downloaded again |

The [CSV manifest](drive_file_manifest.csv) contains each file's relative path and exact byte size. It is an inventory, not a cryptographic integrity guarantee. Existing experiment verification records contain checkpoint/archive hashes.

## Restore

1. Clone this GitHub repository.
2. Download/extract the Drive folders directly into the cloned repository root, so `data/`, `outputs/` and optional `model_cache/` sit beside `app.py`.
3. Recreate `.venv` using the root README. Do not copy an environment from a different laptop.
4. Use `START_FINET.bat` for the app. The setup script `SETUP_NEW_LAPTOP.cmd` also runs a real-image check and therefore requires the restored CAMO sample under `data/TestDataset/CAMO/`.

The repository already includes `checkpoints/baseline/last.pth`, `checkpoints/boundary/last.pth` and `checkpoints/industrial/baseline/last.pth`. Their hashes are in [CHECKPOINT_MANIFEST.json](CHECKPOINT_MANIFEST.json). Keep these domain paths separate. The latest cloud industrial boundary checkpoint is not yet installed locally; download and verify the completed run before adding it.

## Files that exceed ordinary GitHub limits

GitHub blocks individual files larger than 100 MiB in ordinary Git. The current workspace contains the following oversized research artifacts (source archives and recovery copies remain useful for backup):

| File | Size (MiB) |
|---|---:|
| `data/downloads/mvtec_anomaly_detection.tar.xz` | 5021.1 |
| `data/downloads/COD-TestDataset.zip` | 1249.7 |
| `outputs/kaggle_migration/finet_kaggle_transfer.zip` | 1196.0 |
| `data/downloads/COD-TrainDataset.zip` | 1039.4 |
| `outputs/industrial_setup/industrial_version004_recovery.zip` | 364.1 |
| `outputs/kaggle_migration/kaggle_version005_recovery.zip` | 351.4 |
| `outputs/kaggle_migration/kaggle_version004_recovery.zip` | 347.4 |
| `outputs/kaggle_migration/kaggle_version003_recovery.zip` | 311.8 |
| `outputs/industrial_setup/industrial_version002_recovery.zip` | 224.2 |
| `outputs/industrial_setup/version002_transfer/industrial_recovery.zip` | 224.2 |
| `outputs/kaggle_migration/kaggle_version002_recovery.zip` | 210.5 |
| `outputs/kaggle_migration/kaggle_version001_recovery.zip` | 202.4 |
| `outputs/industrial_setup/gpu_smoke_recovery.zip` | 125.2 |

## Recreate instead of uploading

`.venv/`, `.venv-from-old-laptop-20261116/`, Python/test caches, `tmp/` and LaTeX build files are excluded. These contain installed dependencies or disposable authoring tools, not unique trained models. The outer `finet/START_FINET.bat` is a wrapper for this repository; the working root launcher is included here. Existing upstream Git history is retained by the push, so a separate `.git/` upload is unnecessary.

Datasets, outputs and optional cache files are kept outside Git to avoid a large repository even when individual files are below the hard limit. MVTec data remain subject to their dataset terms. The Drive link and its access settings are controlled by the owner.
