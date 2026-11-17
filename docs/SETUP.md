# Setup and inference — Phase 1

For the current own-baseline training route, use [COLAB_TRAINING.md](COLAB_TRAINING.md) and [baseline_training_colab.ipynb](../notebooks/baseline_training_colab.ipynb). ImageNet loading and real-batch CPU training/checkpoint tests have now passed. Full baseline/GPU training remains unexecuted; a selected project-trained `best.pth` will work with the inference commands below. The historical official-checkpoint route is retained for reference.

## This workspace: ready-to-run environment

A local Python environment was created at `.venv` using the Codex bundled Python **3.12.14** because neither `python` nor `py` was initially available on PATH. Use its executable directly; activation is optional.

```powershell
Set-Location 'C:\Users\HP.DESKTOP-LG6I0U2\Desktop\finet\boundary-aware-finet'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe predict.py --help
```

The local machine has Intel HD Graphics 630 and no detected CUDA device; this environment uses CPU PyTorch. The full original FINet checkpoint remains unavailable. Tests of untrained architecture and synthetic unit inputs do not establish pretrained inference success. See [checkpoint recovery](CHECKPOINTS.md).

## Fresh Windows installation

Install 64-bit Python 3.12 from [python.org](https://www.python.org/downloads/windows/) if necessary. Open PowerShell in the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements-lock-cpu.txt` records the exact tested local CPU packages, including transitive dependencies. `requirements.txt` lists direct dependencies and minimum API requirements for other environments. A fresh Windows installation following the lock has not been independently rebuilt; the lock matches the successfully installed/tested environment.

To resolve a new compatible package combination instead of reproducing the lock:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an NVIDIA GPU environment use a matching torch/torchvision installation from the [official PyTorch installer](https://pytorch.org/get-started/locally/) appropriate to the driver. Do not install the CPU lock over an existing CUDA environment. This project's CUDA execution has not been tested on local hardware.

Optional original evaluation/profiling/plotting packages are listed in `requirements-optional.txt`. Their workflows and compatibility are not verified by Phase 1; Gradio is deliberately deferred to Phase 2.

## Single-image prediction after obtaining weights

Create `checkpoints/` and `data/samples/` if absent. Obtain the official full FINet EfficientNet-B0 checkpoint following `docs/CHECKPOINTS.md`. Place a real photograph at `data/samples/camouflaged.jpg` (or substitute its actual path).

```powershell
.\.venv\Scripts\python.exe predict.py --image data/samples/camouflaged.jpg --checkpoint checkpoints/FINet.pth --output-dir outputs/example --device cpu --threshold 0.5
```

Use quoted paths for names with spaces. `--device auto` chooses available CUDA, otherwise CPU. `--device cuda:0` explicitly requests the first CUDA device and reports an error if unavailable. A missing checkpoint is an error; there is no untrained fallback.

Outputs:

| File | Meaning |
|---|---|
| `probability.png` | 8-bit grayscale sigmoid probability, matching upstream truncation |
| `probability.npy` | Full float32 probabilities in [0,1], preserving precision |
| `binary.png` | 0/255 mask; foreground is probability **>= threshold** |
| `overlay.png` | Original RGB image with green foreground tint, opacity 0.45 |
| `prediction.json` | Checkpoint hash, model/device, timings, dimensions, probability range and output paths |

All masks/overlays have the original decoded image dimensions. For exact parity with the source loader, EXIF rotation is not applied; inputs should have their pixels in the intended display orientation. Each output directory holds one prediction; use a different directory to retain previous runs. The CLI refuses to overwrite the input image.

The threshold changes binary masks and overlays only. Run again with another `--threshold` to produce another visualization; the neural-network weights and probability map remain unchanged. Timing is a single synchronized forward-pass measurement without warm-up, not a benchmark average. `processing_ms` includes decoding, preprocessing, device transfer and restoration; it excludes model loading and saving. No accuracy metrics are computed without ground truth.

Validate actual pretrained output files:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --image data/samples/camouflaged.jpg --checkpoint checkpoints/FINet.pth --output-dir outputs/smoke --device auto
```

This adds `smoke_report.json` after checking shapes, ranges and saved artifact alignment. It does not assess segmentation quality against unknown ground truth.

## Google Colab inference instructions (not executed here)

Use a private Colab notebook and choose a free GPU runtime if available. No training is performed in these instructions. Keep a copy of the **current workspace additions**, because the upstream repository alone does not contain `predict.py` or `src/`.

In local PowerShell, prepare a small code archive that excludes environments, datasets and weights:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
Compress-Archive -Path Model,utils,src,scripts,predict.py,requirements.txt -DestinationPath outputs/finet-phase1.zip -Force
```

Upload that ZIP with Colab's Files sidebar. Upload the authentic `FINet.pth` and real `camouflaged.jpg` separately, or use your Google Drive. Then run these notebook cells:

```python
!unzip -q /content/finet-phase1.zip -d /content/boundary-aware-finet
%cd /content/boundary-aware-finet
!pip install -r requirements.txt
```

```python
import torch, torchvision
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

If your files are in Google Drive, mount it explicitly in Colab and substitute their real paths in the command below:

```python
from google.colab import drive
drive.mount('/content/drive')
```

For files uploaded directly to `/content`:

```python
!python scripts/smoke_test.py --image /content/camouflaged.jpg --checkpoint /content/FINet.pth --output-dir /content/finet_outputs --device auto
```

```python
from IPython.display import display, Image
display(Image('/content/finet_outputs/overlay.png'))
```

Download results from the Files sidebar or copy them to Drive before the Colab session ends. CUDA/Colab execution remains unverified until it is actually run with authentic weights. Do not apply the Windows CPU lock to Colab's GPU environment.

## Troubleshooting

- **Checkpoint not found:** follow `docs/CHECKPOINTS.md`; the published links currently fail. An ImageNet backbone is insufficient.
- **“Cannot safely read checkpoint”:** confirm that the download is a real PyTorch tensor state dict, not an HTML error page/archive. Do not disable safe loading to force an unknown file to load.
- **Incomplete/wrong checkpoint:** use the EfficientNet-B0 FINet full state, not TinyNet-A, encoder-only weights, or default-channel model weights.
- **Python command not found:** use the existing `.\.venv\Scripts\python.exe` here, or install Python for a fresh machine. Activating a venv is unnecessary.
- **Missing/incompatible dependencies:** install the local CPU lock, or a platform-matched torch/torchvision pair followed by `requirements.txt`. Use `python -m pip check` in that same environment.
- **CUDA unavailable:** use `--device cpu` or `auto`.
- **Out of memory:** close other applications; use one image and CPU when GPU memory is insufficient. The resolution remains 384 to preserve the official pipeline.
- **Invalid image:** use a real JPEG/PNG file; changing an unsupported file's extension does not convert its contents.
- **Tests report two skipped cases:** authentic checkpoint/image integration tests require `FINET_CHECKPOINT` and `FINET_TEST_IMAGE`; skips are not successful pretrained verification.
