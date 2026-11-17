# Official FINet checkpoint setup

**Current route:** the user has authorized training our own original FINet from torchvision ImageNet-pretrained EfficientNet-B0. That backbone download and real-data CPU training/checkpoint smoke test passed; see [training verification](TRAINING_VERIFICATION.md) and [Colab instructions](COLAB_TRAINING.md). A future selected project-trained `best.pth` is accepted by `predict.py`; the unavailable authors' full checkpoint is not required for that training route. Initialization and engineering-smoke checkpoints are explicitly rejected for application inference.

## Current blocker — 2026-11-09

The full **FINet (EfficientNet-B0)** checkpoint is required for genuine segmentation. No such file is present, and the user confirmed they do not have it. Both links in the [official README](https://github.com/CRRCOO/FINet#3-pre-computed-maps-and-trained-weights) currently fail:

| Official source | Exact link / code | Observed result |
|---|---|---|
| Google Drive | [FINet EfficientNet-B0 weights](https://drive.google.com/file/d/1ePSiaW14I_NecnBAXE7O_-ApIuFFduuI/view?usp=sharing) | HTTP 404; direct-download endpoint also 404 |
| Baidu Drive | [FINet EfficientNet-B0 weights](https://pan.baidu.com/s/1JNtzpoVZbVjF-15rFK6WuQ), code **`hdrs`** | Page title `百度网盘-链接不存在` (“link does not exist”); no file returned |

This is not resolved by installing ImageNet backbone weights. Pre-computed prediction maps in the README are also not trained model weights. Do not use TinyNet-A weights with the EfficientNet-B0 architecture.

## Manual recovery and exact placement

1. Open the official repository README and use **Trained Weights → FINet (EfficientNet-B0)**. The two direct links above are the original links, not mirrors.
2. If a link is restored, download the actual checkpoint using the service's Download button; for Baidu use extraction code `hdrs`. If the download is an archive, extract the full model `.pth` file. Do not rename an HTML error page or a ZIP to `.pth`.
3. If both remain unavailable, ask the FINet repository maintainer/authors for a replacement download of the **complete EfficientNet-B0 FINet state_dict**, with channels `(8, 24, 32, 64)`. An alternate download needs author provenance before it can be treated as official. No message has been sent on your behalf.
4. Place the actual full checkpoint at:

   ```text
   C:\Users\HP.DESKTOP-LG6I0U2\Desktop\finet\boundary-aware-finet\checkpoints\FINet.pth
   ```

   The filename can differ if you pass its actual path to `--checkpoint`.
5. Place a real camouflaged-object photograph at `data/samples/camouflaged.jpg`, or pass another real JPEG/PNG path. No artificial prediction or paper-results montage should serve as successful real-image demo evidence.
6. From the project root in PowerShell, run:

   ```powershell
   .\.venv\Scripts\python.exe scripts/smoke_test.py --image data/samples/camouflaged.jpg --checkpoint checkpoints/FINet.pth --output-dir outputs/verified --device cpu
   ```

7. The command must succeed and produce `probability.png`, `binary.png`, `overlay.png`, `probability.npy`, `prediction.json` and `smoke_report.json`. Inspect the overlay against the photograph; the smoke test checks pixel alignment and numerical integrity, not segmentation accuracy.
8. Enable the genuine integration tests:

   ```powershell
   $env:FINET_CHECKPOINT = (Resolve-Path checkpoints/FINet.pth).Path
   $env:FINET_TEST_IMAGE = (Resolve-Path data/samples/camouflaged.jpg).Path
   .\.venv\Scripts\python.exe -m pytest -q
   ```

## What the loader verifies

The inspected architecture contains **3,742,988 learned parameters and 965 state-dict tensors** (the tensor count also includes BatchNorm buffers). The adapter explicitly checks every expected key and then performs strict shape-compatible loading. It will reject standalone encoder weights, missing tensors, extra tensors, wrong model channels, NaN/infinite tensors, HTML pages and unsafe object pickles. It accepts an ordinary tensor state dict, conventional `state_dict`/`model_state_dict` wrappers, and a uniform DataParallel `module.` prefix.

`prediction.json` records the SHA-256 of the file actually loaded, exact model name, device, parameter count and measured timing. The authors do not provide a reference checksum in the inspected README, so a locally computed hash provides reproducibility, not independent proof of provenance.

Original model and checkpoint redistribution permissions remain unspecified: the inspected upstream commit has no explicit license. Keep weights out of Git and preserve author attribution.
