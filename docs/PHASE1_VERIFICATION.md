# Phase 0/1 verification report

Historical report from before the own-training route was authorized. The latest verified backbone/real-batch training results and current limitations are in [TRAINING_VERIFICATION.md](TRAINING_VERIFICATION.md). The unavailable authors' full checkpoint is no longer required to begin our own baseline training.

Date: **2026-11-09**, local timezone Asia/Calcutta.

**Phase 0 is verified. Phase 1 implementation passes the available checks, but Phase 1 itself is NOT verified: authentic pretrained weights are unavailable.** There has been no genuine pretrained segmentation run, and no model predictions, segmentation-quality scores or pretrained inference latency are reported.

## 1. Phase names and outcome

- **Phase 0 — Understand the existing repository: complete.** Inspected all relevant original sources, documented the actual model/preprocessing/loss, identified official weights and dataset sources, checked licensing, and prepared the dependency and implementation plan.
- **Phase 1 — Working original FINet inference: implemented and unit-tested; externally blocked.** The CLI, adapter and smoke test are ready for verification once the complete official checkpoint is recovered. A working pretrained application cannot yet be claimed.

No Phase 2, boundary loss or GPU training work has started. No remote push, commit, paid service or message to the authors was performed.

## 2. Files created or modified

Created:

| File | Purpose |
|---|---|
| `.gitignore` | Exclude environment, datasets, checkpoints and generated outputs |
| `requirements.txt` | Direct Phase 1 dependencies with required API minimums |
| `requirements-lock-cpu.txt` | Exact tested Python 3.12 CPU environment |
| `requirements-optional.txt` | Deferred original evaluation/profiling/plotting packages |
| `pytest.ini` | Test discovery and genuine-pretrained-test marker |
| `src/__init__.py` | Added helper package |
| `src/preprocessing.py` | Exact RGB/DCT preprocessing without a GT-mask requirement |
| `src/finet_adapter.py` | Strict full checkpoint loading, device selection and timed inference |
| `src/postprocessing.py` | Original-size probability restoration, binary mask, overlay and saving |
| `predict.py` | Single-image CLI and understandable error reporting |
| `scripts/smoke_test.py` | Real-checkpoint integration and saved-artifact checks |
| `tests/conftest.py` | Explicit synthetic unit-test fixtures and limited test threads |
| `tests/test_preprocessing.py` | Exact reference-loader parity, color modes and input validation |
| `tests/test_postprocessing.py` | Resize/sigmoid ordering, thresholding, dimensions and saved artifact integrity |
| `tests/test_inference.py` | Architecture check, checkpoint errors, strict tensor coverage, device selection and gated pretrained tests |
| `tests/test_cli.py` | CLI errors and input-overwrite prevention |
| `docs/ARCHITECTURE.md` | Phase 0 findings and source attribution |
| `docs/CHECKPOINTS.md` | Official links, access evidence and exact recovery/placement instructions |
| `docs/SETUP.md` | Windows/Colab setup, commands and troubleshooting |
| `docs/PHASE1_VERIFICATION.md` | This report |
| `PROGRESS.md` | Full phase checklist, completed tasks and blockers |

Modified:

- `Model/EfficientNet.py`: optional `pretrained=True` constructor argument controls the existing torchvision ImageNet initializer.
- `Model/FINet.py`: optional `pretrained_backbone=True` argument forwards to the original backbones. Original default behavior is preserved. No layers, forward operations or state-dict keys are changed.
- `README.md`: added project status and documentation/command links above the retained original README.

Generated locally, excluded from Git: `.venv/` and `outputs/verification/pytest.xml`. The XML is actual test evidence. **No segmentation output PNGs were generated in `outputs/`.** Synthetic postprocessing artifacts were created only inside pytest's temporary directories and are not presented as model results.

## 3. What was implemented

The wrapper constructs the original `FINet(backbone='efficientb0', channels=(8,24,32,64))`, disables redundant ImageNet initialization, and requires the full trained state before exposing inference. It uses safe tensor-only deserialization (`weights_only=True`), explicitly checks complete key coverage (including BatchNorm tracking buffers), then strictly checks shapes. It records the loaded file's SHA-256. It accepts conventional state-dict wrappers and a uniform DataParallel prefix without silently dropping incompatible keys.

Preprocessing reuses the original YCbCr/block-DCT implementation, bundled frequency normalization and factor of 7. It produces RGB `[1,3,384,384]` and high/low `[1,96,48,48]` tensors. The first model output is resized as logits with `align_corners=True` and then passed through sigmoid, matching original inference.

The prediction path saves original-dimension grayscale/binary/overlay PNGs, full float probabilities and metadata. Thresholding affects visualization only. Timing separates the synchronized forward pass from decoding/preprocessing/restoration and excludes model loading/saving. CPU and available CUDA selection are supported; only CPU architecture/unit checks have been executed here.

## 4. Significant commands actually executed

Working directory: `C:\Users\HP.DESKTOP-LG6I0U2\Desktop\finet\boundary-aware-finet`.

Inspection used `rg --files`, `Get-Content` on the original sources, `git status --short`, `git remote -v`, `git log -1 --format='%H %s'`, and `git ls-files '*LICENSE*' '*COPYING*'`. The source commit is `9e42b6bf607a2c89823f5f88dd8d38069039102e`. No explicit license was found.

Environment commands:

```powershell
& 'C:\Users\HP.DESKTOP-LG6I0U2\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt --disable-pip-version-check
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m pip freeze
& '.\.venv\Scripts\python.exe' -m pip install --dry-run -r requirements-lock-cpu.txt --disable-pip-version-check
```

Verification commands:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m pytest -q --junitxml=outputs/verification/pytest.xml
& '.\.venv\Scripts\python.exe' -m compileall -q Model src scripts tests predict.py
& '.\.venv\Scripts\python.exe' predict.py --help
& '.\.venv\Scripts\python.exe' scripts/smoke_test.py --image imgs/vis.png --checkpoint checkpoints/FINet.pth --device cpu
& '.\.venv\Scripts\python.exe' -S predict.py --image imgs/vis.png --checkpoint checkpoints/FINet.pth
git diff --check
git check-ignore .venv/Scripts/python.exe checkpoints/FINet.pth data/samples/camouflaged.jpg outputs/verification/pytest.xml
```

The `imgs/vis.png` command was **only a negative smoke-test invocation** to check the missing-checkpoint error. It did not run a model or establish successful real-photograph inference. The `-S` command deliberately disabled installed packages to exercise the missing-dependency message.

Checkpoint access was checked with `Invoke-WebRequest` against the exact official URLs and direct Drive download endpoint; see the recorded results below. Additional Python inspection counted model parameters/state tensors and validated the bundled frequency statistics; JUnit XML was parsed to confirm totals.

## 5. Tests executed and results

Final run:

```text
49 passed, 2 skipped in 4.53s
51 tests; 0 failures; 0 errors
```

The initial run had 48 passed and 2 skipped. One additional test was added for missing BatchNorm tracking buffers; the final run above includes it and the stronger full-state validation.

| Check | Actual result |
|---|---|
| Model, torchvision, timm and DCT imports | Passed in the installed CPU environment |
| Preprocessing vs actual upstream `TestDataset` | Passed, RGB/high/low tensors equal with **zero tolerance** |
| Statistics lookup outside project cwd | Passed |
| RGB/grayscale/RGBA and JPG/JPEG/PNG handling | Passed |
| Invalid files, formats, dimensions and thresholds | Passed |
| Original architecture forward at 384 | Passed using explicitly **untrained** weights and synthetic unit input; four finite `[1,1,384,384]` outputs |
| Offline initializer switch preserves tensor layout | Passed; default still requests original ImageNet initialization |
| Incomplete/HTML/nonfinite checkpoint rejection | Passed |
| Missing BatchNorm buffer rejection | Passed |
| CPU auto-selection / unavailable CUDA error | Passed using controlled availability tests |
| Resize logits before sigmoid | Passed against the original operations; wrong-order interpolation differs |
| Binary/overlay alignment and threshold behavior | Passed on known unit fixtures |
| Saved PNG dimensions/values and NumPy precision | Passed on known temporary unit fixtures |
| CLI input-overwrite protection | Passed |
| **Authentic full checkpoint tensor equality** | **Skipped**: `FINET_CHECKPOINT` not supplied |
| **Genuine pretrained image forward pass** | **Skipped**: `FINET_CHECKPOINT` and `FINET_TEST_IMAGE` not supplied |
| `pip check` | `No broken requirements found.` |
| Lock-file installation dry run | All recorded packages already satisfied |
| Compilation and `git diff --check` | Passed, exit 0 |
| CLI help | Passed, exit 0 |
| Missing-checkpoint smoke invocation | Expected exit 1, clear “FINet checkpoint not found” message; no outputs |
| Missing-dependency invocation (`-S`) | Expected exit 1, clear “No module named 'numpy'” / install-requirements message |
| Git ignore checks | Environment, checkpoint, dataset and generated evidence paths ignored |

The 4.53 seconds above is **test-suite runtime**, not model inference latency.

## 6. Evidence and installed environment

- Actual machine-readable test results: [JUnit XML](../outputs/verification/pytest.xml).
- Python **3.12.14**, PyTorch **2.15.0+cpu**, torchvision **0.29.2+cpu**, timm **1.0.31**, torch-dct **0.1.6**, NumPy **2.5.4**, Pillow **13.0.0**, pytest **9.2.0**.
- Full installed versions: `requirements-lock-cpu.txt`, captured from the successful environment and checked with pip's dry run.
- Architecture measured directly from the original-model constructor: **3,742,988 parameters**, **965 state-dict tensors**. These are architecture measurements, not evidence of pretrained weight loading.
- Original preprocessing and postprocessing are directly covered by reference-operation tests.
- No fabricated checkpoint, trained log, pretrained result, accuracy metric or paper-FPS claim is included.

## 7. Remaining issues and exact blocker

The official README's [Google Drive EfficientNet-B0 checkpoint](https://drive.google.com/file/d/1ePSiaW14I_NecnBAXE7O_-ApIuFFduuI/view?usp=sharing) returned **HTTP 404** both for the view URL and the direct-download endpoint. Its [Baidu alternative](https://pan.baidu.com/s/1JNtzpoVZbVjF-15rFK6WuQ), code `hdrs`, returned a page titled **“link does not exist.”** No checkpoint bytes were obtained. The user confirmed they have no local copy.

A replacement full checkpoint with author provenance is needed. The exact recovery and placement instructions are in [CHECKPOINTS.md](CHECKPOINTS.md). A real input photograph is also needed for the successful demo/smoke test; no benchmark datasets were downloaded. Dataset counts and segmentation accuracy have not been evaluated.

CUDA, Colab, real checkpoint compatibility, pretrained segmentation quality, actual pretrained latency and a separately rebuilt fresh environment remain unverified. Optional original evaluation/profiling dependencies were not installed/tested. Upstream source and checkpoint redistribution licensing is unspecified.

## 8. Exact instructions to run

Already runnable:

```powershell
Set-Location 'C:\Users\HP.DESKTOP-LG6I0U2\Desktop\finet\boundary-aware-finet'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe predict.py --help
```

After obtaining the original full checkpoint and placing a real photograph as described in `docs/CHECKPOINTS.md`:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --image data/samples/camouflaged.jpg --checkpoint checkpoints/FINet.pth --output-dir outputs/verified --device cpu
$env:FINET_CHECKPOINT = (Resolve-Path checkpoints/FINet.pth).Path
$env:FINET_TEST_IMAGE = (Resolve-Path data/samples/camouflaged.jpg).Path
.\.venv\Scripts\python.exe -m pytest -q
```

Success requires the smoke report, real saved masks/overlay, correct dimensions/ranges, manual overlay inspection and the two pretrained tests passing. `docs/SETUP.md` also documents clean Windows installation and Colab inference; the latter instructions have not been executed.

## 9. Whether the phases are verified

**Phase 0: yes. Phase 1: no — awaiting authentic pretrained checkpoint and real-image verification.** Unit-test success does not satisfy the Phase 1 completion criterion. This is a blocker report, not a pretrained-inference completion report.

## 10. Next step and phase gate

Next work remains in **Phase 1**: recover genuine original weights, run the real-image smoke test and inspect saved results. Update this report and `PROGRESS.md` with actual evidence only after that succeeds.

After verified Phase 1 and the user's approval, Phase 2 will connect this same model pipeline to a Gradio image-upload application. No Phase 2 work is authorized by the current task.
