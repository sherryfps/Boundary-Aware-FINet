# FINet baseline training preparation — verification report

Date: **2026-11-09**, Asia/Calcutta. This follows the user's approval to train our own original FINet from an ImageNet-pretrained torchvision EfficientNet-B0 backbone. It supersedes the earlier requirement to obtain the authors' unavailable full segmentation checkpoint before any training preparation.

**Verified locally: genuine ImageNet encoder loading, full FINet forward/backward on real image/mask batches, finite gradients, Adam updates, checkpoint saving/reloading and a resumed update. No long training has been launched.** Colab GPU execution and a full baseline remain unverified.

## Backbone and architecture

`Model/EfficientNet.py` retains torchvision's `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` initialization when `pretrained=True`; `Model/FINet.py` uses it with `pretrained_backbone=True` for a new training run. No timm backbone or alternate network was introduced. The frequency-injection modules, HFA/LFA branches, asymmetric decoder, four heads and every model `forward` method remain unchanged.

Official torchvision file downloaded and loaded:

- URL: [efficientnet_b0_rwightman-7f5810bc.pth](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth).
- SHA-256: `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`; torchvision's download hash check passed.
- Cache: `C:\Users\HP.DESKTOP-LG6I0U2\.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth`.
- **All 352 retained encoder state tensors equal the official torchvision tensors exactly.**

| Component | Initialization / measured size |
|---|---|
| Retained EfficientNet-B0 encoder | ImageNet pretrained; **3,595,388 parameters** |
| Reductions, FFM/HFA/LFA, decoder and output heads | Randomly initialized; **147,600 parameters** |
| Complete FINet | **3,742,988 parameters**, 965 state tensors |

The classification head and final classification feature projection are not retained by the original FINet encoder. All retained learned parameters are trainable. Fixed DCT preprocessing adds no learned parameters.

Backbone feature channels verified from a forward pass: **`[16,24,40,112,320]`**, with spatial sizes `[192,96,48,24,12]` at 384 input. FINet consumes the last four scales, matching its channel reductions. Four full-model output maps are each `[1,1,384,384]` in the backbone-interface check.

Machine-readable evidence: [backbone.json](../outputs/training_verification/backbone.json).

## Real data used in the short test

No full dataset existed locally. Six real COD10K **training** photographs and original PNG object masks were fetched from the [nobg/COD10K mirror](https://huggingface.co/datasets/nobg/COD10K), whose card declares packaging from the original [SINet](https://github.com/DengPingFan/SINet) distribution. This is a **third-party mirror**, not the dataset authors' official host. The bytes have not been independently matched to an official archive.

- Mirror revision: `b3d03206c0db4c3cb6e03b78d12661fb74fbc3d5`.
- Original examples include `COD10K-CAM-1-Aquatic-1-BatFish-{1,3,7,8,9}` and `COD10K-CAM-1-Aquatic-10-LeafySeaDragon-415`.
- Original mask bytes were read from a Parquet row group with HTTP range requests. **JPEG-compressed viewer mask previews were rejected**, not thresholded into replacement ground truth.
- Transferred 98,891,137 bytes for the necessary Parquet metadata/column chunks; only the six decoded original pairs were retained.
- All six pairs have matching dimensions and genuine binary 0/255 PNG masks; file hashes and original names are in [provenance.json](../data/smoke_cod10k/provenance.json). The first photograph/mask pair was also inspected visually and aligned with the fish silhouette.
- Fixed split: **4 training / 2 validation** pairs, seed 123456. No final-test set was used.
- Manifest fingerprint: `d37a3835b42e8647b2bd95e6ac20deeda9eacbda25329055b651160535ab8e68`.

This tiny, mostly batfish sample is only for engineering checks. It is not representative enough for claims about accuracy, generalization or the boundary research hypothesis. Full COD10K-Train/CAMO-Train counts have not been verified locally.

## Actual training smoke results

Device: **CPU**, CUDA unavailable. Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.31, torch-dct 0.1.6, NumPy 2.5.4, Pillow 13.0.0. Optional PyArrow 26.0.1 was installed and used only to retrieve original smoke-data bytes.

Configuration: original EfficientNet-B0 FINet with channels `(8,24,32,64)`, 384 input, ImageNet-pretrained encoder, batch **2**, workers **0**, CPU threads **2**, seed **123456**, FP32, Adam lr **0.00026**, weight decay **0.00000004**, original structure loss on all four outputs. The schedule budget was 200 epochs, but the smoke executes **only two measured updates plus one resumed-verification update**.

| Check | Observed result |
|---|---|
| RGB batch | `[2,3,384,384]` |
| Mask batch | `[2,1,384,384]`, finite values in [0,1] |
| High / low frequency batches | Each `[2,96,48,48]`, original YCbCr/DCT/statistics/scaling |
| All four model outputs | Each `[2,1,384,384]` |
| Gradients | **581 trainable parameter tensors**, all received finite gradients |
| Optimizer update | Both encoder and output-head probes changed after each checked update |
| Checkpoint reload | Every full-model tensor reloaded exactly |
| Adam / scheduler reload | State tensors/position reloaded exactly |
| Reloaded evaluation logits | Equal with zero tolerance |
| Further optimization after reload | Passed, finite gradients and changed weights |

Actual losses, not fabricated logs:

| Update | Head 1 | Head 2 | Head 3 | Head 4 | Total original loss |
|---|---:|---:|---:|---:|---:|
| 1 | 1.542612 | 1.612055 | 1.571630 | 1.598699 | **6.324997** |
| 2 | 1.493284 | 1.534704 | 1.429403 | 1.474730 | **5.932121** |
| Resumed verification, same last batch | 1.493727 | 1.518280 | 1.423145 | 1.383331 | **5.818483** |

Totals use actual float32 summation; displayed rounded head values may not sum exactly. A small loss decrease over these few updates does not establish segmentation quality.

The separate two-image validation diagnostic after the resumed update measured four-head structure loss **5.793766** and native-resolution raw-probability MAE **0.489116**. This is a smoke diagnostic, not a benchmark score or trained-model performance claim. Validation took 2.7214 seconds in this run.

Measured CPU batch times, including data loading/augmentation and optimization: **23.6850 s** first step and **11.0384 s** second step. These exclude model initialization, checkpoint save/reload and validation. Two CPU steps do not support a reliable full-run or Colab GPU duration estimate. The notebook measures the actual assigned GPU instead. GPU peak memory has not been measured here.

## Checkpoint evidence

Saved locally, excluded from Git:

- [initialization.pth](../outputs/training_smoke_cpu/initialization.pth), **15,467,840 bytes**, tagged `initialization`; exact full-model starting weights before any smoke update.
- [smoke_checkpoint.pth](../outputs/training_smoke_cpu/smoke_checkpoint.pth), **45,900,269 bytes**, tagged `smoke_test`; model, Adam, scheduler, RNG states, settings and split identity after the two measured updates.
- Smoke checkpoint SHA-256: `47b5a4552a9f81a98aab51394d54ee29aacb7f18cc27794b35f8c4a13aeef06e`.
- [smoke_report.json](../outputs/training_smoke_cpu/smoke_report.json) and [steps.jsonl](../outputs/training_smoke_cpu/steps.jsonl) contain actual results.

The current runner additionally records environment/package versions and source SHA-256 fingerprints in newly saved checkpoints. That metadata addition was checked with safe-deserialization/resume tests; it does not alter training calculations.

**Neither saved smoke artifact is an application-ready segmentation checkpoint.** The inference adapter explicitly rejects initialization and engineering-smoke checkpoints. This rejection was also executed against the actual saved smoke checkpoint; it returned a clear error and created no output directory. No untrained/briefly-trained segmentation masks have been presented as successful detection results.

## Original behavior and tested compatibility fixes

1. **Original loss untouched.** `src/training.py` imports `train.structure_loss` directly and sums it for all four outputs. The original BCE `reduction='mean'` discrepancy is documented and preserved, including the effective cancellation of BCE weighting. No boundary term, AMP, gradient accumulation, clipping or alternative objective was introduced.
2. **Original DCT untouched.** Training subclasses the original dataset and reuses its augmentations, bilinear mask resize, RGB normalization, DCT, statistics and divide-by-7 step. The supervised resized mask may therefore be soft near boundaries, as upstream; binary auditing applies to the original GT files before resize.
3. **Statistics path fix.** `utils/dataloader_freq.py` resolves its bundled statistics relative to itself; no dependence on current working directory. Transform tensor parity is tested with zero tolerance.
4. **Scheduler fix and resume state.** Original `CosineDecay` failed at `max_epoch=1` because attributes were undefined. Added a stable one-epoch case and state save/load. Its 200-epoch formula is tested against the original expression and preserved.
5. **Pair and split checks.** The wrapper matches stems, audits actual data, hashes inputs and persists a disjoint fixed validation split. No silent skipping of mismatched GT or substitution of generated masks.
6. **BatchNorm constraint.** Training requires batch >=2 because the original pooled attention branches contain BatchNorm on 1×1 activations. The loader drops incomplete batches; it does not replace normalization layers.

Source checks confirmed that all original FINet `forward` methods and the executable ASTs of `train.py` and `utils/dct.py` are unchanged against upstream. A byte-only comparison initially differed due to checkout line endings; syntax-tree comparison and Git's unchanged-file list resolved that distinction.

## Files and commands

New training files:

- `src/training.py`, `src/training_data.py`, `train_baseline.py`.
- `scripts/verify_backbone.py`, `scripts/fetch_smoke_data.py`, `scripts/prepare_training_data.py`, `scripts/package_colab.py`.
- `configs/baseline.json`, `notebooks/baseline_training_colab.ipynb`.
- `requirements-training.txt`, `requirements-smoke-data.txt`, `tests/test_training.py`.
- `docs/COLAB_TRAINING.md`, `docs/TRAINING_VERIFICATION.md`.

Updated: `utils/dataloader_freq.py`, `utils/LRScheduler.py`, `src/finet_adapter.py`, `predict.py`, `tests/test_inference.py`, `requirements-lock-cpu.txt`, README/status/setup/checkpoint documentation. Model architecture files were inspected and unchanged in this follow-up task.

Key commands actually executed from the workspace:

```powershell
.\.venv\Scripts\python.exe scripts/verify_backbone.py
.\.venv\Scripts\python.exe -m pip install pyarrow --disable-pip-version-check
.\.venv\Scripts\python.exe scripts/fetch_smoke_data.py
.\.venv\Scripts\python.exe scripts/prepare_training_data.py --data-root data/smoke_cod10k --output outputs/splits/smoke_cod10k.json --validation-fraction 0.3333333333333333 --allow-small
.\.venv\Scripts\python.exe train_baseline.py --mode smoke --data-root data/smoke_cod10k --split-manifest outputs/splits/smoke_cod10k.json --output-dir outputs/training_smoke_cpu --device cpu --batch-size 2 --num-workers 0 --steps 2
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/training_verification/pytest.xml
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src scripts tests Model utils train_baseline.py predict.py
.\.venv\Scripts\python.exe scripts/package_colab.py
.\.venv\Scripts\python.exe predict.py --image data/smoke_cod10k/Imgs/COD10K-smoke-row0000.jpg --checkpoint outputs/training_smoke_cpu/smoke_checkpoint.pth --output-dir outputs/rejected_smoke_inference --device cpu
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
```

The CR-at-EOL option accommodates original CRLF blobs. Standard `git diff --check` flagged new CRLF lines in the existing CRLF data loader; no behavioral fix or broad source newline rewrite was needed.

Tests: **63 passed, 2 skipped in 19.81 seconds** in the final full-suite run. Both skipped cases require a fully trained FINet segmentation checkpoint; they do not indicate failure to load the now-verified ImageNet backbone. Actual evidence: [JUnit XML](../outputs/training_verification/pytest.xml). An earlier run had 62 passed before the malformed-provenance test was added. The post-metadata checkpoint/RNG-resume and notebook compilation checks were also rerun separately.

Tests cover original-loss identity and gradients, preservation of its actual BCE behavior, scheduler formula and one-epoch fix, exact Adam/RNG/scheduler continuation on a small unit model, dataset pairing/split/hash checks, preprocessing parity, full-run opt-in/GPU guards, smoke-artifact rejection and notebook Python syntax/default-disabled training. The real FINet backward/checkpoint evidence is the separate actual real-data smoke run above, not the small synthetic unit-model test.

## Colab and full baseline: prepared, not executed

Use [baseline_training_colab.ipynb](../notebooks/baseline_training_colab.ipynb), the [ready-to-upload code ZIP](../outputs/finet_colab_training.zip) and [COLAB_TRAINING.md](COLAB_TRAINING.md). ZIP integrity was checked: 47 members, required training/notebook/statistics/configuration files present, no dataset, checkpoint, environment or bytecode payloads. The notebook detects CUDA/GPU memory, installs dependencies, mounts Drive, audits the actual official 4,040 training pairs, creates a fixed training-only holdout, performs measured GPU smoke checks, estimates resources from that run and prints settings before the disabled full-training cell.

Proposed GPU starting settings are batch **4**, workers **2**, FP32, 384 input, original Adam/loss/cosine schedule. **Batch 4 on a Colab GPU has not been tested in this session.** The wrapper explicitly enables cuDNN with deterministic flag true/benchmark false; original `train.py` disabled cuDNN. This runtime change and smaller batch are recorded, not silently described as identical paper reproduction.

Full mode requires an explicit flag, actual CUDA availability, the audited full dataset, a passed matching GPU smoke report, and the exact pre-smoke full-model initialization. It records training/validation logs, saves recoverable epoch-boundary `last.pth`, selects `best.pth` by full validation MAE, and saves diagnostic validation outputs. Source/environment fingerprints are saved for reproducibility. No GPU/full-run duration or memory requirement is invented.

## What remains blocked and where we stop

- **No available local CUDA GPU.** GPU availability, GPU batch 4, Colab dependency compatibility and GPU memory/timing must be verified inside the user's actual Colab runtime.
- **No audited full COD10K-Train + CAMO-Train collection locally.** Official download and placement instructions are provided; the six mirrored pairs cannot replace the full dataset.
- **No trained-quality full baseline checkpoint yet.** Only real smoke artifacts exist. Unseen-image segmentation verification, application integration and final metrics remain later work.
- **No boundary experiment started.** Later A/B runs must share the same saved initial full-model weights, split, augmentation, optimizer and runtime settings; only the boundary term should differ. No improvement is assumed.
- A separate fresh environment, long-run stability, Drive checkpoint I/O and CUDA epoch-boundary recovery have not been executed. Local CPU checkpoint/RNG/scheduler tests do not prove those cloud conditions.

The requested pretraining/training-engineering checks passed locally. **Stop here before long training.** Open the supplied Colab notebook and follow the guide to obtain the real GPU smoke evidence before deciding to launch the full baseline.
