# FINet industrial defect segmentation

This is a separate academic experiment for factory quality inspection. Subtle scratches, cracks and contamination can resemble normal surface texture; pixel masks help inspect their location and boundary. This prototype makes no certified production-inspection or product acceptance claim.

## Dataset and usage

Original source: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019. Data is licensed **CC BY-NC-SA 4.0**, for non-commercial use. Preserve attribution and share-alike obligations when distributing derived data. Keep raw data private in this project.

The supplied archive downloaded successfully on November 13, 2026: **5,264,982,680 bytes**, **6,880 tar entries**, SHA-256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`. Full xz/tar decoding passed. No publisher SHA-256 was supplied for independent authenticity comparison. `scripts/download_mvtec.py` records the source URL, size, hash and extraction verification in `data/MVTecAD_original/archive_verification.json`. Kaggle checks the archive against that recorded hash before extracting it.

## Explicit custom supervised protocol

The official training split contains normal images; defect masks belong to abnormal test images. Training supervised FINet on those annotations creates a **custom supervised MVTec split**, `custom_supervised_mvtec_v1`. Its results are not official anomaly-benchmark performance.

`scripts/prepare_mvtec.py` audits all 15 original categories and all 5,354 images, verifies exact image-mask geometry and pairs abnormal images with `ground_truth/<defect>/<stem>_mask.png`. Good images receive genuinely empty masks. Masks are converted to binary 0/255. The original archive and extracted source files are retained.

The split uses seed **123456** and approximately **70/15/15 train/validation/test**, stratified by category, original split and defect type. Rounding is recorded in the actual manifest. Decoded pixel-identical images are kept together. Strata with fewer than three independent examples are held out entirely. Cross-category or inconsistent-label duplicates require review. The manifest records exact image IDs, original paths and splits, category, defect type, byte hashes, decoded-pixel hashes, native dimensions and foreground pixels before/after nearest-neighbor resizing to 384. Train/validation/test IDs and hashes are checked for leakage. No held-out test image is used for training, checkpoint selection or parameter tuning.

The original dataset remains available for a **separate, freshly initialized normal-only experiment** using the official protocol. A custom-supervised checkpoint must never be reused to claim that independent official protocol.

## Model and training

FINet architecture, frequency injection and decoder are unchanged. The backbone uses the existing torchvision ImageNet-pretrained EfficientNet-B0 initialization. Segmentation/frequency weights begin randomly. All model parameters remain trainable. Industrial weights and logs live under `FINetIndustrial/`, never in the camouflage experiment directories.

Both industrial variants use the same complete initial state, seed, exact split, sampler, optimizer and schedule. Configuration files are `configs/industrial_baseline.json` and `configs/industrial_boundary.json`.

| Setting | Registered industrial configuration |
|---|---|
| Image and batch | 384×384 RGB, batch 4, FP32 |
| Optimizer | Adam, initial LR 0.00026, weight decay 0.00000004 |
| Schedule | Original cosine schedule, minimum LR 0.0000001, 200 epochs |
| Frequency preprocessing | Original YCbCr block DCT, normalization statistics and high/low coefficient selection |
| Baseline objective | Exact upstream structure loss summed over four outputs |
| Boundary objective | Same structure sum + 0.1 × existing differentiable boundary BCE+Dice sum |
| Industrial augmentation | Horizontal flip only; nearest-neighbor mask resize |
| Sampling | Replacement sampling with equal total probability per category × normal/abnormal stratum |
| Selection | Highest validation Dice averaged equally across categories' defective images, at threshold 0.5 |

Industrial augmentation changes are explicit and do not affect COD training. Pepper noise would create false defects in normal masks; cropping could remove small defects. Original RGB normalization and DCT processing remain intact. Class imbalance is addressed by matched sampling, without changing the original structure loss. Report any defects that vanish at 384; do not claim fine-defect preservation without measuring it. Higher-resolution/tiling experiments would require a new documented protocol.

## Verification and recovery

The November-14 continuation independently verified saved Version 2 at **96/200 baseline epochs / 89,472 updates**, next epoch **97**, LR **0.00013722662274203082**. Its 235,085,456-byte recovery archive passed all CRCs; strict full-state restoration and every canonical target hash passed. A separate private recovery dataset is attached alongside the complete independent **FINet MVTec Prepared Cache** saved output, preserving the exact split and avoiding raw extraction. Corrected saved **Version 4 (`369287259`) genuinely resumed epoch 97**, with step **90,000**, finite gradients and the correct LR verified. Fresh GPU recovery probe passed and **42 protocol checks passed**. It uses an 8-hour session budget; the measured 104 remaining epochs represent about 7.38 training hours before setup. Runtime limits change no training settings. See `PROGRESS.md` for actual execution snapshots; Version 3 stopped in preflight and was not a successful training resume.

`scripts/probe_industrial_resume.py` performs an isolated real-batch test from the trained recovery checkpoint: all-four-output structure loss, finite gradients, encoder/head updates, exact model/Adam/scheduler/RNG restoration, engineering save/reload and exact four-output equality. It verifies that the official checkpoint's byte hash stays unchanged and never credits its test update as an epoch. The notebook embeds the tested helper with a SHA guard and executes it on the actual GPU before resuming. The local CPU checkpoint-96 test passed; fresh GPU results must be recorded separately.

The actual T4 checkpoint-96 probe also passed: structure loss **0.3351786733**, all **581 gradients finite**, encoder/head changes, exact CUDA-inclusive restoration and exact four-output reload, original checkpoint unchanged. It credits no official epoch. The active Version-4 training source is frozen. A concurrent integrity verifier is prepared locally for later sessions to reduce remote-storage latency; tests retain all hash/path checks and RNG state, while architecture/transforms/loss remain unchanged. This I/O update has not been applied to the running job, and local verification timing is not a Kaggle benchmark. Before any subsequent launch, independently verify its latest completed recovery output and attach that newer state; do not reuse only the older epoch-96 recovery dataset after more epochs have finished.

`train_industrial.py --mode smoke` performs a real normal/defective image check, measured training batches, all-four-output losses, finite gradients, encoder/head optimizer updates, safe checkpoint saving, exact model/Adam/scheduler reload, exact four-output equality, a resumed optimizer update and a validation diagnostic. Smoke checkpoints are explicitly rejected by the application and evaluator.

Example local engineering test after data preparation:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_mvtec.py
.\.venv\Scripts\python.exe train_industrial.py --config configs/industrial_baseline.json --device cpu --num-workers 0 --output-dir outputs/industrial_cpu_smoke/baseline
```

The Kaggle notebook is `notebooks/industrial_training_kaggle.ipynb`. `scripts/package_industrial.py` creates a source-only upload in `outputs/industrial_setup/`. The private MVTec input is imported from the supplied public archive URL; the source package is a separate private input. Install additional packages without replacing the GPU torch/torchvision pair. Record the actual environment in every run.

Run the GPU smoke cell before enabling its long-training cell. Duration estimates come from measured GPU steps, with validation/startup/checkpoint overhead excluded explicitly. The bounded supervisor `scripts/run_industrial_kaggle.py` saves model, Adam, scheduler, RNG, exact split, settings and training logs every completed epoch. It writes `industrial_recovery.zip` at session exit and on exceptions when possible. Save & Run All commits output on Kaggle's servers; interactive persistence alone is not a backup. Download saved recovery outputs and attach the latest saved notebook version as an input before the next session.

The recovery selector checks domain, full state, completed epochs, optimizer finiteness and archive CRC, selects the highest completed A/B state, rejects conflicting equal-epoch states and refuses to roll back a newer working state. Resume validates configuration and split fingerprint before the next optimizer update. A hard platform termination may discard an unfinished epoch; only completed saved epochs are credited. No free-quota bypass, paid compute or unattended account verification is part of this workflow.

Kaggle's [documented saved-run limitation](https://www.kaggle.com/docs/packages) automatically advances notebook-output inputs to their latest version. Keeping an older self-referencing output selected in the editor therefore does not reliably preserve its prepared cache. This was caught before any Version-3 update. The complete cache was copied from original Version 1 in an interactive CPU session, hash-verified, and saved with output in the separate **FINet MVTec Prepared Cache** notebook, Version 1 (`369285852`). Keep that independent cache notebook frozen and attach its complete output alongside the separate recovery dataset; do not use the training notebook's changing output as the only cache input. Corrected training Version 4 (`369287259`) uses this arrangement. All 10,708 image/mask files and the exact split are retained; no training epoch was repeated or credited by the CPU copy.

The user requested **one final baseline checkpoint after all 200 epochs** and explicitly authorized temporary recovery saves between sessions. Internal `last.pth` is overwritten each completed epoch; validation selection state, logs and session archives remain recovery/research material. They are not final model deliveries. Verify epoch 200 and the complete saved state before exporting the final baseline file; never relabel a smoke or partial-training state as complete. Preserve the existing camouflage artifacts.

Recovery accepts both intact ZIPs and Kaggle-expanded `FINetIndustrial/` output folders. Its portable ZIP retains every checkpoint/log/setting and the newest validation preview, while older previews remain in the saved version's outputs. Prepared masks are retained canonically so a Pillow/zlib version difference cannot silently change target byte hashes. The prepared `MVTecAD_supervised_v1/` image/mask cache is stored separately in the first saved run's working outputs; attach that output when resuming to avoid repeated archive decompression. The supervisor hash-verifies attached prepared data and refuses inconsistent caches. `data_location.json` records the actual active path; training signatures exclude machine-specific paths while retaining the exact split fingerprint.

A saved engineering snapshot can also restore the exact GPU-tested shared initialization and both matching smoke reports at **zero completed training epochs**. The loader verifies full tensor layout, initialization digests and initialization-file hashes; it never credits smoke updates as full training. The portable notebook prefers the dedicated source input over the older project copy inside a saved output, and verifies the runner's source SHA before execution. Kaggle Quick Save is used for a prepared-data snapshot only with **Save output for this version** selected; the default Quick Save does not preserve output files. Long training uses a bounded Save & Run All version.

The recovery ZIP excludes its own `recovery_archive.json` report. Otherwise regenerating the archive after writing that report would change the ZIP and invalidate the previously printed archive hash. The finalizer writes and prints the final archive hash after verification. `scripts/verify_industrial_snapshot.py` independently checks downloaded engineering archives, exact model/Adam/scheduler/loader restoration, common full initialization and all native target pixels across local/Kaggle PNG encodings.

## Evaluation and application

**Local baseline available November 16, 2026:** the verified final epoch-200 industrial baseline is installed at `checkpoints/industrial/baseline/last.pth` (SHA-256 `fd92df468740e977fb16e9bcc6f44dfcf424be8822d155f36ea4cc209d4db94a`). Start the main app with `START_FINET.bat`, then choose **Factory Quality Inspection → Original FINet**. App/registry defaults now use industrial `last.pth` files for the final-epoch protocol. The industrial boundary file is not installed while its full-budget training remains incomplete; comparison requires both finished matched runs. Real API inference on held-out normal/defective images, native-size artifacts and ZIP downloads passed, as did browser inference on the user's already-uploaded photograph. Evidence is in `outputs/industrial_app_20261116/`; these functional checks are not aggregate model-quality measurements.

`evaluate_industrial.py` requires both matched full-budget final checkpoints. It evaluates only the separate custom held-out test IDs and saves per-image/per-category Dice, IoU, native MAE, two-native-pixel Boundary F-score and measured forward mean/median/p95 latency after three validation-image warmups. Epoch validation computes structure loss, Dice, IoU and MAE; its unmeasured boundary field is explicitly null to avoid expensive native-resolution contour transforms every epoch. Final evaluation measures all requested metrics. Dice/IoU/BF1 use 1 for two empty masks and 0 when only one is empty. Report defective-only category metrics alongside all-image results so normal images do not hide failures. Genuine example predictions include normal and defective images. A comparison table is generated only after real evaluation succeeds.

Threshold **0.5** is preregistered for evaluation. There is no per-image min-max normalization, forced positive mask, hidden cleanup or test-tuned threshold. Tiny foreground noise is never sufficient for a product-level defect verdict: the application presents candidate defect regions and abstains from acceptance/rejection. A future product-level decision/filter must be calibrated on validation data and evaluated on untouched tests before it can be described as validated.

The same Gradio app, **FINet Vision AI**, has a domain selector and model selector. Camouflage and industrial registries/cache entries are separate. Factory inference requires the explicit MVTec protocol/domain in a genuinely trained checkpoint; placing a camouflage checkpoint in the industrial folder fails closed. Missing industrial weights produce an actionable error. A/B comparison requires completed matched industrial runs.

Launch with separate paths when trained weights exist:

```powershell
.\.venv\Scripts\python.exe app.py --baseline <camouflage-checkpoint> --industrial-baseline <industrial-baseline-checkpoint> --industrial-boundary <industrial-boundary-checkpoint> --device cpu
```

Do not substitute initialization or smoke files for these trained weights. Actual execution status, losses, checkpoint hashes, GPU availability and remaining gates belong in `PROGRESS.md`.
