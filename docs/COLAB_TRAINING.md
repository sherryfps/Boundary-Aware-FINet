# Colab training instructions

**Current instructions (2026-11-11): follow [RUNBOOK.md](RUNBOOK.md).** The notebook now obtains and verifies the official archive, preserves two original grayscale masks, groups duplicate images, uses the portable canonical split and backs up data/code to Drive. ImageNet initialization and dependency checks have been executed in Colab on a Tesla T4. Current actual training status is in [PROGRESS.md](../PROGRESS.md).

The November 9 preparation notes below are historical. Their statements about unexecuted Colab setup, unavailable full data and unimplemented application/boundary/evaluation code no longer describe the current project.

## Historical November 9 preparation

Current scope: verify torchvision ImageNet initialization, test real-batch training/checkpoints, and prepare Colab. **No long training has been launched.** Results and limits are in [TRAINING_VERIFICATION.md](TRAINING_VERIFICATION.md).

## Files to upload

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `notebooks/baseline_training_colab.ipynb` from this workspace.
2. In Colab choose **Runtime → Change runtime type → GPU**. The notebook checks `torch.cuda.is_available()`, the actual GPU name, and free/total GPU memory. GPU access is not guaranteed on the free tier; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html). No paid resource is required or started by these files.
3. Upload **`outputs/finet_colab_training.zip`** using Colab's Files sidebar. It contains current model/helpers, training CLI, tests, configuration, original statistics/DCT, notebook and documentation. It excludes `.venv`, datasets, weights and predictions.
4. Obtain the real full training collection separately and place its `Imgs` and `GT` folders in Drive as shown below. You do **not** need the unavailable authors' full FINet segmentation checkpoint for this training route.
5. Run notebook cells in order. The final full-training cell has **`RUN_FULL_TRAINING = False`**. Leave it false until the GPU smoke evidence and printed configuration have been reviewed.

Regenerate the code archive after editing files:

```powershell
Set-Location 'C:\Users\HP.DESKTOP-LG6I0U2\Desktop\finet\boundary-aware-finet'
.\.venv\Scripts\python.exe scripts/package_colab.py
```

The notebook has been parsed and every code cell compiled locally. **It has not been executed on a Google Colab GPU in this session.**

## Official full training data

FINet's [README](https://github.com/CRRCOO/FINet) directs users to [SINet-V2](https://github.com/GewelsJI/SINet-V2). Its current training/validation download is:

- [COD-TrainDataset.zip — Google Drive](https://drive.google.com/file/d/1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG/view?usp=sharing). The view page returned HTTP 200 with that archive title on 2026-11-09. The full archive was not downloaded or count-verified in this session.
- Original [COD10K project](https://dengpingfan.github.io/pages/COD.html) and [COD10K-v3 archive](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view?usp=sharing).
- Original [CAMO project/download page](https://sites.google.com/view/ltnghia/research/camo) and [CAMO release folder](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6).

For the combined archive, extract it and locate its paired training `Imgs`/`GT` directories. Copy those folders into the following Drive layout. If assembling original releases instead, use the **3,040 camouflaged COD10K Train images**, with their `Train/GT_Object` masks, plus **1,000 CAMO training images and object masks**. Do not include COD10K NonCAM, CAMO-Test, COD10K-Test, CHAMELEON or NC4K in this collection. The code cannot establish official Train/Test provenance merely from renamed files: preserve the actual training partition and download source.

```text
MyDrive/FINet/
  data/TrainDataset/
    Imgs/    # lowercase .jpg or .png, original filenames
    GT/      # binary 0/255 .png object masks with identical filename stems
  splits/    # notebook creates and retains a fixed split manifest
  experiments/
```

Masks must be object segmentation ground truth, not edges, instance IDs, scribbles or predicted masks. Match each image's decoded dimensions. Do not rename a JPEG-compressed preview to PNG. The audit checks mask values, filename matching, dimension matching, content duplicates, decoded readability, source counts and file hashes. A count mismatch stops preparation instead of being silently ignored. Classifying a filename as COD10K/CAMO is an audit aid, not independent proof of origin.

The notebook copies data to `/content/finet_data/TrainDataset` and stores checkpoints/logs in Drive. Its default fixed 10% holdout is stratified by dataset. If the expected full release is present, that produces **3,636 training / 404 validation** pairs. Those are planned counts; the actual counts are printed by the audit. Test datasets stay separate. Do not change the manifest while resuming a run.

## What is pretrained

`Model/EfficientNet.py` still calls torchvision's `efficientnet_b0` with `EfficientNet_B0_Weights.DEFAULT`. For initial training, `pretrained_backbone=True`. No timm backbone is substituted; timm remains an import dependency of the unused original TinyNet module.

Verified locally: the retained encoder has **3,595,388 ImageNet-pretrained parameters**. The full FINet has **3,742,988 parameters**, leaving **147,600 randomly initialized parameters** in channel reductions, frequency injection, decoder and output heads. All layers are trainable. No frequency modules or decoder blocks were changed.

ImageNet pretraining gives visual features; it does not by itself train FINet to segment camouflaged objects. A few smoke updates do not make a reliable segmentation model.

## Baseline settings to review

| Setting | Proposed Colab baseline |
|---|---|
| Model | Original FINet, EfficientNet-B0, channels `(8,24,32,64)` |
| Resolution | 384×384 |
| Initialization | Exact saved full-model state from before the GPU smoke run |
| Loss | Actual `train.structure_loss`, summed across all four outputs; no boundary term |
| Optimizer | Adam, lr 0.00026, weight decay 0.00000004 |
| Schedule | Original cosine formula, 200-epoch budget, min lr 0.0000001 |
| Precision | FP32; no AMP, gradient accumulation or gradient clipping |
| Batch size | **4 initially; GPU measurement required** |
| Workers | 2 in Colab; local Windows smoke used 0 |
| Seed | 123456 |
| Validation | Fixed 10% training-only holdout, no augmentation |
| Model selection | Lowest full held-out validation MAE |
| Saves | `last.pth` every epoch and `best.pth` on validation improvement |
| cuDNN | Enabled, deterministic flag true, benchmark false |

Changes from original `config.py`: batch 32 → initial candidate 4, workers 8 → 2, and a held-out validation split/checkpoint-selection protocol. For practical GPU execution the wrapper enables cuDNN; original `train.py` disabled it. This is an explicit runtime choice, not a change to loss or architecture. Set `cudnn_enabled=false` and rerun smoke if that original runtime setting is desired. The final A/B experiments must use identical settings.

**Original loss discrepancy is preserved:** upstream BCE uses `reduction='mean'` before multiplying by the weight map, so BCE weighting cancels. The weighted-IoU component remains weighted. This wrapper imports the actual original function; it does not silently change BCE to `reduction='none'`.

## What the GPU smoke checks

The notebook uses **three real training batches plus one checkpoint-resume verification update**, not 200 epochs. It checks `[B,3,384,384]` RGB, `[B,1,384,384]` masks and `[B,96,48,48]` frequency inputs; all four heads contribute original structure loss. Every trainable parameter must receive a finite gradient; encoder and head probes must change after Adam.

It saves `initialization.pth` before any updates and `smoke_checkpoint.pth` after measured steps. It reloads every model tensor and Adam/scheduler state exactly, checks identical evaluation logits, and performs a resumed optimizer update. These two files are tagged initialization/smoke: `predict.py` deliberately rejects them as application checkpoints.

Peak GPU allocated/reserved memory and timed steps are recorded. The memory peak includes save/reload verification with two model instances, so it is conservative for the single-model training path. Runtime estimation uses measured later steps and a measured two-image validation sample, extrapolated to the real split size. It excludes checkpoint I/O and is not a duration guarantee. **Local CPU timings cannot estimate Colab GPU time.**

If memory is insufficient, set notebook `BATCH_SIZE=2`, choose a fresh `RUN_NAME`, and rerun smoke. Batch 1 fails the original pooled BatchNorm branches, and dropping one-image final batches is intentional (`drop_last=True`). Batch 8 should be considered only after a successful separate measurement. Changing batch size or workers alters the run signature and requires a new smoke run, not a silent mid-experiment adjustment.

## Full run and resuming — only after review

The notebook refuses full training without actual CUDA availability, a full audited 4,040-pair manifest, and a passed GPU smoke report matching the configuration, split, GPU and initial-state file. It starts from the saved **pre-smoke** initialization, so smoke optimization does not contaminate baseline initialization. Keep this file for Model B later.

Only after reviewing the printed evidence, set `RUN_FULL_TRAINING=True` and run the final cell. The CLI equivalent is:

```text
python train_baseline.py --config outputs/colab_full_config.json --mode train --start-full-training --verified-smoke-report /content/drive/MyDrive/FINet/experiments/baseline_seed123456_b4_v1_smoke_v1/smoke_report.json
```

The notebook supplies actual paths and automatically adds `--resume <RUN_DIR>/last.pth` when it exists. After a runtime disconnect, rerun setup, mount/copy, path/config and report cells, keeping the same `RUN_NAME`, batch size, workers, seed, data and schedule. The existing passed smoke report is reused only if the configuration and current GPU match. If the GPU type changes, change only `SMOKE_TAG` to create a new smoke directory and measure again. The notebook reloads the run's existing pre-smoke initialization for this measurement, points the full configuration at the new smoke snapshot of those same starting weights, and passes the new GPU report. `RUN_NAME` remains unchanged so `last.pth` resumes the existing baseline.

Checkpoints contain model tensors, Adam state, scheduler position, completed epoch, global step, best validation MAE, Python/NumPy/PyTorch/CUDA/DataLoader RNG states, configuration and split fingerprint. Recovery is at **saved epoch boundaries**. An interrupted epoch is replayed; exact mid-epoch continuation is not claimed. A resume marker identifies any earlier partial log tail that will be replayed; do not count that interrupted tail as additional completed optimization. Stochastic CUDA execution may still depend on platform/kernel reproducibility.

Drive outputs:

```text
experiments/<RUN_NAME>/
  initialization.pth       pretraining start state, not a detector
  last.pth                 latest saved full epoch + optimizer/scheduler/RNG
  best.pth                 selected by full validation MAE
  config.json
  split_manifest.json
  hardware.json
  steps.jsonl
  epochs.jsonl
  validation_samples/      up to three diagnostic images per epoch
```

Validation records the four-head structure loss and **native-resolution probability MAE**, averaged per image. Sample validation masks/overlays are diagnostic outputs after real baseline epochs, not final benchmark results. S-measure and Boundary F-score evaluation remain later work.

## Existing local evidence and reproduction

Local CPU smoke used six real COD10K training pairs (four train, two validation) from a [public mirror](https://huggingface.co/datasets/nobg/COD10K) declaring [SINet](https://github.com/DengPingFan/SINet) provenance. Its original PNG mask bytes were read from Parquet; JPEG viewer previews were rejected. Mirror revision, original names, hashes and dimensions are recorded in `data/smoke_cod10k/provenance.json`. They have not been byte-matched against an official release. This small source is for engineering checks only and is not the full training collection.

Already executed locally:

```powershell
.\.venv\Scripts\python.exe scripts/verify_backbone.py
.\.venv\Scripts\python.exe scripts/prepare_training_data.py --data-root data/smoke_cod10k --output outputs/splits/smoke_cod10k.json --validation-fraction 0.3333333333333333 --allow-small
.\.venv\Scripts\python.exe train_baseline.py --mode smoke --data-root data/smoke_cod10k --split-manifest outputs/splits/smoke_cod10k.json --output-dir outputs/training_smoke_cpu --device cpu --batch-size 2 --num-workers 0 --steps 2
```

To repeat the smoke using existing data/split, choose a new output directory such as `outputs/training_smoke_cpu_repeat`. The runner preserves earlier evidence by refusing a nonempty directory. To fetch the six-pair sample on another machine, install `requirements-smoke-data.txt` and run `scripts/fetch_smoke_data.py`; it uses public HTTP range reads of the original encoded bytes, not the lossy previews. The baseline trainer itself does not need PyArrow.

## After actual training

Use `best.pth` with the existing single-image pipeline, for example:

```text
python predict.py --image /content/unseen_camouflage.jpg --checkpoint /content/drive/MyDrive/FINet/experiments/baseline_seed123456_b4_v1/best.pth --output-dir /content/unseen_result --device cuda:0
```

The adapter requires the complete state and records project-trained provenance, SHA-256, actual device, timing and original-size grayscale/binary/overlay outputs. It is ready for a later Gradio integration. No successful unseen-image inference or Gradio result is claimed yet because a full baseline has not been trained.

For the later boundary comparison, use the exact same pre-smoke full-model initialization, dataset fingerprint, holdout, optimizer, schedule, augmentation, seed and runtime settings. Model A uses original structure loss; Model B will add the separately tested boundary term. No boundary implementation, ablation, improvement or final test score is claimed here.
