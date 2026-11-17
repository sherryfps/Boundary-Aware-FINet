# Demonstration and continuation

## Current matched-model demonstration — November 14

Both camouflage experiments completed **200/200 epochs**, independently verified from Kaggle Version 5. The app now loads the exact final checkpoints by default:

```powershell
.\.venv\Scripts\python.exe app.py --device cpu --port 7860
```

Open http://127.0.0.1:7860 . Both model availability entries must show **200/200 epochs; training budget complete**. In **Research Comparison**, upload `data/TestDataset/CAMO/Imgs/camourflage_00012.jpg` and optional matching `GT/camourflage_00012.png`, retain threshold 0.5 and compare. The browser verified both overlays, predicted boundaries, real timing, optional ground-truth metrics and seven-file downloadable result ZIPs. The unseen sample is excluded from development overlap by the saved data audit. Screenshot: `outputs/final_camouflage_app/gradio_comparison_verified.jpg`; independent app-service evidence: `outputs/final_camouflage_app/verification.json`.

Single-image results are demonstration evidence, not aggregate proof of improvement. Fixed-budget A/B benchmark evaluation continues separately in Kaggle Version 6 (`369384835`); check `PROGRESS.md` for the verified execution status. Validation-selected epoch-153 baseline and epoch-182 boundary weights remain preserved separately. The app uses both **final epoch-200** checkpoints for the controlled demonstration.

## Historical completed-baseline demonstration — November 13

The original baseline has completed its 200-epoch budget. Its final checkpoint is `outputs/kaggle_migration/version003/FINet/experiments/baseline_seed123456_b4_v1/last.pth`, SHA-256 `660a43bc1f5d57cbfd58d5cec7dd452f592932f34e1b8931567a30784af62810`. The separately preserved validation-selected `best.pth` is from epoch 153 of the completed run; do not confuse its selected epoch with the full experiment's completion.

```powershell
.\.venv\Scripts\python.exe app.py --baseline outputs/kaggle_migration/version003/FINet/experiments/baseline_seed123456_b4_v1/last.pth --device cpu --port 7860
```

Open http://127.0.0.1:7860 . The availability panel must show **200/200 epochs; training budget complete**. Upload `data/TestDataset/CAMO/Imgs/camourflage_00061.jpg`, keep threshold 0.5 and run segmentation. Original, grayscale, binary, overlay and boundary outputs were browser verified. Download the actual ZIP; its report identifies the exact completed checkpoint. Screenshot: `outputs/baseline_completed_epoch200/gradio_verified.png`. CLI artifact-integrity evidence is under `outputs/baseline_completed_epoch200/unseen_camo_00061/`; no benchmark score or generalization claim follows from this single image.

The browser run measured 399.3778 ms forward / 470.6292 ms processing on the current CPU; model loading and saving are excluded. Actual full-checkpoint inference/app tests: **40 passed in 13.75s**. The boundary model remains unavailable in the app while its matched full-budget training continues. Its backed-up partial checkpoint is preserved for recovery, not presented as a completed A/B result.

Training continues in the private Kaggle saved background Version 4, `scriptVersionId=369058695`. It skips the finished baseline and resumes boundary; do not start another trainer from the draft. Check `PROGRESS.md` for the last observed epoch and verify the latest completed saved output before the next resume. The current idle draft/portable notebook include the evaluation dependency-graph fix needed for later evaluation. The Colab and epoch-6 instructions below are historical.

## Historical partial-training demonstration

The current verified local snapshot is `outputs/partial_baseline_epoch006/checkpoint.pth` (6 of 200 epochs), selected by validation among the first seven epochs. It is an actual checkpoint after 5,454 updates, not an initialization or smoke state. Its outputs are training diagnostics and must be described that way. The earlier epoch-1 snapshot remains separately preserved.

```powershell
.\.venv\Scripts\python.exe app.py --baseline outputs/partial_baseline_epoch006/checkpoint.pth --device cpu --port 7860
```

Open http://127.0.0.1:7860 . The availability panel must show `6/200 epochs; partial training checkpoint`. Upload `data/TestDataset/CAMO/Imgs/camourflage_00012.jpg`, keep threshold 0.5 and run segmentation. Show original image, gray probability, binary mask, overlay and predicted boundary. The optional original mask is `data/TestDataset/CAMO/GT/camourflage_00012.png`; supplying it enables actual per-image metrics. They are a single diagnostic, not final benchmark results. Download the generated ZIP and inspect its `prediction.json` for checkpoint hash, native dimensions, timing and training progress.

The current browser-verified screenshot is `outputs/project_verification/app_partial_epoch006.png`. CLI outputs are under `outputs/partial_baseline_epoch006/unseen_camo_result/`. The older epoch-1 screenshot is historical. Do not claim that the boundary model works in the comparison tab until its real checkpoint has been trained and loaded.

## Continue the live baseline

Open the saved Colab notebook: https://colab.research.google.com/drive/1BC_-WneXDaAKESRK2q5tYovliimnEqBh . The current foreground cell supervises the actual baseline, boundary and evaluation stages. Its process object is `project_job`; its stdout shows actual completed-epoch records. Eight completed baseline epochs were observed after recovery. Do not rerun setup/unpack or start a second training process while this job is active. Read stage status/logs from `MyDrive/FINet/projects/finet_matched_v1/`. Other kernel cells queue while the foreground cell runs.

The distributed notebooks leave long training disabled by default. If the runtime resets, use `project_training_colab.ipynb` and `RUNBOOK.md` to restore Drive/source/data, verify the same configuration and enable the sequential run. It resumes `last.pth` and retains completed-stage evidence. If GPU type changes, remeasure smoke with the same initial weights and a fresh smoke tag. Free runtime termination can require repeated resumes. Do not bypass service limits.

## Final demonstration after completed experiments

Install the selected actual checkpoints in `checkpoints/baseline/best.pth` and `checkpoints/boundary/best.pth`, or pass explicit paths. Check recorded completed/selected epochs and experiment audit. Run the same unseen image through both, show both overlays/boundaries and explain actual metrics using matching ground truth. Display measured evaluation CSV/plots, hardware and protocol. State any negative result or mismatch plainly. Replace the partial screenshot with final genuine outputs only after they exist.

Present methods from `METHODOLOGY.md`, execution evidence from `GPU_TRAINING_VERIFICATION.md`, and the limits/result interpretation in `VIVA.md`. Final baseline completion, boundary training, A/B scores, graphs and final conclusions remain pending actual computation.
## FINet Vision AI domain extension

The existing app is now named **FINet Vision AI** and provides Camouflaged Objects / Factory Quality Inspection selection plus domain-specific model weights. The main action is **Analyze Image**. Grayscale and binary views remain visible; boundary visualization is optional. A domain change clears previous results. A selected model's availability and training dataset remain visible.

The complete camouflage baseline continues to work. Industrial weights remain unavailable until genuine training completes; the app must display that state and reject inference rather than substituting camouflage or engineering smoke weights. Factory masks will be candidate defect regions, with no acceptance/rejection verdict inferred from small noise. Do not present an industrial success example until genuine held-out inference with trained weights has been saved.

Current browser verification evidence is under `outputs/industrial_setup/`; actual training/evaluation state is recorded in `PROGRESS.md`.
