# Completed camouflage experiments

Verified November 14, 2026. Both original FINet and Boundary-Aware FINet finished their full training budget. Aggregate held-out evaluation is a separate stage; this document does not claim an improvement before that stage finishes.

## Training evidence

| Item | Original FINet | Boundary-Aware FINet |
|---|---:|---:|
| Completed epochs | 200/200 | 200/200 |
| Adam updates | 181,800 | 181,800 |
| Model parameters | 3,742,988 | 3,742,988 |
| Boundary-loss coefficient | 0 | 0.1 |
| Final validation MAE, 404 images | 0.044454429 | 0.044045741 |
| Validation-selected epoch | 153 | 182 |
| Best validation MAE | 0.043108570 | 0.042976885 |

The original architecture, torchvision EfficientNet-B0 backbone initialization, all frequency-injection/decoder blocks, original DCT and four-head structure loss are preserved. Boundary-aware training adds differentiable morphological-boundary BCE+Dice to that original objective; it introduces no inference layer or extra parameters. Both experiments use the exact same initial full-model tensors, split, seed 123456, batch 4, image size 384, Adam settings and 200-epoch schedule. Configuration differences are only the boundary coefficient and run/initialization paths. A single seed cannot establish statistical significance.

The **primary controlled comparison uses the final epoch-200 checkpoints**. Validation-selected checkpoints remain separately preserved and must not be mixed into the fixed-budget comparison. Training losses have different objectives and should not be used to rank the two models.

Actual histories and charts: `outputs/final_camouflage_training/summary.json`, `training_curves.png`, `training_curves.pdf`.

## Independent recovery

Saved Kaggle Version 5 (`369255109`) completed successfully in 15,394.4 seconds. The independently downloaded backup is `outputs/kaggle_migration/kaggle_version005_recovery.zip`, **368,451,440 bytes / 214 members**, SHA-256:

`ab479e0c1b560bf03106af5e70e9a4fc1f95abb7c241ad3b4af1ecc10ed071f4`

Every member CRC passed. Both full checkpoints restored exactly on CPU: **965 model tensors, 581 Adam states, scheduler, Python/NumPy/torch and loader RNG**. Two saved CUDA RNG states are retained for GPU restoration. Every completed epoch 1–200 is present in each history.

Final checkpoint files and SHA-256:

- `checkpoints/baseline/last.pth`: `660a43bc1f5d57cbfd58d5cec7dd452f592932f34e1b8931567a30784af62810`
- `checkpoints/boundary/last.pth`: `c7859e2b2445dbc195dfc1f94908c862f678871e286eb6a1dd99ef45f3caee48`

Machine-readable verification: `outputs/kaggle_migration/kaggle_version005_completion_verification.json`. Reproducer: `scripts/verify_completed_camouflage.py`. The completed baseline was preserved byte-for-byte; no completed epoch was retrained during this task.

## Working application

```powershell
.\.venv\Scripts\python.exe app.py --device cpu
```

Open http://127.0.0.1:7860 . Both models load their final checkpoint by default. Research Comparison processes the same photograph and threshold through both models. Genuine grayscale probabilities, binary masks, colored overlays, predicted boundaries, native-size outputs, measured inference time and downloadable ZIPs are available.

Browser and app-service verification used unseen CAMO `camourflage_00012`, native 632 × 420, threshold 0.5. The saved audit excludes development overlap. Each result ZIP contains seven valid artifacts. Single-image metrics and timings are stored in `outputs/final_camouflage_app/verification.json`; they are demonstration measurements, not aggregate benchmark scores. The browser screenshot is `outputs/final_camouflage_app/gradio_comparison_verified.jpg`.

Validation: **28 relevant local checks passed**, including both genuine full-checkpoint/real-image inference checks. Fresh Kaggle protocol validation separately passed **46 checks** on a Tesla T4 with the original pinned training packages. Neither initialization nor engineering smoke weights are used by the app.

## Evaluation continuation

[Kaggle Version 6: Evaluate matched final epoch 200 models](https://www.kaggle.com/code/shreyashchilweri/finet-baseline-resume-from-colab/log?scriptVersionId=369384835) is the evaluation-only continuation. Both exact final checkpoint hashes and completed budgets are required before entering the existing supervisor; both trainers are skipped. Fresh Tesla T4 CUDA execution and 46 protocol tests have passed. Actual evaluation progress and final completion still require verification.

The unchanged evaluation pipeline audits CAMO, CHAMELEON, COD10K and NC4K; excludes development overlap and invalid geometry; evaluates native-resolution raw sigmoid probabilities at fixed threshold 0.5; and reports S-measure, MAE and Boundary F1 with two-pixel tolerance. It saves per-image CSVs, sample outputs, latency summaries, controlled-comparison audits, figures and `MEASURED_RESULTS.md`. A separate `finet_evaluation_results.zip` retains those outputs, including actual sample images. The portable evaluation notebook is `notebooks/project_evaluation_kaggle.ipynb`, disabled by default.

Remaining: verify and download completed aggregate evaluation, inspect comparison audits, write evidence-based research conclusions and finish submission material. The industrial MVTec experiments are a separate extension and were not rechecked in this camouflage task.
