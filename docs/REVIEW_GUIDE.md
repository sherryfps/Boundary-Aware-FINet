# Faculty review guide

Start with the seven-page [paper](../output/pdf/finet_research/paper.pdf), then inspect the implementation and the evidence behind its claims.

| Path | Purpose |
|---|---|
| `Model/` | Original FINet, EfficientNet encoder, frequency injection and decoder |
| `src/boundary_loss.py` | Added differentiable morphological BCE and Dice objective |
| `train.py` | Original four-output structure-loss implementation |
| `src/training.py`, `configs/` | Matched initialization, seed, optimization and camouflage training |
| `src/industrial_data.py`, `src/industrial_training.py` | Custom supervised industrial split, sampling and training |
| `src/finet_adapter.py`, `src/preprocessing.py` | Strict checkpoint loading and shared inference preprocessing |
| `src/evaluation_metrics.py` | Native-resolution region and contour metrics |
| `app.py`, `src/app_service.py` | Gradio UI, domain-specific models and result exports |
| `notebooks/` | Colab/Kaggle training and evaluation entry points |
| `tests/` | Unit, integration, protocol and recovery checks |
| `checkpoints/` | Three verified final checkpoints, each below GitHub's file-size limit |
| `output/pdf/finet_research/` | Paper, figures, source provenance and review reports |

The boundary experiment changes the training objective, with lambda 0.1 and a 3×3 morphological kernel. It adds no inference layers. The camouflage runs share their complete initialization, split and 200-epoch budget. Industrial models train independently from the shared initialization, rather than fine-tuning camouflage weights.

The three industrial examples are spot checks, not a dataset-wide accuracy estimate. The industrial boundary model has a saved Kaggle epoch-174 recovery; continuation Version 6 was submitted, but final completion has not been independently verified. The paper predates that cloud recovery and states its own verified artifact scope. The locally installed industrial model is the completed baseline.

For the documented equation checks and preservation checks, read `technical_verification.json`, `humanizer_review.json` and `verification.json` beside the paper. Attribution review was limited to named sources; no dedicated AI-detector or plagiarism score is claimed.

## Validation

Run the suite from the project root after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Some integration checks require the datasets/recovery artifacts listed in [GOOGLE_DRIVE.md](GOOGLE_DRIVE.md). The historical laptop verification passed 148 tests; see dated entries in [PROGRESS.md](../PROGRESS.md) for their actual scope. The GitHub preparation entry records the checks run for publication.

## Documentation chronology

[METHODOLOGY.md](METHODOLOGY.md), [INDUSTRIAL_MVTEC.md](INDUSTRIAL_MVTEC.md) and the newest [PROGRESS.md](../PROGRESS.md) entries explain the current experiment. `SETUP.md`, `PHASE1_VERIFICATION.md`, the archived README and older progress entries retain earlier states; statements there about unfinished baseline training are historical. Use the root README for current app startup. `AGENTS.md`, `CODEX_HANDOFF.md` and `handoff/` preserve development continuity.
