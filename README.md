# Boundary-Aware FINet

Pixel-level segmentation of **camouflaged objects** and **industrial defects**, with a local Gradio application and a controlled comparison of FINet with and without boundary supervision.

FINet combines EfficientNet-B0 spatial features with block-DCT frequency features. Our boundary variant adds morphological BCE and Dice losses during training. Both variants retain the same **3,742,988-parameter** inference network.

## Verified results

| Experiment | Saved training status | Available evidence |
|---|---|---|
| Camouflage baseline | 200/200 epochs | Final validation MAE 0.044454 |
| Camouflage boundary | 200/200 epochs | Final validation MAE 0.044046 |
| Industrial baseline | 200/200 epochs | Three individual test examples |
| Industrial boundary | Kaggle recovery at 174/200 epochs | Final model and matched test comparison pending verification |

Camouflage MAE uses the same 404 validation images and one seed. The small final difference does not establish aggregate test or contour improvement. Industrial training uses a **custom supervised MVTec AD split**, which differs from the official normal-only anomaly benchmark.

## Run the application

The three verified final app checkpoints are included in `checkpoints/`. Dataset downloads are unnecessary for inference on your own photographs.

On Windows, install 64-bit Python 3.12, then run in the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-transfer-cpu.txt
.\START_FINET.bat
```

Open **http://127.0.0.1:7860**. Choose **Camouflaged Objects** for the completed baseline/boundary comparison, or **Factory Quality Inspection → Original FINet** for the completed industrial baseline. Upload a photograph; a matching mask is optional for metrics. Predictions export native probability maps, binary masks, overlays and provenance. Industrial boundary comparison becomes available after its completed checkpoint is installed.

## Faculty review

- [Research paper (PDF)](output/pdf/finet_research/paper.pdf) · [LaTeX source](output/pdf/finet_research/paper.tex) · [Complete LaTeX ZIP](output/pdf/finet_research/finet_latex_source.zip)
- [Review guide and code map](docs/REVIEW_GUIDE.md)
- [Controlled methodology](docs/METHODOLOGY.md) · [Industrial protocol](docs/INDUSTRIAL_MVTEC.md)
- [Checkpoint hashes](docs/CHECKPOINT_MANIFEST.json) · [Detailed evidence and dated progress](PROGRESS.md)

## Datasets and recovery files on Google Drive

**[Google Drive folder](https://drive.google.com/drive/folders/1dOmyEh9O8oKWYxZb__fVAMuBv6avcKbZ?usp=sharing)** for datasets and recovery archives excluded from Git. [Upload list and restore instructions](docs/GOOGLE_DRIVE.md) identify every required folder; the [file manifest](docs/drive_file_manifest.csv) lists individual files and sizes. Drive upload completeness has not been verified. This repository includes the app's three final checkpoints and the paper. Virtual environments and temporary caches can be recreated.

## Attribution

Based on [FINet by Liang et al.](https://github.com/CRRCOO/FINet), IEEE Signal Processing Letters, 2024, [DOI: 10.1109/LSP.2024.3356416](https://doi.org/10.1109/LSP.2024.3356416). The original architecture and four-head structure loss are retained. The loss source credits [F3Net](https://doi.org/10.1609/aaai.v34i07.6916). Original author documentation is preserved in the [archived README](docs/archive/README_before_github_20261118.md). No project-wide license is asserted for the upstream source. MVTec AD data retain their CC BY-NC-SA 4.0 terms; this is an academic prototype.
