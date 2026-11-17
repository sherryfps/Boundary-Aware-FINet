# Research evidence

Paths below are relative to the repository root. None of the original FINet publication's benchmark values are presented as results of this project.

| Paper content | Evidence |
|---|---|
| Inference architecture, initialization and DCT inputs | `Model/FINet.py`, `Model/EfficientNet.py`, `utils/dataloader_freq.py`, `utils/dct.py`, `src/preprocessing.py` |
| Original structure loss and scalar BCE behavior | `train.py`, function `structure_loss`; original FINet source credits F3Net. The retained FINet BCE uses a batch scalar, unlike F3Net's reference pixelwise weighted BCE. |
| Morphological boundary, BCE clamping, Dice smoothing and four head objective | `src/boundary_loss.py` |
| Matched training configuration, seed and scheduler | `configs/baseline.json`, `configs/boundary.json`, `src/training.py`, `utils/LRScheduler.py` |
| Both camouflage runs completed 200 epochs and 181800 updates | `outputs/kaggle_migration/kaggle_version005_completion_verification.json` |
| Camouflage final and minimum validation MAE | `outputs/final_camouflage_training/summary.json`; continuous `epochs.jsonl` histories under `outputs/kaggle_migration/version005/FINet/experiments/` |
| Camouflaged object input, original ground truth and baseline versus boundary predictions | `data/TestDataset/CAMO/Imgs/camourflage_00012.jpg`, matching `GT/camourflage_00012.png`; `outputs/final_camouflage_app/473a9caeeccc41b481abc9b6bae86eba/` and `22f0806f5f04488baf5e3b8f592b88ac/` |
| Fixed camouflage development split | Saved `split_manifest.json` in those experiments; fingerprint `2100aa40c52f01442d59b28973fee653eec655e370c8b1c1ff3c8d40cf61d724` |
| Supervised MVTec splits and all 15 categories | `data/MVTecAD_supervised_v1/split_manifest.json`; canonical GPU manifest in `outputs/industrial_setup/version004/FINetIndustrial/split_manifest.json` |
| Industrial final baseline and 186400 updates | `outputs/industrial_setup/industrial_version004_checkpoint_verification.json` |
| Industrial masks and sample measurements | `outputs/industrial_three_samples_20261116/report.json`, `results.csv`, individual float probability maps and original masks |
| Boundary F1 tolerance and empty mask convention | `src/evaluation_metrics.py`, `src/industrial_training.py` |
| Exact DCT and color matrix formulas | `utils/dct.py`, `src/preprocessing.py`; the implemented color transform retains coefficient 0.564 |
| Parameter composition and tensor stages | `outputs/training_verification/backbone.json`, `Model/EfficientNet.py`, `Model/FINet.py` |
| Common final validation structure loss and crossing histories | Saved 200-epoch histories used by `build_figures.py`; final values 2.5909169464 / 2.6009304885 |
| Pixel errors and counts in the new output figures | Same saved native probability files and masks; independently counted TP/FP/FN in `evidence.json` |
| Equation and sampler engineering checks | `technical_verification.json`; global batch BCE, weighted IoU, boundary loss, orthonormal DCT, cosine endpoints and equal stratum mass |

The evidence builder rechecks the 200 epoch histories and 404 validation samples. For the CAMO example it verifies native pixels, absence of the ID from the development manifest, exact final checkpoint hashes, saved thresholds and independently recomputed MAE, Boundary F1, Dice and IoU. The image is a previously demonstrated example, not a random test sample. The builder also verifies the industrial binary masks and Dice and IoU counts. Source hashes and calculations are in `evidence.json`. No new inference or training was run to select a favorable result.

## Primary reference sources

1. FINet publication metadata and authors' implementation: https://github.com/CRRCOO/FINet and https://doi.org/10.1109/LSP.2024.3356416
2. EfficientNet proceedings record: https://proceedings.mlr.press/v97/tan19a.html
3. COD10K publication: https://openaccess.thecvf.com/content_CVPR_2020/html/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.html
4. CAMO publication and journal metadata: https://arxiv.org/abs/2105.09451 and https://doi.org/10.1016/j.cviu.2019.04.006
5. MVTec dataset and author publication record: https://www.mvtec.com/research-teaching/datasets/mvtec-ad and https://www.mvtec.com/research-teaching/publications
6. DCT original publication DOI: https://doi.org/10.1109/T-C.1974.223784 . Orthonormal convention and original citation also checked in the IEEE article https://read.nxtbook.com/ieee/signal_processing/signal_processing_sept_2023/perspectives.html .
7. Adam authors' paper and conference metadata: https://arxiv.org/abs/1412.6980 .
8. F3Net loss lineage: https://ojs.aaai.org/index.php/AAAI/article/view/6916 and authors' reference implementation https://github.com/weijun-arc/F3Net/blob/master/src/train.py . The manuscript bibliography follows first-citation order, so F3Net precedes Adam there. This credit does not attribute FINet's scalar-BCE behavior to the F3Net formulation.

## Final prose and attribution review

The November 17 final review covered the whole current manuscript using Humanizer 4.0.0. All twenty numbered equation blocks, inline/display mathematics, ten table bodies, numerical research tokens, existing reference entries, author details and figure/evidence bytes were preserved. The only added technical-source material is the FINet/F3Net loss-lineage credit and F3Net bibliography entry. All seven final pages were compiled and visually inspected.

`attribution_review.json` records a limited local exact-phrase comparison against nine named public primary-source documents, including author code and dataset/model abstracts. No manuscript was sent to a checker or search engine. No six- or eight-word prose matches were found against that corpus; this is not a plagiarism percentage or a comprehensive originality determination. `originality_review.md` summarizes the method and its limits. No AI-detector or dedicated plagiarism-service score was obtained.

## Availability checked during preparation

On November 16, 2026, the camouflage evaluation Version 6 link returned "We can't find that page" in the current Chrome Kaggle account. This does not prove that evaluation failed or produced no outputs. Its aggregate results remain unavailable for independent verification and are not reported.

At that November-16 check, the industrial Version 5 log was accessible and showed a running Tesla T4 job with actual epoch 26 updates through at least step 24050 and finite gradients. These live updates are not a completed saved checkpoint or a final industrial result. The artifact set used in the paper contains an independently downloaded boundary recovery at epoch 6. These are dated observations, not a current claim about the cloud job. No cloud job was accessed or changed during the technical expansion.
