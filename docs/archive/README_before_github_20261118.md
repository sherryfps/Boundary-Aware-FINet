# FINet Vision AI: Camouflage and Industrial Segmentation

Third-year computer vision project based on the original FINet implementation below.

**New laptop setup verified November 16, 2026:** double-click **`START_FINET.bat`** to start the full app and open **http://127.0.0.1:7860** automatically when ready. The launcher reuses an already-running app, keeps the server running after its console closes, and saves startup errors under `outputs/app_logs/`. `START_APP.cmd` is a compatible shortcut to the same launcher. The fresh local Python 3.12 environment passed dependency checks, both final models' real-image inference and all **148 tests**. Real comparison uploads and result downloads passed through the running app's local API. To restore dependencies, run `SETUP_NEW_LAPTOP.cmd`; it supports the system Python launcher or Codex's bundled Python 3.12. See the newest [progress entry](PROGRESS.md) and [laptop transfer guide](docs/TRANSFER_TO_NEW_LAPTOP.md). Cloud results below remain dated observations until the existing jobs are checked.

**Industrial baseline installed November 16, 2026:** the **200/200-epoch / 186,400-update** MVTec baseline is loaded in the main app at `checkpoints/industrial/baseline/last.pth`. Select **Factory Quality Inspection → Original FINet**. The actual running app passed real-image prediction and result-download checks on held-out defective/normal images, and the photograph already open in Chrome rendered a genuine prediction with epoch-200 provenance. **22 relevant tests passed**, including exact full-checkpoint loading and real-image inference. Industrial boundary weights and the matched final evaluation remain pending; Version 5 was launched separately with a session-end recovery checkpoint policy. This is a custom supervised MVTec split. See [the industrial methodology](docs/INDUSTRIAL_MVTEC.md) and [current evidence](PROGRESS.md).

**Camouflage training verified November 14: both baseline and boundary-aware FINet completed 200/200 epochs, 181,800 updates each.** The saved Version-5 archive passed every CRC and exact full-state CPU restoration; both continuous epoch histories, shared initialization and matched settings are verified. Both final epoch-200 checkpoints are installed in the local Gradio app, and genuine comparison inference, masks, overlays, timing and downloads passed on an unseen image. **28 relevant checks passed**, including both real-checkpoint inference tests. Kaggle Version 6 is the evaluation-only continuation; aggregate benchmark results and research conclusions remain pending verification. See [PROGRESS.md](PROGRESS.md), [Kaggle migration details](docs/KAGGLE_MIGRATION.md), [the demonstration](docs/DEMO.md) and [the runbook](docs/RUNBOOK.md).

FINet combines EfficientNet-B0 spatial features with block-DCT frequency features to produce pixel-level segmentation. The added single-image pipeline preserves the original model and preprocessing. Its only model-source compatibility change makes the initial ImageNet download optional when loading a complete FINet checkpoint; default upstream construction behavior is preserved.

- [Source architecture and inspection findings](docs/ARCHITECTURE.md)
- [Current training, application and evaluation runbook](docs/RUNBOOK.md)
- [Actual T4 smoke evidence and measured resource estimate](docs/GPU_TRAINING_VERIFICATION.md)
- [Methodology and controlled comparison protocol](docs/METHODOLOGY.md)
- [Viva preparation and honest result interpretation](docs/VIVA.md)
- [Genuine partial-training demonstration and continuation](docs/DEMO.md)
- [Training verification: real losses, gradients and checkpoint evidence](docs/TRAINING_VERIFICATION.md)
- [Exact Colab training instructions and full-run gate](docs/COLAB_TRAINING.md)
- [Colab training notebook](notebooks/baseline_training_colab.ipynb)
- [Windows/Colab setup, prediction commands and troubleshooting](docs/SETUP.md)
- [Official checkpoint links, current download blocker and exact placement](docs/CHECKPOINTS.md)
- [Earlier Phase 1 report, before the own-training route was authorized](docs/PHASE1_VERIFICATION.md)
- [Phase checklist and remaining work](PROGRESS.md)

In the prepared local environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe predict.py --help
```

After obtaining a fully trained EfficientNet-B0 FINet checkpoint (official or our selected `best.pth`) and a real unseen input photograph:

```powershell
.\.venv\Scripts\python.exe predict.py --image data/samples/camouflaged.jpg --checkpoint checkpoints/FINet.pth --output-dir outputs/example --device cpu
```

This saves grayscale and binary masks, a colored overlay, float probabilities and measured timing/checkpoint metadata at the original image dimensions. Large datasets, weights and outputs are ignored by Git. No invented predictions or metrics are included. The inspected upstream commit has no explicit license; preserve attribution and clarify permissions before redistributing original source/weights.

For training preparation and the ready-to-upload code bundle:

```powershell
.\.venv\Scripts\python.exe scripts/verify_backbone.py
.\.venv\Scripts\python.exe scripts/package_colab.py
```

Open the notebook and follow `docs/COLAB_TRAINING.md`; full training is disabled by default. The inference CLI rejects initialization and engineering-smoke checkpoints instead of presenting them as trained detector results.

Run the local application with `.\.venv\Scripts\python.exe app.py --device cpu` and open `http://127.0.0.1:7860`. Both verified final camouflage models are installed at `checkpoints/baseline/last.pth` and `checkpoints/boundary/last.pth`, or can be supplied with `--baseline` / `--boundary`. Validation-selected weights remain separately preserved in the recovery archive. The application never substitutes an untrained prediction.

The original authors' README is retained below. Its published benchmark numbers describe their research, not measurements from this project.

---

# FINet: Frequency Injection Network for Lightweight Camouflaged Object Detection (IEEE-SPL 2024)

> **Authors:**
> Weiyun Liang,
> Jiesheng Wu,
> Yanfeng Wu,
> Xinyue Mu,
> and Jing Xu.

## 1. Overview

- This repository provides code for "_**FINet: Frequency Injection Network for Lightweight Camouflaged Object Detection**_", IEEE Signal Processing Letters, 2024. [Paper](https://ieeexplore.ieee.org/document/10409558)

### 1.1. Introduction

 Existing camouflaged object detection (COD) methods typically have large model parameters and computations, hindering their deployment in real-world applications. Although using lightweight backbones can help alleviate this problem, their weaker feature representation often leads to performance degradation. To address this issue, we observe that frequency information has shown effective for cumbersome networks, but its effectiveness for lightweight ones has not been thoroughly investigated. Biological studies indicate that the human visual system utilizes distinct neural pathways to respond to different frequency stimuli, contributing to specialization and efficiency. Motivated by this, we propose an efficient frequency injection module (FIM) to aid lightweight backbone features by separately injecting detailed high frequency and object-level low frequency cues at each stage. FIM can be used as a plug-and-play component in existing COD networks to enhance backbone features at a low cost. With FIM, our proposed frequency injection network (FINet) achieves competitive performance against most state-of-the-art methods with much faster speed (692FPS for the input size of 384×384) and fewer parameters (3.74 M).

### 1.2. Framework

<p align="center">
    <img src="imgs/FINet.png"/> <br />
    <em>
    Figure 1: Overall architecture of the proposed FINet, which comprises two components: frequency injection module (FIM) and asymmetric decoder block (ADB). The FIM mainly consists of two modules: high frequency injection module (HFIM) and low frequency injection module (LFIM). F<sup>h</sup> and F<sup>l</sup> denote high and low frequency features, respectively.
    </em>
</p>

### 1.3. Quantitative Results

<p align="center">
    <img src="imgs/results.png"/> <br />
    <em>
    Figure 2: Quantitative Comparison.
    </em>
</p>

### 1.4. Qualitative Results

<p align="center">
    <img src="imgs/vis.png"/> <br />
    <em>
    Figure 3: Qualitative Comparison.
    </em>
</p>

## 2. Proposed Baseline

### 2.1. Prepare the Data

The training and testing datasets can be downloaded from https://github.com/GewelsJI/SINet-V2/tree/main .

You can modify `config.py` to set all the data paths. Here is an example of a setup:

1. Put the training data in folder `./data/TrainDataset`, and put the testing data in folder `./data/TestDataset`.

2. `TrainDataset` should contain two folders: `Imgs` and `GT`.

3. `TestDataset` should contain four folders: `CHAMELEON`, `CAMO`, `COD10K`, and `NC4K`. Each of the four folders contains two sub-folders: `Imgs` and `GT`.

### 2.2 Training Configuration

+ Traning hyperparameters and data paths can be modified in `config.py`.

+ Installing necessary packages:
   + pytorch: https://pytorch.org
   + torch-dct: https://github.com/zh217/torch-dct
   + pysodmetrics: https://github.com/lartpang/PySODMetrics
   + timm: https://github.com/huggingface/pytorch-image-models

### 2.3 Training, Inference, and Testing

+ Run `train.py` to train the model.

+ Run `inference.py` to generate the prediction maps. Set path to your trained checkpoint (`pth_path`).

+ Run `evaluate.py` to calculate the evaluation metrics for the prediction maps. Set path to your predictions (`pred_path`).

## 3. Pre-computed Maps and Trained Weights

+ Pre-computed Maps:
   + FINet (EfficientNet-B0): [Google Drive](https://drive.google.com/file/d/1mLOV0RcCVc0jlGFOcPQsEZ1j53OhvPjh/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1wUd4E9_Dt00adKh7o0JXpA) (extraction code: 0w3w)
   + FINet (TinyNet-A): [Google Drive](https://drive.google.com/file/d/1ogdU4KURdJSQGs-e7ciz1fo-viw_AVBf/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1RJKRg9sPjO9GQs_5nbUaOQ) (extraction code: ymjm)

+ Trained Weights:
   + FINet (EfficientNet-B0): [Google Drive](https://drive.google.com/file/d/1ePSiaW14I_NecnBAXE7O_-ApIuFFduuI/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1JNtzpoVZbVjF-15rFK6WuQ) (extraction code: hdrs)
   + FINet (TinyNet-A): [Google Drive](https://drive.google.com/file/d/1rzyEp-DeoYLePngk6rzdiGgNr1juYJTM/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/165lLc7iOD0WD6DsllgpC2Q) (extraction code: 22gg)


## 4. Citation

Please cite our paper if you find the work useful, thanks!

    @ARTICLE{10409558,
        author={Liang, Weiyun and Wu, Jiesheng and Wu, Yanfeng and Mu, Xinyue and Xu, Jing},
        journal={IEEE Signal Processing Letters},
        title={FINet: Frequency Injection Network for Lightweight Camouflaged Object Detection},
        year={2024},
        volume={31},
        pages={526-530},
        doi={10.1109/LSP.2024.3356416}
    }


**[⬆ back to top](#1-overview)**
## Industrial extension

The existing application now supports separate **Camouflaged Objects** and **Factory Quality Inspection** registries. Industrial weights must carry genuine trained MVTec provenance; unavailable weights cause a clear error. This extension preserves the original camouflage experiments. See [the industrial protocol and recovery documentation](docs/INDUSTRIAL_MVTEC.md) and [the industrial Kaggle notebook](notebooks/industrial_training_kaggle.ipynb). Implementation is not evidence that industrial training or evaluation has completed; current verified status is in `PROGRESS.md`.
