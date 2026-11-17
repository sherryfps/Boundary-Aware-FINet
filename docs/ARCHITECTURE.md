# FINet source inspection — Phase 0

This preserves the original inspection. The follow-up request now authorizes training preparation from the original ImageNet-pretrained backbone; see [training verification](TRAINING_VERIFICATION.md) and [Colab instructions](COLAB_TRAINING.md) for the current scope and tested state.

Inspected on 2026-11-09. Foundation: [CRRCOO/FINet](https://github.com/CRRCOO/FINet), local commit `9e42b6bf607a2c89823f5f88dd8d38069039102e`.

## In plain English

FINet looks for hidden objects using two views of the same image. EfficientNet-B0 reads normal RGB pixels. A discrete cosine transform (DCT) describes patterns inside small image blocks. Frequency injection modules mix those patterns with image features at four scales. The decoder combines the scales to predict which pixels belong to the object. The output is a segmentation map, not an object label or bounding box.

## Inspected files

- `README.md`: paper attribution, datasets, checkpoint links and original commands.
- `config.py`: 384-pixel square inputs, data paths, CUDA-only default and training settings.
- Every file in `Model/`: FINet, its two backbones and convolution/attention modules.
- `utils/dataloader_freq.py`, `utils/dct.py`, `utils/freq_mean_std.pkl`: RGB/frequency transforms and bundled statistics (statistics inspected programmatically during tests).
- `train.py`, `utils/data_augmentation.py`, `utils/LRScheduler.py`: four-output supervision, augmentation, Adam and cosine learning-rate decay.
- `inference.py`, `speed_test.py`: checkpoint loading, output selection, resizing, sigmoid and timing.
- `evaluate.py`, `utils/metrics.py`, `utils/tools.py`: saved-map evaluation, metric wrappers and optional complexity profiling.

## Exact architecture and interfaces

The official inference constructor is `FINet(backbone='efficientb0', channels=(8, 24, 32, 64))`. **Do not use the class's different default channels `(8, 12, 24, 48)` when loading the official checkpoint.**

`EfficientNet_B0` uses torchvision's EfficientNet-B0 features through feature index 7. Returned feature channels are `[16, 24, 40, 112, 320]`. FINet uses the last four of these, reduces their channels, and applies `FFM` at each scale. `FFM` contains `HFA` (high-frequency injection) and `LFA` (low-frequency injection). These are the source-code names for the proposal's FIM/HFIM/LFIM.

The asymmetric decoder is `DeBlock`: a 1×1 projection and three summed depthwise-separable paths with 3×3, 1×3 and 3×1 kernels. Skip connections merge spatial scales. Four convolutional heads produce **logits**, each resized to the network input resolution. Inference selects output index 0.

At resolution 384, `forward(rgb, high, low)` receives:

| Input/output | Shape | Meaning |
|---|---|---|
| RGB | `[B, 3, 384, 384]` float32 | ImageNet-normalized RGB |
| High frequencies | `[B, 96, 48, 48]` float32 | Last 32 flattened DCT coefficients per color channel |
| Low frequencies | `[B, 96, 48, 48]` float32 | First 32 flattened DCT coefficients per color channel |
| Outputs 1–4 | each `[B, 1, 384, 384]` | Segmentation logits |

## Exact preprocessing

1. Decode with Pillow and convert to RGB, as in `TestDataset.rgb_loader`. Keep original width/height. No EXIF rotation, cropping, padding or test augmentation is applied, matching the original loader.
2. RGB branch: torchvision PIL `Resize((384, 384))` (bilinear), `ToTensor()` (0–1), then normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
3. Frequency branch: independently resize the RGB PIL image, then `PILToTensor()`; **retain the 0–255 scale**, not the normalized RGB tensor.
4. Call the original `utils.dct.dct_2d`: convert using its exact YCbCr matrix and offsets, reshape into nonoverlapping 8×8 blocks, and apply orthonormal 2D DCT.
5. Rearrange to 192 channels: 64 row-major coefficients for Y, then 64 for Cb, then 64 for Cr. This is a first-half/second-half coefficient split, **not zigzag ordering**.
6. Normalize with the bundled `utils/freq_mean_std.pkl` and divide by **7.0**. Split each color channel's 64 coefficients into 32 low and 32 high channels, concatenating Y/Cb/Cr within each branch.

`src/preprocessing.py` will reuse the original DCT function and verify parity against `TestDataset`. It resolves the statistics path relative to the source file, so inference can run from other working directories. Only the repository-bundled statistics pickle is read; never pass an arbitrary external pickle.

## Checkpoints and inference

Upstream saves `model.state_dict()` directly. Inference needs the **complete trained FINet state**, including encoder and decoder; standalone ImageNet EfficientNet weights are insufficient. The adapter will load to CPU with `torch.load(..., weights_only=True)`, check state-dict contents and finite values, and use `load_state_dict(..., strict=True)`. It will also support conventional `state_dict`/`model_state_dict` wrappers and a uniform `module.` prefix, without dropping mismatched keys. A SHA-256 fingerprint records the file actually loaded; a fingerprint alone does not prove author provenance.

Upstream unconditionally initializes the backbone using ImageNet weights, causing an unnecessary download before loading full FINet weights. The only planned model-source compatibility change is an optional `pretrained_backbone` switch (default preserves upstream behavior) passed to the existing backbone. The inference adapter disables this initialization, then requires a complete checkpoint. No layers, forward operations, channel settings or state-dict keys change.

Postprocessing must resize **logits first**, using bilinear interpolation with `align_corners=True`, then apply sigmoid, matching `inference.py`. Do not min-max normalize predictions. Save the grayscale map using `(probability * 255).astype(uint8)` as upstream does; save the original float probability map as NumPy as well. Thresholding and overlay creation use the restored float map. Timing must separate the synchronized forward pass from image preprocessing/postprocessing; no paper speed is represented as a local measurement.

## Original training loss: important source discrepancy

All four outputs receive `structure_loss`, and those four losses are summed. The function computes a boundary-weight map using a 31×31 average pool and combines BCE-with-logits with weighted IoU. However, its BCE call uses **`reduction='mean'`**, producing a scalar before applying the weight map. Consequently the subsequent weighted average cancels the weighting for the BCE component. The source's actual BCE behavior differs from the usual pixelwise weighted BCE description. Preserve it for any later baseline; changing to `reduction='none'` would be a separate research-method change requiring explicit discussion. Phase 0/1 does not modify training.

## Dependencies and local compatibility

- Core: Python, PyTorch, torchvision, NumPy, Pillow, torch-dct.
- `timm` is required even for EfficientNet because `Model/FINet.py` imports the TinyNet module at import time.
- Upstream utilities: tqdm; evaluation additionally needs OpenCV and `pysodmetrics`; optional profiling uses `thop`; the scheduler's standalone plot uses Matplotlib.
- Phase 1 uses Pillow for PNG saving and overlays. Evaluation/profiling dependencies belong in an optional requirements file; Gradio belongs to Phase 2.
- Original `Config` defaults to CUDA and original inference calls `.cuda()` directly. The new adapter selects available CUDA or CPU and reports the actual device.
- Original dataset loaders require ground-truth folders and read frequency statistics relative to the working directory. Single-image inference removes the ground-truth dependency without changing model preprocessing.
- DCT assumes square dimensions divisible by 8. HFA also upsamples stride-two intermediate maps. The default verified interface will use 384; optional preprocessing sizes must be positive multiples of 64 to preserve compatible spatial scales.
- This machine reports Intel HD Graphics 630, with no NVIDIA device found. CPU is the local verification target. No `python`/`py` command was initially available on PATH; a bundled Python 3.12.14 was found and used to create `.venv`.
- Versions will be recorded only after installation and compatibility checks; no pretrained compatibility claim is possible until authentic weights load.

## Official checkpoint access (checked 2026-11-09)

From the original README's **Trained Weights → FINet (EfficientNet-B0)**:

- [Google Drive](https://drive.google.com/file/d/1ePSiaW14I_NecnBAXE7O_-ApIuFFduuI/view?usp=sharing): HTTP **404** for both the view URL and direct-download endpoint.
- [Baidu Drive](https://pan.baidu.com/s/1JNtzpoVZbVjF-15rFK6WuQ), extraction code **`hdrs`**: HTTP 200, but page title `百度网盘-链接不存在` means “link does not exist.” It did not provide checkpoint bytes.

These are unavailable links, not a successful authenticated-download test. The user confirmed they have no checkpoint. See `docs/CHECKPOINTS.md` for placement and recovery instructions. No random checkpoint, cached prediction map or alternate network can verify Phase 1.

## Datasets and separation

FINet directs users to [SINet-V2](https://github.com/GewelsJI/SINet-V2). That project's README lists [training/validation data](https://drive.google.com/file/d/1M8-Ivd33KslvyehLK9_IUBGJ_Kf52bWG/view?usp=sharing) and [test data](https://drive.google.com/file/d/1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o/view?usp=sharing). These large archives are not downloaded or verified in Phase 1. Follow the current source README if its links change.

Expected FINet placement:

```text
data/TrainDataset/Imgs/       training RGB images
data/TrainDataset/GT/         matching training masks
data/TestDataset/CHAMELEON/Imgs/ and GT/
data/TestDataset/CAMO/Imgs/ and GT/
data/TestDataset/COD10K/Imgs/ and GT/
data/TestDataset/NC4K/Imgs/ and GT/
```

Count downloaded image/mask pairs and check filename and dimension matches before later training. Derive a fixed validation split from training data only; do not select boundary hyperparameters with final test datasets. No dataset counts are verified yet. Single-image inference needs no ground truth or dataset download.

## Attribution and license finding

The inspected commit contains **no LICENSE or COPYING file** and the README provides a citation but no explicit reuse license. Public source availability does not establish redistribution permission. Preserve the original files and attribution; do not add a blanket license covering the original model. Ask the authors about licensing before redistributing their source/weights. No remote push is authorized or performed.

Liang, W.; Wu, J.; Wu, Y.; Mu, X.; Xu, J. “FINet: Frequency Injection Network for Lightweight Camouflaged Object Detection.” IEEE Signal Processing Letters 31 (2024), 526–530. [DOI: 10.1109/LSP.2024.3356416](https://doi.org/10.1109/LSP.2024.3356416).

## Development plan and phase gate

Phase 0: inspect original sources, document interfaces/dependencies/checkpoint access and source restrictions. Phase 1: create an isolated CPU environment, add a strict full-checkpoint adapter, exact preprocessing, postprocessing, CLI, smoke test and automated tests. A genuine local photograph and the official full checkpoint must pass the smoke test before Phase 1 is verified.

Stop after Phase 1, or report its precise external blocker. Do not implement Gradio, boundary loss, training or new metrics in this task. The user's immediate instructions authorize Phase 0 and Phase 1 together; later phases require approval.
## Industrial domain adaptation

The industrial experiment retains the same FINet architecture, parameter count, torchvision ImageNet EfficientNet-B0 initialization, frequency injection, DCT preprocessing and decoder. It changes the supervised data domain and documents its own loader/sampling policy. Normal MVTec examples have empty masks; defective examples use binary annotations. Industrial targets use nearest-neighbor resizing and horizontal flipping, with no label pepper noise or defect-removing crop. Both industrial variants use that identical adaptation.

`src/industrial_data.py` audits the custom category-aware train/validation/test protocol. `src/industrial_training.py` reuses the original structure objective and common optimizer/checkpoint primitives without changing COD training. `evaluate_industrial.py` evaluates completed matched states on the separate held-out custom test set.

The shared app routes `(application domain, model variant)` to separate checkpoint paths and cache entries. Industrial inference requires explicit MVTec provenance; camouflage weights cannot silently serve factory requests. Missing industrial checkpoints produce a clear error. See `docs/INDUSTRIAL_MVTEC.md` for limitations, licensing and the distinction from official anomaly-benchmark evaluation.
