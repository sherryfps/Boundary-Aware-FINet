# Actual Colab GPU training verification

Measured on 2026-11-11 and 2026-11-12 in the user's Colab notebook. This verifies execution, gradients and checkpoint recovery. It does not establish final camouflage-detection accuracy or an A/B improvement.

## November 12 recovery and matched boundary GPU smoke

Restored the same source/data/split and T4 software environment after the earlier runtime ended. Drive `last.pth` contained seven completed epochs and 6,363 updates. The resumed foreground run completed epoch 8: mean training loss 2.558689652758725, validation structure loss 2.6343660549362107, native MAE 0.05878847478692982 across 404 validation images in 31.682531393 seconds. The selected best updated; full training continues.

Boundary smoke uses identical complete initial tensors, lambda 0.1, kernel 3 and the same batch/seed/data/optimizer/schedule. Real total losses: 6.647243976593018, 6.733438968658447, 6.661501884460449. Their structure components were 6.194746017456055, 6.302337646484375 and 6.237039566040039; boundary components 4.524979591369629, 4.311013698577881 and 4.244622230529785. All 581 gradient tensors were finite and encoder/head weights updated. All model tensors, Adam/scheduler states and output reloaded exactly. The resumed update loss was 6.509711265563965. Peak allocation was 1,434,628,608 bytes. The shared initial digest remains `e7c3b4edd324ad589dc4404b952caaa7aefd17151d0844bf1b9219f7e2b578e9`.

Actual downloaded evidence is under `outputs/colab_verification_nov12/`. The local epoch-6 selected checkpoint SHA-256 is `a7919c82ae9bffb1d81d7ec2593ac2e5f73e5938a0e65c8162ce3448ca40cc46`. It loaded all 965 model tensors strictly and produced finite native 632x420 probability/mask/overlay artifacts for the unseen flatfish image. One measured CPU CLI forward was 182.8957 ms, processing 219.0514 ms. The actual matching-mask S-measure 0.9358313, MAE 0.04009477 and Boundary F1 0.2574028 describe only this single partial-training diagnostic. The app displays `6/200` and an explicit partial-quality note. Sixteen inference tests passed with this newer checkpoint in 7.55 seconds; the complete suite previously passed 106 tests in 19.70 seconds.

The sections below retain the original November 11 smoke and first-epoch evidence.

## Environment and initialization

- Tesla T4: 14.563171 GiB total memory, 14.460754 GiB free at the initial check.
- Python 3.13.16; PyTorch 2.14.1+cu130; torchvision 0.29.1+cu130; CUDA runtime 13.0.
- `pip check` passed after installing Colab's missing Jedi dependency.
- Official torchvision EfficientNet-B0 ImageNet file SHA-256: `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`.
- All 352 retained encoder tensors matched torchvision exactly. Feature channels were 16, 24, 40, 112 and 320. All four full-model outputs had shape `[1,1,384,384]` in the architecture check.
- 3,595,388 parameters are ImageNet initialized; 147,600 FINet parameters start randomly; total 3,742,988. Frequency injection, decoder and output heads retain the original implementation.
- Colab unit suite: 85 passed, 4 skipped in 8.62 seconds. The optional metric package was absent, and genuine trained-checkpoint inference was not yet available. Local full suite with application/evaluation dependencies: 90 passed, 2 checkpoint-dependent skips in 13.98 seconds.

## Real data and measured steps

The official archive was extracted and audited in Colab: 3,040 COD10K + 1,000 CAMO pairs, with 3,636 training / 404 validation. Canonical split fingerprint: `2100aa40c52f01442d59b28973fee653eec655e370c8b1c1ff3c8d40cf61d724`.

Batch 4, FP32, input size 384, workers 2, seed 123456, Adam lr 0.00026 / weight decay 4e-8, original 200-epoch cosine schedule / minimum lr 1e-7. No boundary term, AMP, accumulation or clipping. cuDNN explicitly enabled/deterministic with benchmark disabled. Exact original four-output structure loss, including its scalar BCE behavior, was retained.

| Update | Total original structure loss | Elapsed seconds |
|---|---:|---:|
| 1 | 6.194746017456055 | 4.121427461 |
| 2 | 6.302223205566406 | 0.248043464 |
| 3 | 6.236763954162598 | 0.333321384 |
| Reload continuation | 6.085261344909668 | Not used for the timing estimate |

Different real batches need not decrease monotonically. Every measured update had finite gradients for all 581 trainable parameter tensors. Encoder and output-head probes changed after Adam. RGB `[4,3,384,384]`, masks `[4,1,384,384]`, high/low frequency `[4,96,48,48]` and four `[4,1,384,384]` outputs were verified.

All model tensors, Adam state and scheduler state reloaded exactly. Evaluation outputs matched exactly after reload; a resumed backward/optimizer update succeeded. A two-image validation diagnostic produced structure loss 6.1091978549957275 and native-probability MAE 0.48270083963871 in 0.430465922 seconds. This tiny diagnostic is not a validation benchmark or successful detection result.

## Saved evidence

Drive directory: `MyDrive/FINet/experiments/baseline_seed123456_b4_v1_smoke_v1/`.

Downloaded and verified locally under `outputs/colab_verification/`:

- `smoke/smoke_report.json`, `config.json`, `steps.jsonl`.
- `smoke/initialization.pth`, SHA-256 `d56a876cc8db9866b86799e2c0cd877e40004ec825e73e3257557997b0ff2958`.
- `smoke/smoke_checkpoint.pth`, SHA-256 `e64c0d903e02d86521e3ae7e08b61e3e8cdcb759ec56cf16c352494346d5afff`.
- `colab_full_config.json`, `colab_pytest.xml`, `backbone.json`.

Both downloaded checkpoint file digests match the actual report. Shared starting tensor digest for future A/B experiments: `e7c3b4edd324ad589dc4404b952caaa7aefd17151d0844bf1b9219f7e2b578e9`. Keep this full initial model, not merely the same ImageNet encoder or seed.

## Measured resource estimate and launch

Peak allocation was 1.254800 GiB; peak reserved memory was 1.345703 GiB, including reload verification. Later-step mean was 0.290682424 seconds. For 909 training steps and an extrapolation of the two-image validation timing, the estimate is 5.853074 minutes per epoch, or approximately 19.51 hours for 200 epochs, excluding checkpoint I/O, contention, interruptions and other variability. This is a short-run extrapolation, not a guaranteed duration.

After the evidence/configuration was shown and T4 availability confirmed, the authorized free baseline was launched with the full-run guard. The first epoch completed 909 updates with mean training loss 3.785940072979733. All 404 validation images were processed in 32.528183408 seconds, with structure loss 3.008369400064544 and native MAE 0.08454456703626725. Both `last.pth` and selected `best.pth` were saved in Drive (45,886,109 bytes each), and training continued into epoch 2.

The preserved epoch-1 checkpoint at `outputs/partial_baseline_epoch001/checkpoint.pth` has SHA-256 `a33601e051a4067470bc6109c570ff323d5e570a0c62892efff67e40cf09d219`, all 965 model tensors, 581 optimizer states and scheduler/RNG recovery data. Its initial tensor digest equals the GPU smoke's saved starting state. It represents 1/200 epochs, not a completed baseline.

Genuine inference on unseen CAMO image `camourflage_00012.jpg` produced finite native 632×420 probability and PNG artifacts. One CPU CLI measurement was 225.3317 ms forward / 271.2917 ms processing; these are single-image observations, not aggregate latency. The app was browser-verified with real masks, overlay, boundary, threshold redraw, matching-ground-truth metrics and a results ZIP. It explicitly labels this checkpoint as partial training. The full automated suite with this checkpoint/image supplied passed all 94 tests in 14.87 seconds. Completion and final quality still depend on full training/evaluation; see `PROGRESS.md` and the live monitor.
