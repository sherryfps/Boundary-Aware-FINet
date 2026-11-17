# Viva preparation

**What does the project do?** It predicts camouflaged foreground pixels and presents probabilities, binary masks, overlays and boundaries. It investigates whether an extra differentiable boundary objective improves the original FINet under matched training conditions.

**What is original work versus prior work?** FINet, its frequency injection modules and decoder come from the cited authors. EfficientNet-B0 is the pretrained spatial backbone. This project adds training/checkpoint recovery, dataset auditing, an optional boundary objective, controlled evaluation and the Gradio application. Do not claim to have invented FINet.

**Why use frequency features?** Block DCT separates coarse variation from finer image detail. The original model injects low/high frequency information into spatial features. Explain this as the model's design motivation; demonstrate effectiveness only with measured experiments.

**What is pretrained?** 3,595,388 retained encoder parameters use torchvision ImageNet weights. The remaining 147,600 FINet parameters start randomly. ImageNet initialization alone is not a camouflage segmentation checkpoint.

**Why not use the authors' full weights?** Both published checkpoint download links were unavailable. The implemented route trains an original FINet checkpoint from the available ImageNet backbone, with provenance recorded.

**Why preserve the original structure-loss quirk?** The upstream scalar BCE reduction causes its spatial weighting to cancel. Changing that while calling the result an original baseline would confound the boundary comparison. The exact function is retained and the discrepancy is documented.

**How is the boundary loss differentiable?** The prediction stays a sigmoid probability. Local maximum and minimum pooling create a boundary magnitude whose gradients can flow through the selected extrema. BCE and Dice compare that map to the target boundary. Do not threshold the predicted probability inside the training objective.

**Does the boundary model need more inference parameters?** No. The extra objective exists only during training. Both inference models have 3,742,988 parameters; any latency difference must still be measured rather than assumed zero.

**Why use the same whole initialization?** Sharing only ImageNet weights leaves random decoder and frequency tensors different. The A/B protocol uses the same complete starting tensor state and checks its digest.

**How do you prevent leakage?** Keep a fixed training-only holdout, group identical image bytes within one partition, audit final-test name/hash overlaps and preserve all exclusions. Never tune lambda or threshold on final-test results.

**Why three metrics?** S-measure assesses structural similarity, MAE measures probability error, and Boundary F-score measures contour agreement under an explicit pixel tolerance. They describe different properties; a better score on one does not imply every aspect improved.

**What did the smoke test prove?** Real batches passed forward, original four-output loss, backward, finite gradients, Adam updates and checkpoint reload/resume. The T4 losses were 6.194746, 6.302223 and 6.236764, with resumed loss 6.085261. These establish execution and recovery, not model quality.

**What if Model B is worse?** Report the measured outcome. Check protocol fairness and possible causes without inventing gains. A negative result under a clear protocol is still a valid experiment.

**How does interruption recovery work?** Save model, Adam, scheduler, completed epoch/global step, random-generator states, configuration and manifest. Resume the latest saved epoch; replay an interrupted unsaved epoch. Do not claim exact mid-epoch recovery.

**What are the practical limits?** Free Colab availability/session limits, a small engineering smoke, possible source-release anomalies, the chosen split/seed and explicit metric definitions. Final accuracy, generalization and the effect of boundary training require completed experiments.

**What can be demonstrated before final training?** Genuine engineering logs, verified checkpoint reload, original architecture/data pipeline, the application's upload/errors and clearly labelled partial training diagnostics. Never show an initialization/smoke mask as successful object detection or quote the authors' benchmark numbers as this project's results.
