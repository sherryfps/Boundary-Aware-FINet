# Start here on the new laptop

Back up and copy this **entire FINet_PROJECT_BACKUP_2026-11-14 folder**, including its `project/data`, `project/outputs` and `project/checkpoints` directories. They are not in Git, but they contain the datasets, trained weights and recovery evidence. Allow at least **30 GB** for this copy, plus space for the new Python environment. This is a folder backup, not a new training run.

1. Install normal 64-bit **Python 3.12** if needed. In the backup folder, run `py -3.12 VERIFY_BACKUP.py`. This checks every saved file against the supplied SHA-256 manifest. It uses only Python's standard library; no project dependencies are needed yet.
2. Open the **`project` subfolder** as the workspace in Codex. Paste the text from `CONTINUE_WITH_CODEX.txt`. The short `AGENTS.md` directs Codex to the complete `CODEX_HANDOFF.md`; no old chat history is required for that written handoff. [Official OpenAI documentation explains project AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
3. In `project`, run **`SETUP_NEW_LAPTOP.cmd`**. It creates `.venv`, installs the exact captured CPU packages, checks their dependencies and verifies both genuine final models on a real image. It uses an existing working project environment, the system Python 3.12 launcher, or Codex's bundled 64-bit Python 3.12 when the launcher is unavailable. Normal internet access is needed to install packages. It does not train a model or change saved checkpoint files. Preserve any copied old `.venv` under another name before setup, because virtual environments contain machine-specific paths. Preserved `.venv-*` folders are ignored by Git and excluded from future transfer bundles. Automated runs may pass `--no-pause` to the setup launcher.
4. Double-click **`START_FINET.bat`**. It starts the complete local app and opens **http://127.0.0.1:7860** after startup succeeds. It reuses a running FINet instance and saves errors under `outputs/app_logs/`; the server stays running when the launcher closes. `START_APP.cmd` forwards to the same launcher. Both camouflage models should show **200/200 epochs**. The industrial mode remains separate and pending until its actual trained weights are retrieved and verified.

The setup/launcher paths work relative to the folder, so your Windows username and drive letter can differ. Historical absolute paths inside logs and checkpoint metadata are retained as evidence; do not replace them globally. `requirements-transfer-cpu.txt` records the actual current package versions, which supersede the older historical CPU lock for this migration. `model_cache/torch` contains the original ImageNet B0 download; the launcher sets `TORCH_HOME` to that local cache.

## What is included

- All local project source, original model code, configs, notebooks, scripts, tests, docs, Git history and uncommitted work.
- All local original/prepared camouflage and industrial datasets, download archives and splits.
- Both verified final camouflage epoch-200 states, all independent training/recovery snapshots, optimizer/scheduler/RNG state, logs, screenshots, charts and sample outputs.
- The original pasted project requests in `project/handoff/reference_requests`, and current decisions/cloud URLs in `project/CODEX_HANDOFF.md`.
- Current CPU dependency lock, environment snapshot, setup/app launchers and complete per-file SHA-256 verification tools.

## What still depends on cloud access

Camouflage training is **complete**. The existing Kaggle **evaluation Version 6 (`369384835`)** was submitted and passed real CUDA/preflight checks; its final aggregate result was not yet independently downloaded. The industrial **Version 4 (`369287259`)** may contain more progress than the independently backed-up local epoch **96**. Codex must inspect those **existing jobs** and download their latest valid saved outputs before any resumption. Never blindly restart industrial from 96, and never retrain the completed camouflage epochs.

You will need to sign into Codex and the authorized Kaggle accounts normally on the new laptop. Browser profiles, passwords, authentication tokens and the global `.codex` directory are not part of this project backup. A running local server/browser session also does not move. The handoff supplies the project context and account/notebook locations, not authentication.

## Backup proof

`manifest.jsonl` lists the size and SHA-256 of every packaged file. `TRANSFER_COMPLETE.json` records the actual final counts, size, manifest digest and complete-copy verification status. Do not rely on a folder that lacks a successful completion report. The verification program is also preserved in `project/scripts/verify_transfer_bundle.py`.

Keep the original laptop copy until the new laptop passes verification and real-model inference.
