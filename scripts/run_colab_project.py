"""Sequential, resumable Colab execution of the authorized FINet research protocol.

This supervises real training processes. It does not keep a runtime alive, alter
Colab limits, change the original training loop, or substitute partial results.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.finet_adapter import extract_state_dict, state_dict_sha256
from src.training import TrainingConfig, run_signature
from src.training_data import file_hash, load_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint_summary(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {key: payload.get(key) for key in (
        "training_status", "completed_epochs", "global_step", "best_validation_mae",
        "initial_model_sha256", "split_fingerprint", "run_signature", "config")}


def is_finished(summary, config):
    expected = "boundary_training" if config.boundary_lambda else "baseline_training"
    return (summary["training_status"] == expected
            and summary["completed_epochs"] == config.epochs
            and summary["config"]["epochs"] == config.epochs)


def matched_boundary_config(baseline, output, shared_initial, coefficient=0.1):
    result = {**baseline, "output_dir": str(output),
              "initial_checkpoint": str(shared_initial), "boundary_lambda": coefficient}
    TrainingConfig(**result).validate()
    if coefficient <= 0:
        raise ValueError("The boundary comparison requires a positive coefficient.")
    allowed = {"output_dir", "initial_checkpoint", "boundary_lambda"}
    differences = {key for key in baseline.keys() | result.keys()
                   if baseline.get(key) != result.get(key)}
    if not differences <= allowed:
        raise ValueError(f"Unexpected A/B configuration differences: {differences - allowed}")
    return result


@contextmanager
def runtime_lock(identity):
    # Local Linux lock: Drive FUSE does not reliably support advisory locks.
    import fcntl
    name = hashlib.sha256(str(identity).encode()).hexdigest()[:20]
    with (Path(tempfile.gettempdir()) / f"finet-project-{name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This project runner is already active in this runtime.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class ProjectRunner:
    def __init__(self, config, smoke, destination):
        self.baseline = TrainingConfig(**read_json(config))
        self.baseline.validate()
        if self.baseline.boundary_lambda:
            raise ValueError("Start with the original FINet baseline configuration.")
        if self.baseline.save_every_epochs != 1:
            raise ValueError("This runner requires saving every epoch for recovery.")
        self.baseline_path = Path(config)
        self.baseline_smoke = Path(smoke)
        self.destination = Path(destination)
        self.progress = self.destination / "project_status.json"
        self.manifest = load_manifest(self.baseline.split_manifest, self.baseline.data_root)
        self.shared_initial = Path(self.baseline.initial_checkpoint)
        payload = torch.load(self.shared_initial, map_location="cpu", weights_only=True)
        self.initial_digest = state_dict_sha256(extract_state_dict(payload))
        self.state = {"status": "prepared", "stages": {}, "shared_initial_model_sha256": self.initial_digest}
        if self.progress.exists():
            previous = read_json(self.progress)
            if previous["shared_initial_model_sha256"] != self.initial_digest:
                raise ValueError("Existing project state uses a different full-model initialization.")
            self.state = previous

    def status(self, stage, status, **details):
        self.state.update(status=status, current_stage=stage,
                          observed_at_utc=datetime.now(timezone.utc).isoformat())
        self.state["stages"][stage] = {"status": status, **details}
        record(self.progress, self.state)
        print(json.dumps({"stage": stage, "status": status, **details}), flush=True)

    def command(self, stage, arguments):
        self.status(stage, "running")
        log = self.destination / "logs" / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            process = subprocess.Popen([sys.executable, "-u", *map(str, arguments)], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            try:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise RuntimeError(f"Stage {stage} exited with code {code}; inspect {log}.")

    def train(self, stage, config_path, smoke):
        config = TrainingConfig(**read_json(config_path))
        proof = read_json(smoke)
        if (proof["status"] != "passed"
                or proof["run_signature"] != run_signature(config, self.manifest)
                or proof["hardware"]["gpu"] != torch.cuda.get_device_name(0)
                or proof["initial_model_sha256"] != self.initial_digest
                or file_hash(Path(config.initial_checkpoint)) != proof["initialization_sha256"]):
            raise ValueError(f"{stage}: smoke/configuration/GPU/shared initialization mismatch.")
        latest = Path(config.output_dir) / "last.pth"
        arguments = ["train_baseline.py", "--config", config_path, "--mode", "train",
                     "--start-full-training", "--verified-smoke-report", smoke]
        if latest.is_file():
            summary = checkpoint_summary(latest)
            if (summary["run_signature"] != run_signature(config, self.manifest)
                    or summary["initial_model_sha256"] != self.initial_digest):
                raise ValueError(f"{stage}: saved checkpoint belongs to a different experiment.")
            if is_finished(summary, config):
                self.status(stage, "complete", completed_epochs=config.epochs)
                return latest
            arguments += ["--resume", latest]
        self.command(stage, arguments)
        summary = checkpoint_summary(latest)
        if not is_finished(summary, config) or summary["initial_model_sha256"] != self.initial_digest:
            raise RuntimeError(f"{stage} exited without a full-budget checkpoint.")
        self.status(stage, "complete", completed_epochs=summary["completed_epochs"],
                    best_validation_mae=summary["best_validation_mae"])
        return latest

    def boundary(self):
        output = Path(self.baseline.output_dir).with_name("boundary_seed123456_b4_v1")
        config_path = self.destination / "boundary_config.json"
        smoke_dir = output.with_name(output.name + "_smoke_v1")
        smoke_config = matched_boundary_config(vars(self.baseline), smoke_dir, self.shared_initial)
        smoke_path = self.destination / "boundary_smoke_config.json"
        record(smoke_path, smoke_config)
        proof_path = smoke_dir / "smoke_report.json"
        if not proof_path.is_file():
            if smoke_dir.exists() and any(smoke_dir.iterdir()):
                # An interrupted smoke is preserved; no evidence is overwritten.
                counter = 2
                while smoke_dir.with_name(output.name + f"_smoke_v{counter}").exists():
                    counter += 1
                smoke_dir = smoke_dir.with_name(output.name + f"_smoke_v{counter}")
                smoke_config["output_dir"] = str(smoke_dir)
                record(smoke_path, smoke_config)
                proof_path = smoke_dir / "smoke_report.json"
            self.command("boundary_smoke", ["train_baseline.py", "--config", smoke_path,
                                            "--mode", "smoke", "--steps", "3"])
        proof = read_json(proof_path)
        if proof["initial_model_sha256"] != self.initial_digest:
            raise ValueError("Boundary smoke changed the shared initial full-model tensors.")
        config = matched_boundary_config(vars(self.baseline), output, smoke_dir / "initialization.pth")
        record(config_path, config)
        self.status("boundary_smoke", "complete", losses=[row["loss"] for row in proof["steps"]],
                    report=str(proof_path))
        return self.train("boundary", config_path, proof_path)

    def evaluate(self, baseline, boundary, test_root=None):
        self.command("evaluation_dependencies", ["-m", "pip", "install", "-r", "requirements-evaluation.txt"])
        self.command("evaluation_dependency_check", ["-m", "pip", "check"])
        drive_root = self.destination.parents[1]
        archive = drive_root / "data" / "COD-TestDataset.zip"
        test_root = Path(test_root or "/content/finet_test_data")
        self.command("test_data_audit", ["scripts/prepare_test_data.py", "--archive", archive,
                     "--data-root", test_root, "--training-manifest", self.baseline.split_manifest,
                     "--report", self.destination / "test_data_audit.json", "--download"])
        # Fixed-budget last checkpoints are the primary controlled comparison.
        # Validation-selected best checkpoints are exported separately for the app.
        from evaluate_models import evaluate, write_comparison
        torch.set_num_threads(self.baseline.cpu_threads)
        all_reports = {}
        for name in ("CAMO", "CHAMELEON", "COD10K", "NC4K"):
            reports = []
            for model_name, checkpoint in (("original", baseline), ("boundary", boundary)):
                stage = f"evaluate_{name}_{model_name}"
                target = self.destination / "evaluation" / name / model_name
                digest = file_hash(checkpoint)
                saved = self.state["stages"].get(stage, {})
                if saved.get("status") == "complete" and saved.get("checkpoint_sha256") == digest:
                    report = read_json(saved["summary"])
                else:
                    if target.exists() and any(target.iterdir()):
                        target = target.with_name(model_name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
                    self.status(stage, "running")
                    report = evaluate(checkpoint, test_root / name, self.baseline.split_manifest,
                                      target, self.baseline.device)
                    self.status(stage, "complete", summary=str(target / "summary.json"),
                                checkpoint_sha256=digest, metrics=report["metrics"])
                reports.append(report)
            audit = write_comparison(*reports, self.destination / "evaluation" / name)
            if not audit["controlled_protocol_verified"]:
                raise RuntimeError(f"{name}: controlled comparison audit failed; inspect comparison.json.")
            all_reports[name] = reports
        lines = ["# Measured FINet results", "", "Primary comparison: both final, full-budget checkpoints.",
                 "Original FINet architecture and structure loss credited to the upstream authors.",
                 "Single-seed measurements; statistical significance is not established.", "",
                 "| Audited subset | Model | Images | S-measure | MAE | Boundary F1 | Median forward ms |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for name, reports in all_reports.items():
            for report in reports:
                m = report["metrics"]
                lines.append(f"| {name} | {report['model']} | {report['sample_count']} | {m['s_measure']:.6f} | "
                             f"{m['mae']:.6f} | {m['boundary_f1']:.6f} | {report['inference_latency_ms']['median']:.3f} |")
        lines += ["", "Two CHAMELEON geometry mismatches and development overlaps are excluded by the saved audit.",
                  "See per-image CSVs, protocol hashes, comparison audits and actual sample outputs for details."]
        (self.destination / "MEASURED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self, include_evaluation=True):
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA GPU is required; do not launch long local CPU training.")
        with runtime_lock(self.destination):
            try:
                baseline = self.train("baseline", self.baseline_path, self.baseline_smoke)
                boundary = self.boundary()
                for name, checkpoint in (("baseline", baseline), ("boundary", boundary)):
                    target = self.destination / "models" / name
                    target.mkdir(parents=True, exist_ok=True)
                    for filename in ("best.pth", "last.pth"):
                        shutil.copy2(checkpoint.with_name(filename), target / filename)
                    record(target / "training_completion.json", checkpoint_summary(checkpoint))
                if include_evaluation:
                    self.evaluate(baseline, boundary)
                self.status("project", "experiments_complete" if include_evaluation else "training_complete",
                            note="Gradio integration and final human-facing deliverables still require verification.")
            except BaseException as exc:
                self.status(self.state.get("current_stage", "setup"), "interrupted", error=str(exc))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--baseline-smoke", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-project", action="store_true", help="Explicitly authorize the long sequential GPU stages")
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    if not args.run_project:
        parser.error("Review the tested GPU configuration, then supply --run-project.")
    ProjectRunner(args.config, args.baseline_smoke, args.output).run(not args.training_only)


if __name__ == "__main__":
    main()
