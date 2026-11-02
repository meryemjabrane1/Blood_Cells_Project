#!/usr/bin/env python3
"""Upload the dataset privately and submit or monitor its Kaggle GPU notebook."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "blood-smear-yolo26-data"
KERNEL_SLUG = "blood-smear-yolo26-fine-tuning"
KAGGLE_STAGE = PROJECT_ROOT / ".kaggle" / "staging"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_kaggle_environment() -> tuple[dict[str, str], str | None]:
    env_values: dict[str, str] = {}
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            env_values[name.strip()] = value
    env = os.environ.copy()
    env.update(env_values)
    if not env.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN is missing from .env or the environment.")
    return env, env_values.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")


def run_kaggle(
    args: list[str], env: dict[str, str], *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["kaggle", *args],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=capture,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Kaggle command failed ({' '.join(args)}): {detail}")
    return result


def discover_username(env: dict[str, str], configured: str | None) -> str:
    if configured:
        return configured
    response = run_kaggle(
        ["kernels", "list", "--mine", "--format", "json", "--page-size", "1"],
        env,
        capture=True,
    )
    kernels = json.loads(response.stdout)
    if kernels:
        return str(kernels[0]["ref"]).split("/", maxsplit=1)[0]
    raise RuntimeError(
        "Could not infer the Kaggle username. Add KAGGLE_USERNAME to .env "
        "or pass --username."
    )


def build_dataset_stage(username: str) -> Path:
    dataset_root = PROJECT_ROOT / "TXL-PBC"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    stage = KAGGLE_STAGE / DATASET_SLUG
    stage.mkdir(parents=True, exist_ok=True)
    archive_path = stage / "dataset_payload.zip"
    source_root = PROJECT_ROOT / "src" / "blood_cells"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Training source not found: {source_root}")
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_root).as_posix())
        for path in sorted(source_root.rglob("*.py")):
            relative = Path("project_src") / "blood_cells" / path.relative_to(source_root)
            archive.write(path, relative.as_posix())

    metadata = {
        "title": "Private Blood Smear YOLO26 Training Data",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private training input for the blood-cell YOLO26 project. "
            "The original source license was not provided."
        ),
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return stage


def create_or_version_dataset(
    username: str, stage: Path, env: dict[str, str], *, reuse_existing: bool = False
) -> None:
    dataset_ref = f"{username}/{DATASET_SLUG}"
    response = run_kaggle(
        ["datasets", "list", "--mine", "--format", "json", "--page-size", "200"],
        env,
        capture=True,
    )
    existing = json.loads(response.stdout)
    if any(item.get("ref") == dataset_ref for item in existing):
        if reuse_existing:
            wait_for_dataset(dataset_ref, env)
            return
        run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                "Update private YOLO26 training data and source.",
                "--keep-tabular",
                "-q",
            ],
            env,
        )
    else:
        run_kaggle(
            ["datasets", "create", "-p", str(stage), "--keep-tabular", "-q"],
            env,
        )
    wait_for_dataset(dataset_ref, env)


def wait_for_dataset(dataset_ref: str, env: dict[str, str]) -> None:
    deadline = time.monotonic() + 600
    while True:
        result = run_kaggle(
            ["datasets", "status", dataset_ref], env, capture=True
        )
        status = result.stdout.strip().lower()
        if status == "ready":
            return
        if "error" in status or "fail" in status:
            raise RuntimeError(f"Kaggle dataset processing failed: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Kaggle dataset did not become ready within 10 minutes: {status}"
            )
        print(f"Waiting for Kaggle dataset processing ({status or 'pending'})...")
        time.sleep(10)


def write_kernel_metadata(username: str) -> Path:
    kernel_dir = PROJECT_ROOT / "kaggle"
    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "Blood Smear YOLO26 Fine-tuning",
        "code_file": "train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def atomic_progress(payload: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=REPORTS_DIR, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, REPORTS_DIR / "training_progress.json")


def copy_downloaded_reports() -> None:
    remote_reports = PROJECT_ROOT / "outputs" / "reports"
    if not remote_reports.is_dir():
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in remote_reports.iterdir():
        if path.is_file():
            shutil.copy2(path, REPORTS_DIR / path.name)


def download_outputs(username: str, env: dict[str, str]) -> None:
    kernel_ref = f"{username}/{KERNEL_SLUG}"
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    run_kaggle(
        ["kernels", "output", kernel_ref, "-p", str(PROJECT_ROOT / "outputs"), "-o"],
        env,
    )
    copy_downloaded_reports()


def watch_training(username: str, env: dict[str, str]) -> None:
    kernel_ref = f"{username}/{KERNEL_SLUG}"
    epoch_pattern = re.compile(r"(?<!\d)(\d{1,3})\s*/\s*(\d{1,3})(?!\d)")
    last_epoch = 0
    deadline = time.monotonic() + 12 * 60 * 60
    initial_status = run_kaggle(
        ["kernels", "status", kernel_ref], env, capture=True
    ).stdout.upper()
    atomic_progress(
        {
            "status": "training" if "RUNNING" in initial_status else "queued",
            "epoch": 0,
            "total_epochs": 100,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    while True:
        process = subprocess.Popen(
            ["kaggle", "kernels", "logs", "-f", kernel_ref],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not read the Kaggle notebook log stream.")
        for line in process.stdout:
            print(line, end="", flush=True)
            match = epoch_pattern.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if match:
                epoch, total = (int(value) for value in match.groups())
                if 0 < epoch <= total == 100 and epoch > last_epoch:
                    last_epoch = epoch
                    atomic_progress(
                        {
                            "status": "training",
                            "epoch": epoch,
                            "total_epochs": total,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        return_code = process.wait()
        status_result = run_kaggle(
            ["kernels", "status", kernel_ref], env, capture=True
        )
        status_text = status_result.stdout.strip()
        status_match = re.search(
            r"KernelWorkerStatus\.([A-Z_]+)|status \"([A-Z_]+)\"",
            status_text,
            re.IGNORECASE,
        )
        status = next(
            (value for value in status_match.groups() if value), "UNKNOWN"
        ).upper() if status_match else "UNKNOWN"
        if status in {"COMPLETE", "COMPLETE_WITH_WARNINGS"}:
            if return_code:
                print(f"Kaggle log stream exited with status {return_code}.")
            download_outputs(username, env)
            return
        if status in {"ERROR", "FAILED", "CANCELLED"}:
            error = f"Kaggle run ended with status {status}: {status_text}"
            atomic_progress(
                {
                    "status": "failed",
                    "epoch": last_epoch,
                    "total_epochs": 100,
                    "error": error,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            raise RuntimeError(error)
        if return_code and status not in {"RUNNING", "QUEUED", "PENDING", "STARTING"}:
            error = f"Kaggle log stream exited with status {return_code}: {status_text}"
            atomic_progress(
                {
                    "status": "failed",
                    "epoch": last_epoch,
                    "total_epochs": 100,
                    "error": error,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            raise RuntimeError(error)
        atomic_progress(
            {
                "status": "training" if status == "RUNNING" else "queued",
                "epoch": last_epoch,
                "total_epochs": 100,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("Stopped waiting for Kaggle after 12 hours.")
        print(f"Notebook remains {status.lower()}; reconnecting to its live logs.")
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="Show the latest Kaggle run status.")
    group.add_argument("--download", action="store_true", help="Download the latest run outputs.")
    group.add_argument("--watch", action="store_true", help="Follow live logs and refresh dashboard reports.")
    parser.add_argument("--username", help="Kaggle username; otherwise inferred from an existing notebook.")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="Reuse an existing dataset version when its data and source are unchanged.",
    )
    args = parser.parse_args()
    env, configured_username = load_kaggle_environment()
    username = discover_username(env, args.username or configured_username)
    kernel_ref = f"{username}/{KERNEL_SLUG}"

    if args.status:
        run_kaggle(["kernels", "status", kernel_ref], env)
        return
    if args.download:
        download_outputs(username, env)
        print(f"Downloaded Kaggle outputs to {PROJECT_ROOT / 'outputs'}")
        return
    if args.watch:
        watch_training(username, env)
        print(f"Reports synced to {REPORTS_DIR}")
        return

    stage = build_dataset_stage(username)
    create_or_version_dataset(username, stage, env, reuse_existing=args.reuse_dataset)
    write_kernel_metadata(username)
    run_kaggle(["kernels", "push", "-p", str(PROJECT_ROOT / "kaggle")], env)
    print(f"Submitted private GPU notebook: https://www.kaggle.com/code/{kernel_ref}")
    print("Use --watch to stream epoch logs and refresh the local dashboard.")


if __name__ == "__main__":
    main()
