#!/usr/bin/env python3
"""Run ten coordination-role ABX folds through eight persistent workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = BUNDLE_ROOT / "scripts" / "run_perovskites_coordination_abx.py"
DEFAULT_OUTPUT_ROOT = (
    BUNDLE_ROOT / "runs" / "perovskites_coordination_abx_10fold_seed42"
)

# Ten jobs distributed across eight simultaneous processes.  Workers 0 and 1
# receive one follow-up job; the other six receive one job each.
ASSIGNMENTS = {
    0: [("official", 0), ("composition_grouped", 0)],
    1: [("official", 1), ("composition_grouped", 1)],
    2: [("official", 2)],
    3: [("official", 3)],
    4: [("official", 4)],
    5: [("composition_grouped", 2)],
    6: [("composition_grouped", 3)],
    7: [("composition_grouped", 4)],
}


def default_crabnet_root() -> Path:
    configured = os.environ.get("CRABNET_ROOT")
    if configured:
        return Path(configured)
    return BUNDLE_ROOT / "crabnet_runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", type=int, choices=range(8), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--crabnet-root", type=Path, default=default_crabnet_root())
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_job(args: argparse.Namespace, split_kind: str, fold: int) -> dict:
    command = [
        sys.executable,
        str(RUNNER),
        "--split-kind",
        split_kind,
        "--fold",
        str(fold),
        "--output-root",
        str(args.output_root),
        "--crabnet-root",
        str(args.crabnet_root),
        "--epochs",
        "1000",
        "--patience",
        "100",
        "--batch-size",
        "128",
        "--seed",
        "42",
        "--cpu-threads",
        str(args.cpu_threads),
        "--cuda-memory-fraction",
        str(args.cuda_memory_fraction),
        "--d-model",
        "512",
        "--layers",
        "3",
        "--heads",
        "4",
    ]
    if args.force:
        command.append("--force")
    log_path = args.output_root / "logs" / "jobs" / f"{split_kind}_fold{fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = str(args.cpu_threads)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        header = f"\n[{started}] START {' '.join(command)}\n"
        log_handle.write(header)
        print(header.rstrip(), flush=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            print(line, end="", flush=True)
        return_code = process.wait()
        finished = datetime.now(timezone.utc).isoformat()
        footer = f"[{finished}] END return_code={return_code}\n"
        log_handle.write(footer)
        print(footer.rstrip(), flush=True)
    return {
        "split_kind": split_kind,
        "fold": fold,
        "return_code": return_code,
        "log": str(log_path),
        "started_at_utc": started,
        "finished_at_utc": finished,
    }


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    if not 0.0 < args.cuda_memory_fraction <= 0.5:
        raise ValueError("--cuda-memory-fraction must be in (0, 0.5]")
    args.output_root = args.output_root.expanduser().resolve()
    args.crabnet_root = args.crabnet_root.expanduser().resolve()
    jobs = ASSIGNMENTS[args.worker_id]
    status_path = args.output_root / "logs" / f"worker{args.worker_id}_status.json"
    status = {
        "schema_version": "1.0.0",
        "worker_id": args.worker_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "jobs_assigned": [
            {"split_kind": split_kind, "fold": fold}
            for split_kind, fold in jobs
        ],
        "training_defaults": {
            "seed_each_fold": 42,
            "epochs": 1000,
            "patience_validation_checks": 100,
            "batch_size": 128,
            "d_model": 512,
            "layers": 3,
            "heads": 4,
            "stable_perovskites_mode": True,
            "site_symmetry_channel": False,
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    write_json_atomic(status, status_path)
    failures = 0
    for split_kind, fold in jobs:
        result = run_job(args, split_kind, fold)
        status["jobs"].append(result)
        failures += int(result["return_code"] != 0)
        write_json_atomic(status, status_path)
    status.update(
        {
            "status": "completed" if failures == 0 else "completed_with_failures",
            "failure_count": failures,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(status, status_path)
    if failures:
        raise SystemExit(f"Worker {args.worker_id} had {failures} failed jobs")


if __name__ == "__main__":
    main()
