from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
RUN_FOLD = Path(__file__).resolve().parent / "run_matbench.py"
DEFAULT_CRABNET_ROOT = Path(
    os.environ.get(
        "CRABNET_ROOT",
        str(BUNDLE_ROOT / "crabnet_runtime"),
    )
)

# Workers 0/1 map to GPU 0 and workers 2/3 map to GPU 1.
WORK_QUEUES = {
    0: [
        *(('mp_e_form', fold) for fold in (0, 2, 4)),
        *(('log_kvrh', fold) for fold in range(5)),
    ],
    1: [
        *(('mp_gap', fold) for fold in (1, 3)),
        *(('dielectric', fold) for fold in range(5)),
        *(('jdft2d', fold) for fold in range(5)),
    ],
    2: [
        *(('mp_e_form', fold) for fold in (1, 3)),
        *(('log_gvrh', fold) for fold in range(5)),
        *(('perovskites', fold) for fold in range(5)),
    ],
    3: [
        *(('mp_gap', fold) for fold in (0, 2, 4)),
        *(('phonons', fold) for fold in range(5)),
    ],
}

ALL_TASKS = {
    "jdft2d",
    "phonons",
    "dielectric",
    "log_gvrh",
    "log_kvrh",
    "mp_e_form",
    "mp_gap",
    "perovskites",
}

METHODS = {"kg", "lm", "wren", "onehot"}


def validate_work_queues() -> None:
    jobs = [job for queue in WORK_QUEUES.values() for job in queue]
    expected = {(task, fold) for task in ALL_TASKS for fold in range(5)}
    if len(jobs) != 40 or len(set(jobs)) != 40 or set(jobs) != expected:
        raise RuntimeError("Work queues must contain each of 8 tasks x 5 folds exactly once")


def run_name(method: str, task: str, seed: int) -> str:
    return f"final_{method}_nositesym_seed{seed}_{task}"


def score_path(args: argparse.Namespace, task: str, fold: int) -> Path:
    return (
        args.crabnet_root
        / "artifacts"
        / run_name(args.method, task, args.seed)
        / f"fold_scores_start{fold}_n1.csv"
    )


def prediction_path(args: argparse.Namespace, task: str, fold: int) -> Path:
    target = "stable_perovskites" if task == "perovskites" else task
    return (
        args.crabnet_root
        / "publication_predictions"
        / f"{run_name(args.method, task, args.seed)}_predictions"
        / f"{target}_test_cv{fold}.csv"
    )


def valid_completed(args: argparse.Namespace, task: str, fold: int) -> bool:
    score = score_path(args, task, fold)
    prediction = prediction_path(args, task, fold)
    if not score.is_file() or not prediction.is_file():
        return False
    if score.stat().st_size == 0 or prediction.stat().st_size == 0:
        return False
    with score.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 1 and int(rows[0]["fold"]) == fold


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one final eight-task per-fold work queue."
    )
    parser.add_argument("--worker", type=int, choices=sorted(WORK_QUEUES), required=True)
    parser.add_argument(
        "--method",
        choices=sorted(METHODS),
        required=True,
    )
    parser.add_argument("--crabnet-root", type=Path, default=DEFAULT_CRABNET_ROOT)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    return parser.parse_args()


def main() -> None:
    validate_work_queues()
    args = parse_args()
    args.crabnet_root = args.crabnet_root.resolve()
    log_root = BUNDLE_ROOT / "runs" / args.method / f"worker{args.worker}"
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = log_root / "status.json"
    status: dict[str, str] = {}
    print(
        json.dumps(
            {
                "worker": args.worker,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "jobs": WORK_QUEUES[args.worker],
                "method": args.method,
                "site_symmetry": False,
                "fold_seed": args.seed,
                "validation_split_seed": args.seed,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "cpu_threads_per_process": args.cpu_threads,
            },
            indent=2,
        ),
        flush=True,
    )

    for task, fold in WORK_QUEUES[args.worker]:
        key = f"{task}_fold{fold}"
        if valid_completed(args, task, fold):
            status[key] = "skipped_complete"
            write_status(status_path, status)
            print(f"Skipping completed {key}", flush=True)
            continue

        task_log_dir = log_root / task
        task_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_log_dir / f"fold{fold}.log"
        command = [
            sys.executable,
            "-u",
            str(RUN_FOLD),
            "--method",
            args.method,
            "--task",
            task,
            "--fold",
            str(fold),
            "--crabnet-root",
            str(args.crabnet_root),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--val-fraction",
            str(args.val_fraction),
            "--seed",
            str(args.seed),
            "--cpu-threads",
            str(args.cpu_threads),
            "--cuda-memory-fraction",
            str(args.cuda_memory_fraction),
        ]
        status[key] = "running"
        write_status(status_path, status)
        print(f"Starting {key}; log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\nCOMMAND: " + " ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=args.crabnet_root,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            status[key] = f"failed_returncode_{completed.returncode}"
            write_status(status_path, status)
            print(f"Failed {key}: returncode={completed.returncode}", flush=True)
            raise SystemExit(completed.returncode)
        if not valid_completed(args, task, fold):
            status[key] = "failed_missing_outputs"
            write_status(status_path, status)
            print(f"Failed {key}: expected outputs are missing", flush=True)
            raise SystemExit(1)
        status[key] = "complete"
        write_status(status_path, status)
        print(f"Completed {key}", flush=True)


if __name__ == "__main__":
    main()
