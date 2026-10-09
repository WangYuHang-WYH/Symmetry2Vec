from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BUNDLE_ROOT / "vector_registry.json"
DEFAULT_CRABNET_ROOT = Path(
    os.environ.get(
        "CRABNET_ROOT",
        str(BUNDLE_ROOT / "crabnet_runtime"),
    )
)

METHOD_TO_REGISTRY = {
    "kg": "kg",
    "lm": "lm",
    "wren": "wren444",
    "onehot": "onehot_1731",
}

TASKS = {
    "jdft2d": ("matbench_jdft2d", "jdft2d", False),
    "phonons": ("matbench_phonons", "phonons", False),
    "dielectric": ("matbench_dielectric", "dielectric", False),
    "log_gvrh": ("matbench_log_gvrh", "log_gvrh", False),
    "log_kvrh": ("matbench_log_kvrh", "log_kvrh", False),
    "mp_e_form": ("matbench_mp_e_form", "mp_e_form", False),
    "mp_gap": ("matbench_mp_gap", "mp_gap", False),
    "perovskites": (
        "matbench_perovskites",
        "stable_perovskites",
        True,
    ),
}


def load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Missing vector registry: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def representation_paths(method: str) -> tuple[Path, Path | None, dict]:
    registry = load_registry()
    registry_name = METHOD_TO_REGISTRY[method]
    metadata = registry["representations"].get(registry_name)
    if metadata is None:
        raise ValueError(f"Unknown registry representation: {registry_name}")
    table = BUNDLE_ROOT / metadata["path"]
    coverage = BUNDLE_ROOT / metadata["coverage"] if metadata["coverage"] else None
    if not table.is_file():
        raise FileNotFoundError(f"Missing vector table: {table}")
    if coverage is not None and not coverage.is_file():
        raise FileNotFoundError(f"Missing coverage table: {coverage}")
    return table, coverage, metadata


def run_name(method: str, task: str, seed: int) -> str:
    return f"final_{method}_nositesym_seed{seed}_{task}"


def build_command(args: argparse.Namespace, fold: int) -> tuple[list[str], dict]:
    table, coverage, metadata = representation_paths(args.method)
    task_name, target_name, stable_mode = TASKS[args.task]
    fold_script = (
        args.crabnet_root
        / "publication_CrabNet"
        / "final_wp_vector_seeded_fold.py"
    )
    mat2vec = args.crabnet_root / "data" / "element_properties" / "mat2vec.csv"
    token_cache_root = (
        args.token_cache_root.resolve()
        if args.token_cache_root is not None
        else args.crabnet_root
        / "artifacts"
        / "feature_cache"
        / f"final_fold_seed{args.seed}"
    )
    token_cache_dir = token_cache_root / args.task
    for required in (fold_script, mat2vec):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required CrabNet file: {required}")

    command = [
        sys.executable,
        "-u",
        str(fold_script),
        "--fold-seed",
        str(args.seed),
        "--task-name",
        task_name,
        "--target-name",
        target_name,
        "--start-fold",
        str(fold),
        "--folds",
        "1",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--val-fraction",
        str(args.val_fraction),
        "--random-state",
        str(args.seed),
        "--entity-embeddings",
        str(table),
        "--disable-site-symmetry",
        "--mat2vec-path",
        str(mat2vec),
        "--run-name",
        run_name(args.method, args.task, args.seed),
        "--token-cache-dir",
        str(token_cache_dir),
    ]
    if args.cuda_memory_fraction is not None:
        command.extend(["--cuda-memory-fraction", str(args.cuda_memory_fraction)])
    if coverage is not None:
        command.extend(
            [
                "--wp-training-coverage",
                str(coverage),
                "--drop-uncovered-wp-samples",
            ]
        )
    if stable_mode:
        command.append("--stable-mode")

    audit = {
        "method": args.method,
        "registry_representation": METHOD_TO_REGISTRY[args.method],
        "representation_dimension": metadata["dimension"],
        "task": args.task,
        "fold": fold,
        "seed": args.seed,
        "validation_split_seed": args.seed,
        "site_symmetry": False,
        "stable_mode": stable_mode,
        "coverage_filter": coverage is not None,
        "token_cache_dir": str(token_cache_dir),
        "token_cache_exists": token_cache_dir.is_dir(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": shlex.join(command),
    }
    return command, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run any KG, LM, Wren, or one-hot Matbench task."
        )
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_TO_REGISTRY),
        required=True,
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument(
        "--fold",
        choices=("all", "0", "1", "2", "3", "4"),
        default="all",
        help="Matbench fold to run; default: all five folds sequentially.",
    )
    parser.add_argument("--crabnet-root", type=Path, default=DEFAULT_CRABNET_ROOT)
    parser.add_argument(
        "--token-cache-root",
        type=Path,
        help=(
            "Root for embedding-independent orbit-token caches. Defaults to "
            "CRABNET_ROOT/artifacts/feature_cache/final_fold_seed{seed}."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.cpu_threads) < 1:
        raise ValueError(
            "epochs, patience, batch-size, and cpu-threads must be positive"
        )
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0, 1)")
    if args.cuda_memory_fraction is not None and not (
        0.0 < args.cuda_memory_fraction <= 0.5
    ):
        raise ValueError("cuda-memory-fraction must be in (0, 0.5] for dual workers")
    args.crabnet_root = args.crabnet_root.resolve()
    folds = range(5) if args.fold == "all" else (int(args.fold),)
    child_env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_env[variable] = str(args.cpu_threads)
    for fold in folds:
        command, audit = build_command(args, fold)
        audit["cpu_threads_per_process"] = args.cpu_threads
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(
            command,
            cwd=args.crabnet_root,
            env=child_env,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
