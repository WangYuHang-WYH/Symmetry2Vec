#!/usr/bin/env python3
"""Train one crystallographic-coordination ABX one-hot perovskite fold."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matbench.bench import MatbenchBenchmark


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "matbench_perovskites"
TARGET_COLUMN = "e_form"
REPRESENTATION = "coordination_abx_onehot"
DISPLAY_NAME = "Coordination-role ABX one-hot"
EXPECTED_SAMPLE_COUNT = 18_928
DEFAULT_ROLE_CACHE = (
    BUNDLE_ROOT
    / "splits"
    / "perovskites_coordination_abx"
    / "coordination_abx_tokens.pkl.gz"
)
DEFAULT_GROUPED_SPLIT_DIR = (
    BUNDLE_ROOT / "splits" / "perovskites_composition_grouped_5fold"
)
DEFAULT_GROUPED_TOKEN_CACHE = (
    DEFAULT_GROUPED_SPLIT_DIR / "perovskites_all_tokens.pkl.gz"
)
DEFAULT_OUTPUT_ROOT = (
    BUNDLE_ROOT / "runs" / "perovskites_coordination_abx_10fold_seed42"
)
ABX_MODULE_PATH = (
    BUNDLE_ROOT / "scripts" / "matbench_crabnet_perovskites_abx_onehot_stable.py"
)
GROUPED_MODULE_PATH = (
    BUNDLE_ROOT / "scripts" / "run_perovskites_composition_grouped.py"
)
ABX_ONEHOT = {
    "A": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "B": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "X": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}


def default_crabnet_root() -> Path:
    configured = os.environ.get("CRABNET_ROOT")
    if configured:
        return Path(configured)
    return BUNDLE_ROOT / "crabnet_runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-kind",
        choices=("official", "composition_grouped"),
        required=True,
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--role-cache", type=Path, default=DEFAULT_ROLE_CACHE)
    parser.add_argument(
        "--grouped-split-dir", type=Path, default=DEFAULT_GROUPED_SPLIT_DIR
    )
    parser.add_argument(
        "--grouped-token-cache", type=Path, default=DEFAULT_GROUPED_TOKEN_CACHE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--crabnet-root", type=Path, default=default_crabnet_root())
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for value, name in (
        (args.epochs, "epochs"),
        (args.patience, "patience"),
        (args.batch_size, "batch-size"),
        (args.cpu_threads, "cpu-threads"),
        (args.d_model, "d-model"),
        (args.layers, "layers"),
        (args.heads, "heads"),
    ):
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.d_model % args.heads:
        raise ValueError("--d-model must be divisible by --heads")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if not 0.0 < args.cuda_memory_fraction <= 0.5:
        raise ValueError("--cuda-memory-fraction must be in (0, 0.5]")
    if args.seed != 42:
        raise ValueError("Final-series comparison requires --seed 42")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def import_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.split_kind / f"fold{args.fold}"


def valid_completed(args: argparse.Namespace, role_cache_sha256: str) -> bool:
    directory = output_dir(args)
    paths = (
        directory / "score.csv",
        directory / "predictions.csv",
        directory / "run_audit.json",
    )
    if not all(path.is_file() and path.stat().st_size for path in paths):
        return False
    try:
        audit = json.loads(paths[2].read_text(encoding="utf-8"))
        scores = pd.read_csv(paths[0])
        predictions = pd.read_csv(paths[1], usecols=["fold", "split_kind"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return False
    return bool(
        len(scores) == 1
        and not predictions.empty
        and predictions["fold"].eq(args.fold).all()
        and predictions["split_kind"].eq(args.split_kind).all()
        and audit.get("status") == "completed"
        and audit.get("role_cache_sha256") == role_cache_sha256
    )


def load_role_cache(path: Path) -> tuple[dict, dict, str]:
    path = path.expanduser().resolve()
    manifest_path = path.with_name(path.name + ".manifest.json")
    for required in (path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    cache_hash = sha256_file(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_sha256") != cache_hash:
        raise RuntimeError("Coordination-role cache SHA256 mismatch")
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    lengths = {
        len(payload.get("material_ids", [])),
        len(payload.get("compositions", [])),
        len(payload.get("targets", [])),
        len(payload.get("tokens", [])),
        len(payload.get("role_indices", [])),
    }
    if lengths != {EXPECTED_SAMPLE_COUNT}:
        raise RuntimeError(f"Unexpected role-cache inventory: {sorted(lengths)}")
    if len(set(payload["material_ids"])) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("Role cache contains duplicate material IDs")
    return payload, manifest, cache_hash


def load_official_positions(
    args: argparse.Namespace, role_cache: dict
) -> tuple[dict[str, np.ndarray], dict]:
    benchmark = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = benchmark.tasks_map[TASK_NAME]
    task.load()
    frame = task.df.copy()
    frame.index = frame.index.map(str)
    ids = [str(value) for value in role_cache["material_ids"]]
    if frame.index.tolist() != ids:
        raise RuntimeError("Role cache and Matbench natural material-ID order differ")
    if not np.allclose(
        frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
        np.asarray(role_cache["targets"], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Role cache and Matbench targets differ")

    id_to_position = {material_id: index for index, material_id in enumerate(ids)}
    train_val_structures, _ = task.get_train_and_val_data(args.fold, as_type="tuple")
    test_structures, _ = task.get_test_data(
        args.fold, as_type="tuple", include_target=True
    )
    if not hasattr(train_val_structures, "index") or not hasattr(test_structures, "index"):
        raise RuntimeError("Matbench structures did not preserve material IDs")
    train_val_positions = np.asarray(
        [id_to_position[str(value)] for value in train_val_structures.index],
        dtype=np.int64,
    )
    test_positions = np.asarray(
        [id_to_position[str(value)] for value in test_structures.index],
        dtype=np.int64,
    )
    local_indices = np.arange(len(train_val_positions), dtype=np.int64)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(local_indices)
    n_validation = max(1, int(round(len(local_indices) * args.val_fraction)))
    validation_local = np.sort(local_indices[:n_validation])
    train_local = np.sort(local_indices[n_validation:])
    positions = {
        "train": train_val_positions[train_local],
        "validation": train_val_positions[validation_local],
        "test": test_positions,
    }
    if set(positions["train"]) & set(positions["validation"]):
        raise RuntimeError("Official inner train/validation overlap")
    if (set(positions["train"]) | set(positions["validation"])) & set(test_positions):
        raise RuntimeError("Official train/validation overlaps the test fold")
    metadata = {
        "split_kind": "official_matbench_5fold",
        "inner_validation_method": "row_random_10_percent",
        "inner_validation_seed": args.seed,
        "train_samples": len(positions["train"]),
        "validation_samples": len(positions["validation"]),
        "test_samples": len(positions["test"]),
    }
    return positions, metadata


def load_grouped_positions(
    args: argparse.Namespace, role_cache: dict
) -> tuple[dict[str, np.ndarray], dict]:
    grouped = import_file_module("final_perovskites_grouped", GROUPED_MODULE_PATH)
    helper_args = argparse.Namespace(
        split_dir=args.grouped_split_dir,
        cache=args.grouped_token_cache,
        output_root=args.output_root,
        crabnet_root=args.crabnet_root,
        seed=args.seed,
        fold=args.fold,
        representation=REPRESENTATION,
    )
    data = grouped.load_fixed_data(helper_args)
    if data["material_ids"].tolist() != [str(value) for value in role_cache["material_ids"]]:
        raise RuntimeError("Grouped split and coordination-role cache IDs differ")
    if not np.allclose(
        data["targets"], role_cache["targets"], rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("Grouped split and coordination-role cache targets differ")
    positions = data["positions"]
    metadata = {
        "split_kind": "composition_group_disjoint_stratified_5fold",
        "assignments_sha256": data["assignments_hash"],
        "fold_roles_sha256": data["roles_hash"],
        "grouped_token_cache_sha256": data["cache_hash"],
        "inner_validation_seed": args.seed,
        "train_samples": len(positions["train"]),
        "validation_samples": len(positions["validation"]),
        "test_samples": len(positions["test"]),
        "composition_overlap_count": 0,
    }
    return positions, metadata


def vectorize_tokens(tokens: list, mat2vec: dict[str, np.ndarray]) -> tuple[list, list, Counter]:
    features = []
    fractions = []
    missing = Counter()
    mat_dim = len(next(iter(mat2vec.values())))
    for structure_tokens in tokens:
        structure_features = []
        structure_fractions = []
        for element, role, fraction in structure_tokens:
            element_vector = mat2vec.get(str(element))
            if element_vector is None:
                missing[str(element)] += 1
                element_vector = np.zeros(mat_dim, dtype=np.float32)
            structure_features.append(
                np.concatenate((element_vector, ABX_ONEHOT[str(role)]))
                .astype(np.float32, copy=False)
            )
            structure_fractions.append(float(fraction))
        features.append(np.vstack(structure_features).astype(np.float32, copy=False))
        fractions.append(np.asarray(structure_fractions, dtype=np.float32))
    return features, fractions, missing


def select(values: list, positions: np.ndarray) -> list:
    return [values[int(position)] for position in positions]


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.role_cache = args.role_cache.expanduser().resolve()
    args.grouped_split_dir = args.grouped_split_dir.expanduser().resolve()
    args.grouped_token_cache = args.grouped_token_cache.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.crabnet_root = args.crabnet_root.expanduser().resolve()
    for required in (args.crabnet_root, ABX_MODULE_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    role_cache, role_manifest, role_cache_hash = load_role_cache(args.role_cache)
    directory = output_dir(args)
    if not args.force and not args.preflight and valid_completed(args, role_cache_hash):
        print(f"Skipping completed {args.split_kind} fold {args.fold}", flush=True)
        return

    if args.split_kind == "official":
        positions, split_metadata = load_official_positions(args, role_cache)
    else:
        positions, split_metadata = load_grouped_positions(args, role_cache)

    thread_count = str(args.cpu_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    torch.set_num_threads(args.cpu_threads)
    sys.path.insert(0, str(args.crabnet_root))
    sys.path.insert(0, str(args.crabnet_root / "publication_CrabNet"))
    abx_runner = import_file_module("final_coordination_abx_model", ABX_MODULE_PATH)
    os.chdir(args.crabnet_root)

    seed_everything(args.seed)
    from utils.get_compute_device import get_compute_device
    from utils.utils import count_parameters

    device = get_compute_device(prefer_last=False)
    if torch.device(device).type != "cuda":
        raise RuntimeError("This benchmark run requires a CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=device)

    mat2vec_path = args.crabnet_root / "data" / "element_properties" / "mat2vec.csv"
    mat2vec = abx_runner.load_mat2vec(mat2vec_path)
    all_features, all_fractions, missing_elements = vectorize_tokens(
        role_cache["tokens"], mat2vec
    )
    targets = np.asarray(role_cache["targets"], dtype=np.float32)
    max_tokens = max(len(values) for values in all_fractions)
    loaders = {
        role: abx_runner.make_loader(
            select(all_features, split_positions),
            select(all_fractions, split_positions),
            targets[split_positions],
            max_tokens,
            args.batch_size,
            shuffle=role == "train",
        )
        for role, split_positions in positions.items()
    }

    model = abx_runner.OrbitCrabNet(
        input_dim=len(next(iter(mat2vec.values()))) + 3,
        d_model=args.d_model,
        N=args.layers,
        heads=args.heads,
        compute_device=device,
    ).to(device)
    trainer = abx_runner.Trainer(
        model,
        device,
        model_name=(
            f"perovskites_coordination_abx_{args.split_kind}_fold{args.fold}"
        ),
        discard_n=args.patience,
    )
    parameter_count = int(count_parameters(model))
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / "run_audit.json"
    audit = {
        "schema_version": "1.0.0",
        "status": "preflight" if args.preflight else "running",
        "task": TASK_NAME,
        "representation": REPRESENTATION,
        "display_name": DISPLAY_NAME,
        "split_kind": args.split_kind,
        "fold": args.fold,
        "seed": args.seed,
        "fold_local_rng_reset": True,
        "epochs": args.epochs,
        "patience_validation_checks": args.patience,
        "validation_check_interval_epochs": 2,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "cpu_threads": args.cpu_threads,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "input": "mat2vec element vector plus crystallographic A/B/X role one-hot",
        "input_dimension": len(next(iter(mat2vec.values()))) + 3,
        "mat2vec_dimension": len(next(iter(mat2vec.values()))),
        "abx_dimension": 3,
        "site_symmetry_enabled": False,
        "wyckoff_enabled": False,
        "max_tokens": max_tokens,
        "model_parameter_count": parameter_count,
        "missing_elements": dict(missing_elements),
        "role_cache": str(args.role_cache),
        "role_cache_sha256": role_cache_hash,
        "role_cache_manifest": role_manifest,
        "split_metadata": split_metadata,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(audit, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)

    if args.preflight:
        model.eval()
        batch_features, batch_fractions, _ = next(iter(loaders["train"]))
        with torch.no_grad():
            output = model(
                batch_features.to(device), batch_fractions.to(device)
            )
        if output.ndim != 2 or output.shape[1] != 2:
            raise RuntimeError(f"Unexpected preflight output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("Preflight output contains NaN or Inf")
        audit.update(
            {
                "status": "preflight_passed",
                "preflight_batch_shape": list(batch_features.shape),
                "preflight_output_shape": list(output.shape),
            }
        )
        write_json_atomic(audit, audit_path)
        return

    started = time.time()
    trainer.fit(loaders["train"], loaders["validation"], epochs=args.epochs)
    elapsed = time.time() - started
    validation_mae = float(trainer.evaluate(loaders["validation"]))
    test_mae = float(trainer.evaluate(loaders["test"]))
    actual, predicted = trainer.predict(loaders["test"])
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    test_positions = positions["test"]
    expected_actual = targets[test_positions].astype(np.float64)
    if len(actual) != len(test_positions) or not np.allclose(
        actual, expected_actual, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("Prediction rows are not aligned to test positions")
    recomputed_mae = float(np.mean(np.abs(predicted - actual)))
    if not np.isclose(test_mae, recomputed_mae, rtol=0.0, atol=1e-10):
        raise RuntimeError("Trainer and prediction-table test MAE differ")

    prediction_frame = pd.DataFrame(
        {
            "representation": REPRESENTATION,
            "display_name": DISPLAY_NAME,
            "split_kind": args.split_kind,
            "fold": args.fold,
            "test_split_position": np.arange(len(test_positions), dtype=np.int64),
            "dataset_position": test_positions,
            "material_id": np.asarray(role_cache["material_ids"], dtype=object)[
                test_positions
            ],
            "composition": np.asarray(role_cache["compositions"], dtype=object)[
                test_positions
            ],
            "actual": actual,
            "predicted": predicted,
            "signed_error": predicted - actual,
            "absolute_error": np.abs(predicted - actual),
        }
    )
    write_frame_atomic(prediction_frame, directory / "predictions.csv")
    result = {
        "representation": REPRESENTATION,
        "display_name": DISPLAY_NAME,
        "split_kind": args.split_kind,
        "fold": args.fold,
        "best_epoch": int(trainer.best_epoch),
        "validation_mae": validation_mae,
        "test_mae": test_mae,
        "test_samples": len(test_positions),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "patience_validation_checks": args.patience,
        "batch_size": args.batch_size,
        "input_dimension": audit["input_dimension"],
        "max_tokens": max_tokens,
        "model_parameter_count": parameter_count,
        "training_elapsed_seconds": elapsed,
    }
    write_frame_atomic(pd.DataFrame([result]), directory / "score.csv")
    audit.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "prediction_sha256": sha256_file(directory / "predictions.csv"),
            "score_sha256": sha256_file(directory / "score.csv"),
        }
    )
    write_json_atomic(audit, audit_path)
    print(
        f"{args.split_kind} fold {args.fold}: best_epoch={trainer.best_epoch} "
        f"validation_mae={validation_mae:.6f} test_mae={test_mae:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
