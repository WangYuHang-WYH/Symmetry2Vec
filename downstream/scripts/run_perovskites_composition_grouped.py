#!/usr/bin/env python3
"""Train one method on one fixed composition-disjoint perovskites fold."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
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

import run_crabnet_original as original


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = BUNDLE_ROOT / "vector_registry.json"
ABX_MODULE_PATH = (
    BUNDLE_ROOT / "scripts" / "matbench_crabnet_perovskites_abx_onehot_stable.py"
)
TASK_NAME = "matbench_perovskites"
TARGET_COLUMN = "e_form"
EXPECTED_SAMPLE_COUNT = 18_928
REPRESENTATIONS = (
    "crabnet_original",
    "kg",
    "lm",
    "wren444",
    "onehot_1731",
    "abx_onehot",
)
VECTOR_REPRESENTATIONS = {
    "kg",
    "lm",
    "wren444",
    "onehot_1731",
}
DISPLAY_NAMES = {
    "crabnet_original": "CrabNet",
    "kg": "Symmetry2Vec-KG",
    "lm": "Symmetry2Vec-LM",
    "wren444": "Wren",
    "onehot_1731": "One-hot",
    "abx_onehot": "ABX one-hot",
}
DEFAULT_SPLIT_DIR = BUNDLE_ROOT / "splits" / "perovskites_composition_grouped_5fold"
DEFAULT_CACHE = DEFAULT_SPLIT_DIR / "perovskites_all_tokens.pkl.gz"
DEFAULT_OUTPUT_ROOT = (
    BUNDLE_ROOT / "runs" / "perovskites_composition_grouped_5fold_seed42"
)


def default_crabnet_root() -> Path:
    configured = os.environ.get("CRABNET_ROOT")
    if configured:
        return Path(configured)
    return BUNDLE_ROOT / "crabnet_runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--crabnet-root", type=Path, default=default_crabnet_root())
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.patience,
        args.batch_size,
        args.cpu_threads,
        args.d_model,
        args.layers,
        args.heads,
    )
    if min(positive) < 1:
        raise ValueError("Positive integer hyperparameters must be at least 1")
    if args.seed != 42:
        raise ValueError("This experiment resets every fold to seed 42")
    if not 0.0 < args.cuda_memory_fraction <= 0.5:
        raise ValueError("cuda-memory-fraction must be in (0, 0.5]")
    if args.d_model % args.heads:
        raise ValueError("d-model must be divisible by heads")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.representation / f"fold{args.fold}"


def valid_completed(
    args: argparse.Namespace,
    assignments_hash: str,
    cache_hash: str,
) -> bool:
    directory = output_dir(args)
    score_path = directory / "score.csv"
    prediction_path = directory / "predictions.csv"
    audit_path = directory / "run_audit.json"
    if not all(path.is_file() and path.stat().st_size for path in (score_path, prediction_path, audit_path)):
        return False
    try:
        with score_path.open("r", encoding="utf-8", newline="") as handle:
            score_rows = list(csv.DictReader(handle))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(prediction_path, usecols=["fold", "representation"])
        return (
            len(score_rows) == 1
            and score_rows[0].get("representation") == args.representation
            and int(score_rows[0].get("fold", -1)) == args.fold
            and not predictions.empty
            and predictions["fold"].eq(args.fold).all()
            and predictions["representation"].eq(args.representation).all()
            and audit.get("status") == "completed"
            and audit.get("assignments_sha256") == assignments_hash
            and audit.get("token_cache_sha256") == cache_hash
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def load_fixed_data(args: argparse.Namespace) -> dict[str, object]:
    split_dir = args.split_dir.expanduser().resolve()
    args.split_dir = split_dir
    args.cache = args.cache.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.crabnet_root = args.crabnet_root.expanduser().resolve()
    paths = {
        "assignments": split_dir / "assignments.csv",
        "roles": split_dir / "fold_roles.csv",
        "split_manifest": split_dir / "manifest.json",
        "cache": args.cache,
        "cache_manifest": args.cache.with_name(args.cache.name + ".manifest.json"),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.crabnet_root.is_dir():
        raise FileNotFoundError(args.crabnet_root)

    split_manifest = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    assignments_hash = sha256_file(paths["assignments"])
    roles_hash = sha256_file(paths["roles"])
    if split_manifest.get("task") != TASK_NAME:
        raise RuntimeError("Unexpected split task")
    if split_manifest.get("split_kind") != "composition_group_disjoint_stratified_5fold":
        raise RuntimeError("Unexpected split kind")
    if int(split_manifest.get("outer_seed", -1)) != args.seed:
        raise RuntimeError("Outer split seed mismatch")
    if int(split_manifest.get("validation_seed_each_fold", -1)) != args.seed:
        raise RuntimeError("Validation split seed mismatch")
    if split_manifest.get("assignments_sha256") != assignments_hash:
        raise RuntimeError("Assignments hash mismatch")
    if split_manifest.get("fold_roles_sha256") != roles_hash:
        raise RuntimeError("Fold-role hash mismatch")

    cache_manifest = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
    cache_hash = sha256_file(paths["cache"])
    if cache_manifest.get("cache_sha256") != cache_hash:
        raise RuntimeError("Token-cache hash mismatch")
    if cache_manifest.get("assignments_sha256") != assignments_hash:
        raise RuntimeError("Token cache was built for a different split table")
    with gzip.open(paths["cache"], "rb") as handle:
        cache = pickle.load(handle)

    assignments = pd.read_csv(paths["assignments"])
    assignments["material_id"] = assignments["material_id"].astype(str)
    roles = pd.read_csv(paths["roles"])
    material_ids = np.asarray([str(value) for value in cache["material_ids"]], dtype=object)
    targets = np.asarray(cache["targets"], dtype=np.float32)
    space_groups = np.asarray(cache["space_group_numbers"], dtype=np.int64)
    tokenized = cache["tokenized"]
    lengths = {len(assignments), len(material_ids), len(targets), len(space_groups), len(tokenized)}
    if lengths != {EXPECTED_SAMPLE_COUNT}:
        raise RuntimeError(f"Unexpected dataset inventory lengths: {sorted(lengths)}")
    if assignments["position"].tolist() != list(range(EXPECTED_SAMPLE_COUNT)):
        raise RuntimeError("Assignments are not in natural dataset order")
    if assignments["material_id"].tolist() != material_ids.tolist():
        raise RuntimeError("Assignment IDs and token-cache IDs differ")
    if not np.array_equal(assignments["target"].to_numpy(dtype=np.float32), targets):
        raise RuntimeError("Assignment targets and token-cache targets differ")
    if sequence_sha256(material_ids.tolist()) != split_manifest["material_ids_sha256"]:
        raise RuntimeError("Material-ID sequence hash mismatch")
    if assignments.groupby("composition")["fold"].nunique().max() != 1:
        raise RuntimeError("A composition appears in multiple outer folds")

    fold_roles = roles.loc[roles["outer_fold"] == args.fold].copy()
    if len(fold_roles) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Fold {args.fold} does not assign every sample exactly once")
    if fold_roles["position"].nunique() != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Fold {args.fold} contains duplicate role positions")
    positions = {
        role: np.sort(
            fold_roles.loc[fold_roles["role"] == role, "position"].to_numpy(
                dtype=np.int64
            )
        )
        for role in ("train", "validation", "test")
    }
    compositions = assignments["composition"].astype(str).to_numpy(dtype=object)
    group_sets = {
        role: set(compositions[split_positions].tolist())
        for role, split_positions in positions.items()
    }
    overlaps = {
        "train_validation": group_sets["train"] & group_sets["validation"],
        "train_test": group_sets["train"] & group_sets["test"],
        "validation_test": group_sets["validation"] & group_sets["test"],
    }
    if any(overlaps.values()):
        raise RuntimeError(
            f"Composition leakage: { {key: len(value) for key, value in overlaps.items()} }"
        )
    return {
        "paths": paths,
        "split_manifest": split_manifest,
        "assignments_hash": assignments_hash,
        "roles_hash": roles_hash,
        "cache_hash": cache_hash,
        "assignments": assignments,
        "material_ids": material_ids,
        "targets": targets,
        "space_groups": space_groups,
        "tokenized": tokenized,
        "compositions": compositions,
        "positions": positions,
        "group_sets": group_sets,
    }


def validate_full_embedding_coverage(
    tokenized_splits: dict[str, list[object]], active_wp_entities: set[str]
) -> dict[str, dict[str, object]]:
    audit: dict[str, dict[str, object]] = {}
    for role, samples in tokenized_splits.items():
        uncovered: Counter[str] = Counter()
        empty = 0
        for tokens in samples:
            if not isinstance(tokens, (list, tuple)) or not tokens:
                empty += 1
                continue
            for token in tokens:
                wp_id = str(token[1])
                if wp_id not in active_wp_entities:
                    uncovered[wp_id] += 1
        audit[role] = {
            "samples_before": len(samples),
            "samples_after": len(samples),
            "dropped_samples": 0,
            "empty_token_sample_count": empty,
            "uncovered_wp_sample_counts": dict(sorted(uncovered.items())),
        }
        if empty or uncovered:
            raise RuntimeError(
                f"Embedding coverage failed for {role}: empty={empty}, uncovered={dict(uncovered)}"
            )
    return audit


def load_vector_representation(
    args: argparse.Namespace,
    base_runner,
    positions_before: dict[str, np.ndarray],
    tokenized_all: list[object],
) -> dict[str, object]:
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(REGISTRY_PATH)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    info = registry["representations"][args.representation]
    embedding_path = (BUNDLE_ROOT / info["path"]).resolve()
    if sha256_file(embedding_path) != info["sha256"]:
        raise RuntimeError(f"Embedding SHA256 mismatch: {embedding_path}")
    embedding_ids = pd.read_csv(embedding_path, sep="\t", usecols=["id"])["id"].astype(str)
    embedding_wp_entities = set(embedding_ids)
    if len(embedding_wp_entities) != int(info["rows"]):
        raise RuntimeError("Embedding IDs are duplicated or incomplete")

    raw_tokens = {
        role: [tokenized_all[int(position)] for position in split_positions]
        for role, split_positions in positions_before.items()
    }
    coverage_relative = info.get("coverage")
    coverage_path = None
    coverage_hash = None
    coverage_rows = None
    if coverage_relative:
        coverage_path = (BUNDLE_ROOT / str(coverage_relative)).resolve()
        coverage_hash = sha256_file(coverage_path)
        if coverage_hash != info.get("coverage_sha256"):
            raise RuntimeError(f"Coverage SHA256 mismatch: {coverage_path}")
        active_wp_entities, coverage_rows = base_runner.load_active_wp_entities(
            coverage_path
        )
        if not active_wp_entities.issubset(embedding_wp_entities):
            raise RuntimeError("Coverage table contains IDs absent from embedding table")
        tokenized_splits, keep_relative, coverage_audit = (
            base_runner._filter_tokenized_splits_by_wp_coverage(
                raw_tokens, active_wp_entities
            )
        )
        positions = {
            role: positions_before[role][keep_relative[role]]
            for role in positions_before
        }
        coverage_policy = "native pretraining-active WP coverage; drop uncovered samples"
    else:
        active_wp_entities = embedding_wp_entities
        tokenized_splits = raw_tokens
        positions = positions_before
        coverage_audit = validate_full_embedding_coverage(
            tokenized_splits, active_wp_entities
        )
        coverage_policy = "all embedding IDs; no sample filtering"

    return {
        "info": info,
        "embedding_path": embedding_path,
        "embedding_sha256": info["sha256"],
        "embedding_wp_entities": len(embedding_wp_entities),
        "active_wp_entities": len(active_wp_entities),
        "coverage_path": coverage_path,
        "coverage_sha256": coverage_hash,
        "coverage_rows": coverage_rows,
        "coverage_policy": coverage_policy,
        "coverage_audit": coverage_audit,
        "tokenized_splits": tokenized_splits,
        "positions": positions,
    }


def module_inventory(crabnet_root: Path) -> dict[str, Path]:
    paths = {
        "base_runner": crabnet_root
        / "publication_CrabNet"
        / "matbench_crabnet_orbit_sym_wp_filter.py",
        "stable_runner": crabnet_root
        / "publication_CrabNet"
        / "matbench_crabnet_orbit_sym_v2_nocif_200_stable.py",
        "mat2vec": crabnet_root / "data" / "element_properties" / "mat2vec.csv",
        "abx_runner": ABX_MODULE_PATH,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training modules: {missing}")
    return paths


def load_training_modules(crabnet_root: Path):
    sys.path.insert(0, str(crabnet_root))
    sys.path.insert(0, str(crabnet_root / "publication_CrabNet"))
    base_runner = importlib.import_module("matbench_crabnet_orbit_sym_wp_filter")
    stable_runner = importlib.import_module(
        "matbench_crabnet_orbit_sym_v2_nocif_200_stable"
    )
    abx_runner = import_file_module("final_perovskites_abx_onehot_stable", ABX_MODULE_PATH)
    return base_runner, stable_runner, abx_runner


def group_audit(
    positions: dict[str, np.ndarray], compositions: np.ndarray
) -> tuple[dict[str, set[str]], dict[str, int]]:
    groups = {
        role: set(compositions[split_positions].tolist())
        for role, split_positions in positions.items()
    }
    overlap_counts = {
        "train_validation": len(groups["train"] & groups["validation"]),
        "train_test": len(groups["train"] & groups["test"]),
        "validation_test": len(groups["validation"] & groups["test"]),
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"Composition leakage after coverage filtering: {overlap_counts}")
    return groups, overlap_counts


def load_structures(data: dict[str, object]) -> list[object]:
    benchmark = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = benchmark.tasks_map[TASK_NAME]
    task.load()
    frame = task.df.copy()
    frame.index = frame.index.map(str)
    if len(frame) != EXPECTED_SAMPLE_COUNT or not frame.index.is_unique:
        raise RuntimeError("Unexpected Matbench perovskites dataset inventory")
    if frame.index.tolist() != data["material_ids"].tolist():
        raise RuntimeError("Loaded Matbench structures are not aligned to the split")
    if not np.array_equal(
        frame[TARGET_COLUMN].to_numpy(dtype=np.float32), data["targets"]
    ):
        raise RuntimeError("Loaded Matbench targets are not aligned to the split")
    return frame["structure"].tolist()


def build_vector_pipeline(
    args: argparse.Namespace,
    data: dict[str, object],
    vector: dict[str, object],
    base_runner,
    stable_runner,
    device,
) -> dict[str, object]:
    positions = vector["positions"]
    split_targets = {
        role: data["targets"][split_positions]
        for role, split_positions in positions.items()
    }
    featurizer = base_runner.OrbitFeaturizer(
        mat2vec_path=args.crabnet_root / "data" / "element_properties" / "mat2vec.csv",
        entity_embeddings_path=vector["embedding_path"],
        symprec=0.1,
        angle_tolerance=5.0,
        include_site_symmetry=False,
    )
    print(f"Vectorizing cached tokens; input_dim={featurizer.input_dim}", flush=True)
    features = {
        role: base_runner._vectorize_split(featurizer, vector["tokenized_splits"][role])
        for role in ("train", "validation", "test")
    }
    max_tokens = max(
        len(fractions)
        for role in features
        for fractions in features[role][1]
    )
    loaders = {
        role: base_runner.make_loader(
            features[role][0],
            features[role][1],
            split_targets[role],
            max_tokens,
            args.batch_size,
            shuffle=role == "train",
        )
        for role in ("train", "validation", "test")
    }
    model = stable_runner.OrbitCrabNet(
        input_dim=featurizer.input_dim,
        d_model=args.d_model,
        N=args.layers,
        heads=args.heads,
        compute_device=device,
    ).to(device)
    trainer = stable_runner.Trainer(
        model,
        device,
        model_name=(
            f"perovskites_composition_grouped_{args.representation}_fold{args.fold}"
        ),
        discard_n=args.patience,
    )
    return {
        "positions": positions,
        "loaders": loaders,
        "trainer": trainer,
        "input_dim": int(featurizer.input_dim),
        "max_tokens": int(max_tokens),
        "feature_audit": {
            "mat2vec_dimension": int(
                getattr(
                    featurizer,
                    "mat_dim",
                    featurizer.input_dim - int(vector["info"]["dimension"]),
                )
            ),
            "wp_embedding_dimension": int(vector["info"]["dimension"]),
            "site_symmetry_dimension": 0,
            "site_symmetry_enabled": False,
            "missing_elements": dict(getattr(featurizer, "missing_elements", {})),
            "missing_wp_tokens": dict(getattr(featurizer, "missing_tokens", {})),
        },
    }


def build_original_pipeline(
    args: argparse.Namespace,
    data: dict[str, object],
    structures: list[object],
    base_runner,
    stable_runner,
    device,
) -> dict[str, object]:
    positions = data["positions"]
    featurizer = original.CompositionFeaturizer(max_elements=16)
    features: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for role, split_positions in positions.items():
        split_structures = [structures[int(position)] for position in split_positions]
        features[role] = original.featurize_structures(featurizer, split_structures)
    max_tokens = max(
        len(fractions)
        for role in features
        for fractions in features[role][1]
    )
    loaders = {
        role: base_runner.make_loader(
            features[role][0],
            features[role][1],
            data["targets"][split_positions],
            max_tokens,
            args.batch_size,
            shuffle=role == "train",
        )
        for role, split_positions in positions.items()
    }
    model = original.build_model(args, device, True, stable_runner)
    trainer = stable_runner.Trainer(
        model,
        device,
        model_name=f"perovskites_composition_grouped_crabnet_fold{args.fold}",
        discard_n=args.patience,
    )
    return {
        "positions": positions,
        "loaders": loaders,
        "trainer": trainer,
        "input_dim": 1,
        "max_tokens": int(max_tokens),
        "feature_audit": {
            "input": "atomic number plus stoichiometric fraction",
            "max_elements_configured": 16,
            "max_elements_seen": int(featurizer.max_elements_seen),
            "truncated_samples": int(featurizer.truncated_samples),
            "wyckoff_enabled": False,
        },
    }


def build_abx_pipeline(
    args: argparse.Namespace,
    data: dict[str, object],
    structures: list[object],
    abx_runner,
    device,
) -> dict[str, object]:
    positions = data["positions"]
    featurizer = abx_runner.OrbitFeaturizer(
        mat2vec_path=args.crabnet_root / "data" / "element_properties" / "mat2vec.csv",
        sym2vec_dir=None,
        symprec=0.1,
        angle_tolerance=5.0,
    )
    features = {}
    for role, split_positions in positions.items():
        split_structures = [structures[int(position)] for position in split_positions]
        features[role] = abx_runner.featurize_split(featurizer, split_structures)
    if featurizer.non_five_site_count:
        raise RuntimeError(
            f"ABX featurizer encountered {featurizer.non_five_site_count} non-five-site structures"
        )
    max_tokens = max(
        len(fractions)
        for role in features
        for fractions in features[role][1]
    )
    loaders = {
        role: abx_runner.make_loader(
            features[role][0],
            features[role][1],
            data["targets"][split_positions],
            max_tokens,
            args.batch_size,
            shuffle=role == "train",
        )
        for role, split_positions in positions.items()
    }
    model = abx_runner.OrbitCrabNet(
        input_dim=featurizer.input_dim,
        d_model=args.d_model,
        N=args.layers,
        heads=args.heads,
        compute_device=device,
    ).to(device)
    trainer = abx_runner.Trainer(
        model,
        device,
        model_name=f"perovskites_composition_grouped_abx_onehot_fold{args.fold}",
        discard_n=args.patience,
    )
    return {
        "positions": positions,
        "loaders": loaders,
        "trainer": trainer,
        "input_dim": int(featurizer.input_dim),
        "max_tokens": int(max_tokens),
        "feature_audit": {
            "input": "mat2vec element vector plus A/B/X site one-hot",
            "mat2vec_dimension": int(featurizer.mat_dim),
            "abx_dimension": int(featurizer.abx_dim),
            "abx_order": list(featurizer.ABX_ORDER),
            "site_index_contract": {"B": [0], "A": [1], "X": [2, 3, 4]},
            "missing_elements": dict(featurizer.missing_elements),
            "non_five_site_count": int(featurizer.non_five_site_count),
            "wyckoff_enabled": False,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    data = load_fixed_data(args)
    if not args.force and not args.dry_run and valid_completed(
        args, data["assignments_hash"], data["cache_hash"]
    ):
        print(
            f"Skipping completed {args.representation} fold {args.fold}", flush=True
        )
        return

    module_paths = module_inventory(args.crabnet_root)
    vector = None
    if args.representation in VECTOR_REPRESENTATIONS:
        # The coverage helper is imported only after the exact CrabNet tree is selected.
        sys.path.insert(0, str(args.crabnet_root))
        sys.path.insert(0, str(args.crabnet_root / "publication_CrabNet"))
        coverage_runner = importlib.import_module("matbench_crabnet_orbit_sym_wp_filter")
        vector = load_vector_representation(
            args, coverage_runner, data["positions"], data["tokenized"]
        )
        positions = vector["positions"]
    else:
        positions = data["positions"]

    groups_after, overlap_after = group_audit(positions, data["compositions"])
    directory = output_dir(args)
    audit_path = directory / "run_audit.json"
    audit = {
        "schema_version": "1.0.0",
        "status": "dry_run" if args.dry_run else "initialized",
        "task": TASK_NAME,
        "target": TARGET_COLUMN,
        "split_kind": "composition_group_disjoint_stratified_5fold",
        "representation": args.representation,
        "display_name": DISPLAY_NAMES[args.representation],
        "fold": args.fold,
        "outer_split_seed": args.seed,
        "validation_split_seed": args.seed,
        "model_seed": args.seed,
        "fold_local_seed_reset": True,
        "samples_before_coverage_filter": {
            role: len(value) for role, value in data["positions"].items()
        },
        "samples_after_coverage_filter": {
            role: len(value) for role, value in positions.items()
        },
        "composition_counts_after_coverage_filter": {
            role: len(value) for role, value in groups_after.items()
        },
        "composition_overlap_after_coverage_filter": overlap_after,
        "train_ids_sha256": sequence_sha256(
            data["material_ids"][positions["train"]].tolist()
        ),
        "validation_ids_sha256": sequence_sha256(
            data["material_ids"][positions["validation"]].tolist()
        ),
        "test_ids_sha256": sequence_sha256(
            data["material_ids"][positions["test"]].tolist()
        ),
        "assignments_sha256": data["assignments_hash"],
        "fold_roles_sha256": data["roles_hash"],
        "split_manifest_sha256": sha256_file(data["paths"]["split_manifest"]),
        "token_cache_sha256": data["cache_hash"],
        "epochs": args.epochs,
        "patience_validation_checks": args.patience,
        "validation_check_interval_epochs": 2,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "cpu_threads": args.cpu_threads,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "stable_perovskites_mode": True,
        "site_symmetry_channel": False,
        "module_sha256": {
            name: sha256_file(path) for name, path in module_paths.items()
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if vector is not None:
        audit["vector"] = {
            "dimension": int(vector["info"]["dimension"]),
            "embedding_path": str(vector["embedding_path"]),
            "embedding_sha256": vector["embedding_sha256"],
            "embedding_wp_entities": vector["embedding_wp_entities"],
            "active_wp_entities": vector["active_wp_entities"],
            "coverage_path": (
                str(vector["coverage_path"]) if vector["coverage_path"] else None
            ),
            "coverage_sha256": vector["coverage_sha256"],
            "coverage_rows": vector["coverage_rows"],
            "coverage_policy": vector["coverage_policy"],
            "coverage_audit": vector["coverage_audit"],
        }
    write_json_atomic(audit, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    thread_count = str(args.cpu_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    torch.set_num_threads(args.cpu_threads)
    base_runner, stable_runner, abx_runner = load_training_modules(args.crabnet_root)
    os.chdir(args.crabnet_root)

    seed_everything(args.seed)
    from utils.get_compute_device import get_compute_device

    device = get_compute_device(prefer_last=False)
    if torch.device(device).type != "cuda":
        raise RuntimeError("This benchmark run requires a CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=device)
    print(f"Running on compute device: {device}", flush=True)
    print("Fold-local RNG seed reset: 42", flush=True)

    structures = None
    if args.representation in {"crabnet_original", "abx_onehot"}:
        structures = load_structures(data)
    if args.representation in VECTOR_REPRESENTATIONS:
        pipeline = build_vector_pipeline(
            args, data, vector, base_runner, stable_runner, device
        )
    elif args.representation == "crabnet_original":
        pipeline = build_original_pipeline(
            args, data, structures, base_runner, stable_runner, device
        )
    elif args.representation == "abx_onehot":
        pipeline = build_abx_pipeline(args, data, structures, abx_runner, device)
    else:
        raise AssertionError(args.representation)

    from utils.utils import count_parameters

    model = pipeline["trainer"].model
    parameter_count = int(count_parameters(model))
    print(
        f"Model parameters={parameter_count}; input_dim={pipeline['input_dim']}; "
        f"max_tokens={pipeline['max_tokens']}",
        flush=True,
    )
    audit.update(
        {
            "status": "running",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "model_parameter_count": parameter_count,
            "input_dimension": pipeline["input_dim"],
            "max_tokens": pipeline["max_tokens"],
            "feature_audit": pipeline["feature_audit"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(audit, audit_path)

    if args.preflight:
        model.eval()
        batch_features, batch_fractions, _ = next(iter(pipeline["loaders"]["train"]))
        with torch.no_grad():
            output = model(
                batch_features.to(device),
                batch_fractions.to(device),
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
                "preflight_output_finite": True,
            }
        )
        write_json_atomic(audit, audit_path)
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
        return

    trainer = pipeline["trainer"]
    loaders = pipeline["loaders"]
    started = time.time()
    trainer.fit(loaders["train"], loaders["validation"], epochs=args.epochs)
    elapsed = time.time() - started
    test_mae = float(trainer.evaluate(loaders["test"]))
    validation_mae = float(trainer.evaluate(loaders["validation"]))
    actual, predicted = trainer.predict(loaders["test"])
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    test_positions = pipeline["positions"]["test"]
    expected_actual = data["targets"][test_positions].astype(np.float64)
    if len(actual) != len(test_positions) or not np.allclose(
        actual, expected_actual, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("Prediction rows are not aligned to the fixed test positions")
    recomputed_mae = float(np.mean(np.abs(predicted - actual)))
    if not np.isclose(test_mae, recomputed_mae, rtol=0.0, atol=1e-10):
        raise RuntimeError("Trainer test MAE differs from prediction-table MAE")

    prediction_frame = pd.DataFrame(
        {
            "representation": args.representation,
            "display_name": DISPLAY_NAMES[args.representation],
            "fold": args.fold,
            "test_split_position": np.arange(len(test_positions), dtype=np.int64),
            "dataset_position": test_positions,
            "material_id": data["material_ids"][test_positions],
            "composition": data["compositions"][test_positions],
            "space_group_number": data["space_groups"][test_positions],
            "actual": actual,
            "predicted": predicted,
            "signed_error": predicted - actual,
            "absolute_error": np.abs(predicted - actual),
        }
    )
    write_frame_atomic(prediction_frame, directory / "predictions.csv")
    groups_final, overlap_final = group_audit(
        pipeline["positions"], data["compositions"]
    )
    result = {
        "representation": args.representation,
        "display_name": DISPLAY_NAMES[args.representation],
        "split_kind": "composition_group_disjoint_stratified_5fold",
        "fold": args.fold,
        "best_epoch": int(trainer.best_epoch),
        "validation_mae": validation_mae,
        "test_mae": test_mae,
        "train_samples": len(pipeline["positions"]["train"]),
        "validation_samples": len(pipeline["positions"]["validation"]),
        "test_samples": len(pipeline["positions"]["test"]),
        "train_compositions": len(groups_final["train"]),
        "validation_compositions": len(groups_final["validation"]),
        "test_compositions": len(groups_final["test"]),
        "composition_overlap_count": sum(overlap_final.values()),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "patience_validation_checks": args.patience,
        "batch_size": args.batch_size,
        "input_dimension": pipeline["input_dim"],
        "max_tokens": pipeline["max_tokens"],
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
        f"Fold {args.fold} {args.representation}: best_epoch={trainer.best_epoch} "
        f"validation_mae={validation_mae:.6f} test_mae={test_mae:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
