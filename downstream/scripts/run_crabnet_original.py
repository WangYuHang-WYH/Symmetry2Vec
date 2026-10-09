from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Element
from torch import nn


TASKS = {
    "jdft2d": ("matbench_jdft2d", "jdft2d", False),
    "phonons": ("matbench_phonons", "phonons", False),
    "dielectric": ("matbench_dielectric", "dielectric", False),
    "log_gvrh": ("matbench_log_gvrh", "log_gvrh", False),
    "log_kvrh": ("matbench_log_kvrh", "log_kvrh", False),
    "mp_e_form": ("matbench_mp_e_form", "mp_e_form", False),
    "mp_gap": ("matbench_mp_gap", "mp_gap", False),
    "stable_perovskites": (
        "matbench_perovskites",
        "stable_perovskites",
        True,
    ),
}


def default_crabnet_root() -> Path:
    configured = os.environ.get("CRABNET_ROOT")
    if configured:
        return Path(configured)
    return BUNDLE_ROOT / "crabnet_runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Final-protocol Matbench fold with the original "
            "composition-only CrabNet model."
        )
    )
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--crabnet-root", type=Path, default=default_crabnet_root())
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.48)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.cpu_threads,
        args.d_model,
        args.layers,
        args.heads,
    ) < 1:
        raise ValueError("positive integer hyperparameters must be at least 1")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0, 1)")
    if not 0.0 < args.cuda_memory_fraction <= 0.5:
        raise ValueError("cuda-memory-fraction must be in (0, 0.5] for dual workers")
    if args.d_model % args.heads != 0:
        raise ValueError("d-model must be divisible by heads")

    args.crabnet_root = args.crabnet_root.expanduser().resolve()
    required = (
        args.crabnet_root / "crabnet" / "kingcrab.py",
        args.crabnet_root
        / "publication_CrabNet"
        / "matbench_crabnet_orbit_sym_wp_filter.py",
        args.crabnet_root
        / "publication_CrabNet"
        / "matbench_crabnet_orbit_sym_v2_nocif_200_stable.py",
        args.crabnet_root / "data" / "element_properties" / "mat2vec.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required CrabNet files: {missing}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(task: str, seed: int) -> str:
    return f"final_crabnet_original_seed{seed}_{task}"


class CompositionFeaturizer:
    """Build the element-number and fraction inputs used by original CrabNet."""

    def __init__(self, max_elements: int = 16):
        self.max_elements = int(max_elements)
        self.max_elements_seen = 0
        self.truncated_samples = 0

    def featurize(self, structure) -> tuple[np.ndarray, np.ndarray]:
        amounts = structure.composition.get_el_amt_dict()
        ordered = sorted(
            amounts.items(),
            key=lambda item: (-float(item[1]), Element(item[0]).Z),
        )
        self.max_elements_seen = max(self.max_elements_seen, len(ordered))
        if len(ordered) > self.max_elements:
            self.truncated_samples += 1
            ordered = ordered[: self.max_elements]
        if not ordered:
            raise ValueError("Encountered a structure with an empty composition")

        counts = np.asarray([float(amount) for _, amount in ordered], dtype=np.float32)
        total = float(counts.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Encountered a non-positive composition total")
        fractions = counts / total
        element_numbers = np.asarray(
            [[Element(symbol).Z] for symbol, _ in ordered],
            dtype=np.float32,
        )
        return element_numbers, fractions


class OriginalCrabNetAdapter(nn.Module):
    """Adapt Final's tensor contract without changing original CrabNet."""

    def __init__(self, crabnet: nn.Module):
        super().__init__()
        self.crabnet = crabnet

    def forward(self, encoded_elements: torch.Tensor, fractions: torch.Tensor):
        if encoded_elements.ndim != 3 or encoded_elements.shape[-1] != 1:
            raise ValueError(
                "Expected encoded element numbers with shape [batch, tokens, 1]"
            )
        source = encoded_elements[..., 0].to(dtype=torch.long)
        return self.crabnet(source, fractions)


def patch_stable_original_encoder(model: OriginalCrabNetAdapter, safe_pow2) -> None:
    """Apply Final stable-mode exponent clamps to the original encoder."""

    def stable_forward(encoder, source, fractions):
        embedded = encoder.embed(source) * safe_pow2(
            encoder.emb_scaler,
            -4.0,
            4.0,
        )
        mask = fractions.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        source_padding_mask = mask[:, 0] != 1

        positional = torch.zeros_like(embedded)
        log_positional = torch.zeros_like(embedded)
        positional_scale = safe_pow2(
            (1 - encoder.pos_scaler) ** 2,
            0.0,
            8.0,
        )
        log_positional_scale = safe_pow2(
            (1 - encoder.pos_scaler_log) ** 2,
            0.0,
            8.0,
        )
        positional[:, :, : encoder.d_model // 2] = (
            encoder.pe(fractions) * positional_scale
        )
        log_positional[:, :, encoder.d_model // 2 :] = (
            encoder.ple(fractions) * log_positional_scale
        )

        if encoder.attention:
            transformed = (embedded + positional + log_positional).transpose(0, 1)
            embedded = encoder.transformer_encoder(
                transformed,
                src_key_padding_mask=source_padding_mask,
            ).transpose(0, 1)
        if encoder.fractional:
            embedded = embedded * fractions.unsqueeze(2).repeat(
                1,
                1,
                encoder.d_model,
            )
        hidden_mask = mask[:, :, 0:1].repeat(1, 1, encoder.d_model)
        return embedded.masked_fill(hidden_mask == 0, 0)

    encoder = model.crabnet.encoder
    encoder.forward = types.MethodType(stable_forward, encoder)


def featurize_structures(featurizer, structures):
    features = []
    fractions = []
    for structure in structures:
        feature, fraction = featurizer.featurize(structure)
        features.append(feature)
        fractions.append(fraction)
    return features, fractions


def build_model(args, device, stable_mode, stable_module):
    from crabnet.kingcrab import CrabNet

    original = CrabNet(
        d_model=args.d_model,
        N=args.layers,
        heads=args.heads,
        compute_device=device,
    )
    model = OriginalCrabNetAdapter(original)
    if stable_mode:
        patch_stable_original_encoder(model, stable_module.safe_pow2)
    return model.to(device)


def preflight_model(model, loader, device) -> None:
    model.eval()
    encoded_elements, fractions, _ = next(iter(loader))
    with torch.no_grad():
        output = model(encoded_elements.to(device), fractions.to(device))
    if output.ndim != 2 or output.shape[1] != 2:
        raise RuntimeError(f"Unexpected original CrabNet output shape: {output.shape}")
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("Preflight output contains NaN or Inf")
    print(
        json.dumps(
            {
                "preflight": "passed",
                "batch_shape": list(encoded_elements.shape),
                "output_shape": list(output.shape),
                "output_finite": True,
            },
            indent=2,
        ),
        flush=True,
    )


def run_fold(args, base_runner, stable_module, device) -> dict[str, object] | None:
    task_name, target_name, stable_mode = TASKS[args.task]
    benchmark = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = benchmark.tasks_map[task_name]
    task.load()

    train_val_structures, train_val_targets = task.get_train_and_val_data(
        args.fold,
        as_type="tuple",
    )
    test_structures, test_targets = task.get_test_data(
        args.fold,
        as_type="tuple",
        include_target=True,
    )
    train_structures, y_train, val_structures, y_val = base_runner.split_train_val(
        train_val_structures,
        train_val_targets,
        args.val_fraction,
        args.seed,
    )
    test_structures = list(test_structures)
    test_targets = np.asarray(test_targets, dtype=np.float32)

    print(
        f"Fold {args.fold}: composition featurization train={len(train_structures)}, "
        f"val={len(val_structures)}, test={len(test_structures)}",
        flush=True,
    )
    featurizer = CompositionFeaturizer(max_elements=16)
    x_train, f_train = featurize_structures(featurizer, train_structures)
    x_val, f_val = featurize_structures(featurizer, val_structures)
    x_test, f_test = featurize_structures(featurizer, test_structures)
    max_tokens = max(
        max(len(fraction) for fraction in f_train),
        max(len(fraction) for fraction in f_val),
        max(len(fraction) for fraction in f_test),
    )
    print(
        f"Fold {args.fold}: max_elements={max_tokens}; "
        f"max_elements_seen={featurizer.max_elements_seen}; "
        f"truncated_samples={featurizer.truncated_samples}",
        flush=True,
    )

    train_loader = base_runner.make_loader(
        x_train,
        f_train,
        np.asarray(y_train, dtype=np.float32),
        max_tokens,
        args.batch_size,
        shuffle=True,
    )
    val_loader = base_runner.make_loader(
        x_val,
        f_val,
        np.asarray(y_val, dtype=np.float32),
        max_tokens,
        args.batch_size,
        shuffle=False,
    )
    test_loader = base_runner.make_loader(
        x_test,
        f_test,
        test_targets,
        max_tokens,
        args.batch_size,
        shuffle=False,
    )

    model = build_model(args, device, stable_mode, stable_module)
    from utils.utils import count_parameters

    print(f"Original CrabNet model size: {count_parameters(model)} parameters", flush=True)
    if args.preflight:
        preflight_model(model, train_loader, device)
        return None

    trainer_class = stable_module.Trainer if stable_mode else base_runner.Trainer
    trainer = trainer_class(
        model,
        device,
        model_name=f"{target_name}_{run_name(args.task, args.seed)}{args.fold}",
        discard_n=args.patience,
    )
    started_at = datetime.now(timezone.utc)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)
    print("Model checkpoint saving disabled", flush=True)

    test_mae = trainer.evaluate(test_loader)
    val_mae = trainer.evaluate(val_loader)
    actual, predicted = trainer.predict(test_loader)
    output_run_name = run_name(args.task, args.seed)
    prediction_dir = Path("publication_predictions") / f"{output_run_name}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{target_name}_test_cv{args.fold}.csv"
    pd.DataFrame(
        {
            "fold": args.fold,
            "test_split_position": np.arange(len(test_targets), dtype=int),
            "actual": actual,
            "predicted": predicted,
        }
    ).to_csv(prediction_path, index=False)

    result = {
        "fold": args.fold,
        "best_epoch": trainer.best_epoch,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(test_targets),
        "max_elements": max_tokens,
        "truncated_samples": featurizer.truncated_samples,
    }
    artifact_dir = Path("artifacts") / output_run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        artifact_dir / f"fold_scores_start{args.fold}_n1.csv",
        index=False,
    )

    audit = {
        "schema_version": "1.0.0",
        "model": "crabnet.kingcrab.CrabNet",
        "input": "composition_only_element_number_mat2vec_fraction",
        "wyckoff": False,
        "site_symmetry": False,
        "space_group": False,
        "drop_unary": False,
        "group_duplicate_formulas": False,
        "task": args.task,
        "matbench_task": task_name,
        "target_name": target_name,
        "fold": args.fold,
        "fold_seed": args.seed,
        "validation_split_seed": args.seed,
        "stable_mode": stable_mode,
        "stable_mode_changes": (
            "clamped encoder scale exponents, clamped uncertainty, gradient clipping, "
            "and finite-value checks"
            if stable_mode
            else None
        ),
        "epochs": args.epochs,
        "patience_validation_checks": args.patience,
        "validation_check_interval_epochs": 2,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cpu_threads": args.cpu_threads,
        "mat2vec_path": str(
            (args.crabnet_root / "data" / "element_properties" / "mat2vec.csv").resolve()
        ),
        "mat2vec_sha256": sha256_file(
            args.crabnet_root / "data" / "element_properties" / "mat2vec.csv"
        ),
        "prediction_path": str(prediction_path.resolve()),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    (artifact_dir / f"run_audit_fold{args.fold}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Fold {args.fold}: best_epoch={trainer.best_epoch}, "
        f"val_mae={val_mae:.6f}, test_mae={test_mae:.6f}",
        flush=True,
    )
    print(f"Prediction: {prediction_path}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    task_name, target_name, stable_mode = TASKS[args.task]
    configuration = {
        "model": "original_crabnet_composition_only",
        "task": args.task,
        "matbench_task": task_name,
        "target_name": target_name,
        "fold": args.fold,
        "seed": args.seed,
        "validation_split_seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "val_fraction": args.val_fraction,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "stable_mode": stable_mode,
        "wyckoff": False,
        "site_symmetry": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "crabnet_root": str(args.crabnet_root),
        "run_name": run_name(args.task, args.seed),
    }
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    os.chdir(args.crabnet_root)
    sys.path.insert(0, str(args.crabnet_root))
    sys.path.insert(0, str(args.crabnet_root / "publication_CrabNet"))
    import matbench_crabnet_orbit_sym_v2_nocif_200_stable as stable_module
    import matbench_crabnet_orbit_sym_wp_filter as base_runner
    from utils.get_compute_device import get_compute_device

    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = get_compute_device(prefer_last=False)
    if torch.device(device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=device,
        )
    print(f"Running on compute device: {device}", flush=True)
    print("Fold-local RNG seed reset: 42" if args.seed == 42 else f"Fold-local RNG seed reset: {args.seed}", flush=True)
    print("Validation split RNG uses the same fold-local seed", flush=True)
    run_fold(args, base_runner, stable_module, device)


if __name__ == "__main__":
    main()
