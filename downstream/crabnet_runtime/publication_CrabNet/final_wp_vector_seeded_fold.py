from __future__ import annotations

import random
import sys

import numpy as np
import torch

import matbench_crabnet_orbit_sym_wp_filter as runner


def extract_wrapper_args(argv: list[str]) -> tuple[int, bool, list[str]]:
    fold_seed = None
    stable_mode = False
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--fold-seed":
            if index + 1 >= len(argv):
                raise SystemExit("--fold-seed requires an integer")
            fold_seed = int(argv[index + 1])
            index += 2
        elif argument.startswith("--fold-seed="):
            fold_seed = int(argument.split("=", 1)[1])
            index += 1
        elif argument == "--stable-mode":
            stable_mode = True
            index += 1
        else:
            cleaned.append(argument)
            index += 1
    if fold_seed is None:
        raise SystemExit("--fold-seed is required")
    if "--disable-site-symmetry" not in cleaned:
        cleaned.append("--disable-site-symmetry")
    return fold_seed, stable_mode, cleaned


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def skip_model_save(trainer) -> None:
    print(
        f"Skipped model save for {trainer.model_name}; "
        "best validation weights remain loaded in memory",
        flush=True,
    )


def force_validation_seed(seed: int) -> None:
    original_split = runner.split_train_val

    def fixed_split(structures, targets, val_fraction, random_state):
        print(
            f"Validation split RNG seed forced: {seed} "
            f"(base runner requested {random_state})",
            flush=True,
        )
        return original_split(structures, targets, val_fraction, seed)

    runner.split_train_val = fixed_split


def main() -> None:
    fold_seed, stable_mode, cleaned_argv = extract_wrapper_args(sys.argv)
    sys.argv = cleaned_argv

    if stable_mode:
        import matbench_crabnet_orbit_sym_v2_nocif_200_stable as stable

        runner.OrbitCrabNet = stable.OrbitCrabNet
        runner.Trainer = stable.Trainer
        print("Stable perovskites loss enabled", flush=True)

    runner.Trainer.save = skip_model_save
    seed_everything(fold_seed)
    force_validation_seed(fold_seed)
    print(f"Fold-local RNG seed reset: {fold_seed}", flush=True)
    print("Site-symmetry channel forced off", flush=True)
    print("Model checkpoint saving disabled", flush=True)
    runner.main()


if __name__ == "__main__":
    main()
