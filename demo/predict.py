#!/usr/bin/env python3
"""Predict with the archived KG + CrabNet perovskite checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "downstream" / "crabnet_runtime"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "demo/model_best.pth")
    parser.add_argument("--kg-vectors", type=Path, default=ROOT / "data/kg/entity_embeddings.tsv")
    parser.add_argument("--mat2vec", type=Path, default=RUNTIME / "data/element_properties/mat2vec.csv")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, help="CSV for the test set; JSON for one sample")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--cif", type=Path, help="One complete CIF file")
    inputs.add_argument("--cif-text", help="One complete CIF as an inline string")
    inputs.add_argument("--orbits", type=Path, help="JSON containing space_group and orbits")
    inputs.add_argument("--orbit", action="append", nargs=3,
                        metavar=("ELEMENT", "WYCKOFF", "FRACTION"),
                        help="Repeat for each element-Wyckoff-fraction row")
    parser.add_argument("--space-group", type=int, help="International space-group number (1-230)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    if args.space_group is not None and not (1 <= args.space_group <= 230):
        parser.error("space-group must be between 1 and 230")
    if args.space_group is not None and not (args.orbit or args.orbits):
        parser.error("space-group is only used with orbit inputs")
    return args


def manual_tokens(rows, sg, featurizer):
    """Resolve, combine and normalize explicit orbit rows using training lookup."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("orbits must be a non-empty list")
    if sg is not None:
        if isinstance(sg, bool) or int(sg) != float(sg) or not 1 <= int(sg) <= 230:
            raise ValueError("space_group must be an integer between 1 and 230")
        sg = int(sg)
    counts = defaultdict(float)
    observed_groups = set()
    for row in rows:
        element, wp = str(row["element"]), str(row["wyckoff"])
        fraction = float(row["fraction"])
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Every fraction must be finite and strictly positive")
        if element not in featurizer.mat2vec:
            raise ValueError(f"No Mat2Vec embedding for element {element!r}")
        full = re.fullmatch(r"WP:(\d{1,3}):(\d+[A-Za-z]+)", wp)
        group = int(full.group(1)) if full else sg
        symbol = full.group(2) if full else wp
        if group is None or not 1 <= group <= 230:
            raise ValueError("Supply space_group / --space-group, or full WP:221:1a tokens")
        if sg is not None and group != sg:
            raise ValueError("A full WP token disagrees with the supplied space group")
        if not re.fullmatch(r"[1-9]\d*[A-Za-z]+", symbol):
            raise ValueError(f"Invalid Wyckoff position: {wp!r}; use e.g. 1a or WP:221:1a")
        token = featurizer.wyckoff_token(group, symbol)
        if token not in featurizer.entity_to_index:
            raise ValueError(f"KG has no matching Wyckoff vector for {wp!r} in SG {group}")
        observed_groups.add(group)
        counts[(element, token)] += fraction
    if len(observed_groups) != 1:
        raise ValueError("All orbits in one sample must belong to the same space group")
    total = sum(counts.values())
    return [(el, wp, None, value / total) for (el, wp), value in
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def load_predictor(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(RUNTIME))
    sys.path.insert(0, str(RUNTIME / "publication_CrabNet"))
    from matbench_crabnet_orbit_sym_wp_filter import OrbitFeaturizer
    from matbench_crabnet_orbit_sym_v2_nocif_200_stable import OrbitCrabNet

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    # The checkpoint includes dictionaries of tensors and primitive metadata only.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    required = {"weights", "scaler_state", "model_config", "preprocessing", "task", "fold"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint is missing required metadata: {sorted(required - checkpoint.keys())}")
    if checkpoint["task"] != "matbench_perovskites" or checkpoint["fold"] != 0:
        raise ValueError("This demo requires a matbench_perovskites fold-0 checkpoint")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing["include_site_symmetry"] or preprocessing.get("coverage_filter", False):
        raise ValueError("This demo expects KG with site symmetry disabled and no coverage filter")
    featurizer = OrbitFeaturizer(
        args.mat2vec, args.kg_vectors,
        symprec=preprocessing["symprec"],
        angle_tolerance=preprocessing["angle_tolerance"],
        include_site_symmetry=False,
    )
    if featurizer.sym_dim != 200 or featurizer.input_dim != checkpoint["model_config"]["input_dim"]:
        raise ValueError("Expected 200-dimensional KG vectors matching the checkpoint input dimension")
    model = OrbitCrabNet(**checkpoint["model_config"], compute_device=device).to(device)
    model.load_state_dict(checkpoint["weights"], strict=True)
    model.eval()
    scaler = checkpoint["scaler_state"]
    mean = torch.as_tensor(scaler["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(scaler["std"], dtype=torch.float32, device=device)
    if mean.numel() != 1 or std.numel() != 1 or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or std.item() <= 0:
        raise ValueError("Invalid checkpoint target scaler")
    max_orbits = int(preprocessing["max_orbits"])
    if max_orbits < 1:
        raise ValueError("Invalid checkpoint max_orbits")

    def predict(tokenized_samples):
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(tokenized_samples), args.batch_size):
                batch = tokenized_samples[start:start + args.batch_size]
                features = np.zeros((len(batch), max_orbits, featurizer.input_dim), dtype=np.float32)
                fractions = np.zeros((len(batch), max_orbits), dtype=np.float32)
                for index, tokens in enumerate(batch):
                    if not tokens or len(tokens) > max_orbits:
                        raise ValueError(f"Checkpoint supports 1-{max_orbits} orbits; got {len(tokens)}")
                    x, frac = featurizer.vectorize(tokens)
                    features[index, :len(tokens)] = x
                    fractions[index, :len(tokens)] = frac / frac.sum()
                output = model(torch.from_numpy(features).to(device), torch.from_numpy(fractions).to(device))
                prediction, _ = output.chunk(2, dim=-1)
                prediction = prediction.reshape(-1) * std + mean
                if not torch.isfinite(prediction).all():
                    raise ValueError("Model produced non-finite predictions")
                predictions.extend(prediction.cpu().tolist())
        return predictions

    return checkpoint, featurizer, predict, str(device)


def diagnostics(featurizer):
    return {
        "wyckoff_fallbacks": dict(featurizer.wp_token_fallbacks),
        "symmetry_analysis_failures": featurizer.failure_count,
        "missing_elements": dict(featurizer.missing_elements),
        "missing_tokens": dict(featurizer.missing_tokens),
    }


def structure_tokens(structure, featurizer):
    tokens = featurizer.tokenize(structure)
    for element, wp, _, _ in tokens:
        if element not in featurizer.mat2vec or wp not in featurizer.entity_to_index:
            raise ValueError(f"Structure contains an unsupported element/Wyckoff pair: {element}, {wp}")
    return tokens


def main():
    args = arguments()
    checkpoint, featurizer, predict, device = load_predictor(args)
    single = args.cif is not None or args.cif_text is not None or args.orbits is not None or args.orbit is not None
    common = {"task": checkpoint["task"], "checkpoint_fold": checkpoint["fold"],
              "checkpoint": str(args.checkpoint.resolve()), "device": device,
              "target": "e_form", "unit": "eV/atom"}
    if single:
        if args.cif is not None or args.cif_text is not None:
            from pymatgen.core import Structure
            structure = Structure.from_file(str(args.cif)) if args.cif is not None else Structure.from_str(args.cif_text, fmt="cif")
            tokens = structure_tokens(structure, featurizer)
            source = {"input_mode": "cif", "formula": structure.composition.reduced_formula}
        else:
            if args.orbits is not None:
                payload = json.loads(args.orbits.read_text(encoding="utf-8-sig"))
                sg = payload.get("space_group", args.space_group)
                if args.space_group is not None and sg != args.space_group:
                    raise ValueError("JSON space_group disagrees with --space-group")
                rows = payload["orbits"]
            else:
                sg = args.space_group
                rows = [dict(element=el, wyckoff=wp, fraction=frac) for el, wp, frac in args.orbit]
            tokens = manual_tokens(rows, sg, featurizer)
            source = {"input_mode": "orbits"}
        result = dict(common, **source, prediction_eV_per_atom=predict([tokens])[0],
                      resolved_orbits=[dict(element=el, wyckoff=wp, fraction=frac) for el, wp, _, frac in tokens],
                      diagnostics=diagnostics(featurizer))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        import numpy as np
        import pandas as pd
        from matbench.bench import MatbenchBenchmark
        benchmark = MatbenchBenchmark(autoload=False, subset=["matbench_perovskites"])
        task = benchmark.matbench_perovskites
        task.load()
        structures, targets = task.get_test_data(0, include_target=True, as_type="tuple")
        tokens = [structure_tokens(structure, featurizer) for structure in structures]
        values = predict(tokens)
        output = args.output or ROOT / "demo/outputs/perovskites_fold0_predictions.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"material_id": structures.index.astype(str),
                      "formula": [s.composition.reduced_formula for s in structures],
                      "target_eV_per_atom": np.asarray(targets, dtype=float),
                      "prediction_eV_per_atom": values}).to_csv(output, index=False)
        result = dict(common, input_mode="official_test_set", test_fold=0, samples=len(values),
                      mae_eV_per_atom=float(np.mean(np.abs(np.asarray(values) - np.asarray(targets)))),
                      output=str(output.resolve()), diagnostics=diagnostics(featurizer))
        output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
