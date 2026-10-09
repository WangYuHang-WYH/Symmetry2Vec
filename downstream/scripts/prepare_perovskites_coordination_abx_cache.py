#!/usr/bin/env python3
"""Build geometry-derived A/B/X role tokens for Matbench perovskites.

The assignment is independent of species and input site order.  For each site,
periodic neighbour distances are inspected at the shell closures expected for
an ideal perovskite: CN=12 for A, CN=6 for B, and CN=2 for X.  All 20 possible
choices of one A site and one B site in the five-site cell are scored globally;
the remaining three sites are X.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
from matbench.bench import MatbenchBenchmark


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "matbench_perovskites"
TARGET_COLUMN = "e_form"
EXPECTED_SAMPLE_COUNT = 18_928
DEFAULT_OUTPUT = (
    BUNDLE_ROOT
    / "splits"
    / "perovskites_coordination_abx"
    / "coordination_abx_tokens.pkl.gz"
)
ROLE_ORDER = ("A", "B", "X")
ROLE_ONEHOT = {
    "A": (1.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0),
    "X": (0.0, 0.0, 1.0),
}
EXPECTED_CN = {"A": 12, "B": 6, "X": 2}
LEGACY_INDEX_ROLES = ("B", "A", "X", "X", "X")
MIN_NEIGHBOURS = 16
MAX_RADIUS_EXPANSIONS = 6
PARENT_ROLE_SITES = {
    "A": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    "B": np.asarray([0.5, 0.5, 0.5], dtype=np.float64),
    "X": (
        np.asarray([0.5, 0.0, 0.5], dtype=np.float64),
        np.asarray([0.5, 0.5, 0.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 0.5], dtype=np.float64),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def dominant_element_and_occupancy(site) -> tuple[str, float]:
    if site.is_ordered:
        return str(site.specie.symbol), 1.0
    specie, occupancy = max(site.species.items(), key=lambda item: item[1])
    return str(specie.symbol), float(occupancy)


def periodic_neighbour_distances_by_target(
    structure, site_index: int
) -> dict[int, np.ndarray]:
    """Return periodic distances from one site to each crystallographic site."""
    radius = max(3.0, 1.25 * float(max(structure.lattice.abc)))
    grouped: dict[int, list[float]] = {index: [] for index in range(5)}
    for _ in range(MAX_RADIUS_EXPANSIONS):
        neighbours = structure.get_neighbors(structure[site_index], radius)
        grouped = {index: [] for index in range(5)}
        for neighbour in neighbours:
            distance = float(neighbour.nn_distance)
            if distance <= 1e-8:
                continue
            grouped[int(neighbour.index)].append(distance)
        for values in grouped.values():
            values.sort()
        if min(len(values) for values in grouped.values()) >= 5:
            break
        radius *= 1.6
    if min(len(values) for values in grouped.values()) < 5:
        raise RuntimeError(
            f"Site {site_index} has insufficient target-resolved periodic neighbours"
        )
    return {
        target: np.asarray(values, dtype=np.float64)
        for target, values in grouped.items()
    }


def distances_to_targets(
    distance_matrix: list[dict[int, np.ndarray]],
    center: int,
    targets: tuple[int, ...],
) -> np.ndarray:
    values = np.concatenate(
        [distance_matrix[center][target] for target in targets]
    )
    values.sort()
    return values


def shell_gap_quality(distances: np.ndarray, coordination: int) -> float:
    """Score how strongly a distance shell closes after ``coordination`` sites."""
    inner = float(distances[coordination - 1])
    outer = float(distances[coordination])
    first = float(distances[0])
    if first <= 0.0 or inner <= 0.0 or outer <= 0.0:
        return -math.inf
    gap = math.log(outer / inner)
    spread = math.log(max(inner / first, 1.0))
    return gap - 0.02 * spread


def nearest_shell_cn(distances: np.ndarray, max_coordination: int) -> tuple[int, float]:
    """Estimate a first-shell CN from the largest early distance gap."""
    limit = min(int(max_coordination), len(distances) - 1)
    ratios = distances[1 : limit + 1] / distances[:limit]
    boundary = int(np.argmax(ratios))
    return boundary + 1, float(ratios[boundary])


def periodic_fractional_distance_squared(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta -= np.round(delta)
    return float(np.dot(delta, delta))


def parent_perovskite_role_cost(
    structure, a_index: int, b_index: int, x_indices: tuple[int, ...]
) -> float:
    """Tie-break coordination-equivalent assignments without using site order."""
    fractional = [np.asarray(site.frac_coords, dtype=np.float64) for site in structure]
    fixed_cost = periodic_fractional_distance_squared(
        fractional[a_index], PARENT_ROLE_SITES["A"]
    ) + periodic_fractional_distance_squared(
        fractional[b_index], PARENT_ROLE_SITES["B"]
    )
    x_cost = min(
        sum(
            periodic_fractional_distance_squared(
                fractional[site_index], prototype
            )
            for site_index, prototype in zip(x_indices, prototype_order)
        )
        for prototype_order in permutations(PARENT_ROLE_SITES["X"])
    )
    return float(fixed_cost + x_cost)


def fractional_coordinate_key(structure, site_index: int) -> tuple[float, float, float]:
    coordinates = np.mod(
        np.asarray(structure[site_index].frac_coords, dtype=np.float64), 1.0
    )
    coordinates[np.isclose(coordinates, 1.0, rtol=0.0, atol=1e-10)] = 0.0
    coordinates[np.isclose(coordinates, 0.0, rtol=0.0, atol=1e-10)] = 0.0
    return tuple(float(value) for value in np.round(coordinates, 10))


def assign_coordination_roles(structure) -> tuple[list[str], dict]:
    if len(structure) != 5:
        raise ValueError(f"Expected a five-site perovskite cell, received {len(structure)}")

    distance_matrix = [
        periodic_neighbour_distances_by_target(structure, index)
        for index in range(5)
    ]

    candidates = []
    site_indices = tuple(range(5))
    for a_index, b_index in permutations(site_indices, 2):
        x_indices = tuple(
            index for index in site_indices if index not in (a_index, b_index)
        )
        a_to_x = distances_to_targets(distance_matrix, a_index, x_indices)
        b_to_x = distances_to_targets(distance_matrix, b_index, x_indices)
        x_to_b = [
            distances_to_targets(distance_matrix, index, (b_index,))
            for index in x_indices
        ]
        x_to_a = [
            distances_to_targets(distance_matrix, index, (a_index,))
            for index in x_indices
        ]
        component_scores = {
            "A_to_X_CN12": shell_gap_quality(a_to_x, 12),
            "B_to_X_CN6": shell_gap_quality(b_to_x, 6),
            "X_to_B_CN2_mean": float(
                np.mean([shell_gap_quality(values, 2) for values in x_to_b])
            ),
            "X_to_A_CN4_mean": float(
                np.mean([shell_gap_quality(values, 4) for values in x_to_a])
            ),
        }
        topology_score = float(
            sum(component_scores.values())
        )
        shell_estimates = {
            "A": nearest_shell_cn(a_to_x, 12),
            "B": nearest_shell_cn(b_to_x, 6),
            "X": [nearest_shell_cn(values, 2) for values in x_to_b],
        }
        cn_cost = float(
            ((shell_estimates["A"][0] - 12) / 12.0) ** 2
            + ((shell_estimates["B"][0] - 6) / 6.0) ** 2
            + sum(
                ((estimate[0] - 2) / 2.0) ** 2
                for estimate in shell_estimates["X"]
            )
            / 3.0
        )
        candidates.append(
            {
                "a_index": a_index,
                "b_index": b_index,
                "x_indices": x_indices,
                "topology_score": topology_score,
                "cn_cost": cn_cost,
                "parent_role_cost": parent_perovskite_role_cost(
                    structure, a_index, b_index, x_indices
                ),
                "coordinate_key": (
                    fractional_coordinate_key(structure, a_index),
                    fractional_coordinate_key(structure, b_index),
                ),
                "component_scores": component_scores,
                "shell_estimates": shell_estimates,
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -round(candidate["topology_score"], 10),
            round(candidate["cn_cost"], 10),
            round(candidate["parent_role_cost"], 10),
            candidate["coordinate_key"],
            candidate["a_index"],
            candidate["b_index"],
        )
    )
    best = candidates[0]
    second = candidates[1]
    if not math.isfinite(best["topology_score"]):
        raise RuntimeError("No finite crystallographic coordination assignment")

    roles = ["X"] * 5
    roles[best["a_index"]] = "A"
    roles[best["b_index"]] = "B"
    if Counter(roles) != Counter({"A": 1, "B": 1, "X": 3}):
        raise AssertionError(f"Invalid role inventory: {roles}")

    raw_score_margin = float(best["topology_score"] - second["topology_score"])
    score_margin = 0.0 if abs(raw_score_margin) <= 1e-10 else raw_score_margin
    assigned_shell_cn = [None] * 5
    assigned_shell_gap = [None] * 5
    assigned_shell_cn[best["a_index"]] = best["shell_estimates"]["A"][0]
    assigned_shell_gap[best["a_index"]] = best["shell_estimates"]["A"][1]
    assigned_shell_cn[best["b_index"]] = best["shell_estimates"]["B"][0]
    assigned_shell_gap[best["b_index"]] = best["shell_estimates"]["B"][1]
    for x_index, estimate in zip(best["x_indices"], best["shell_estimates"]["X"]):
        assigned_shell_cn[x_index] = estimate[0]
        assigned_shell_gap[x_index] = estimate[1]
    audit = {
        "a_index": int(best["a_index"]),
        "b_index": int(best["b_index"]),
        "x_indices": [int(value) for value in best["x_indices"]],
        "topology_score": float(best["topology_score"]),
        "second_best_score": float(second["topology_score"]),
        "score_margin": score_margin,
        "ambiguous": bool(score_margin <= 1e-10),
        "coordination_tie_broken_by_parent_role": bool(score_margin <= 1e-10),
        "parent_role_cost": float(best["parent_role_cost"]),
        "coordination_component_scores": best["component_scores"],
        "nearest_shell_cn": [int(value) for value in assigned_shell_cn],
        "nearest_shell_gap_ratio": [float(value) for value in assigned_shell_gap],
        "legacy_exact_match": bool(tuple(roles) == LEGACY_INDEX_ROLES),
    }
    return roles, audit


def grouped_role_tokens(structure, roles: list[str]) -> list[tuple[str, str, float]]:
    grouped: defaultdict[tuple[str, str], float] = defaultdict(float)
    for site, role in zip(structure, roles):
        element, occupancy = dominant_element_and_occupancy(site)
        grouped[(element, role)] += occupancy
    total = float(sum(grouped.values()))
    if total <= 0.0:
        raise RuntimeError("Encountered a structure with zero total occupancy")
    return [
        (element, role, float(occupancy / total))
        for (element, role), occupancy in sorted(
            grouped.items(),
            key=lambda item: (ROLE_ORDER.index(item[0][1]), item[0][0]),
        )
    ]


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_name(output.name + ".manifest.json")
    if output.is_file() and manifest_path.is_file() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(output) == manifest.get("cache_sha256"):
            print(f"Validated existing cache: {output}", flush=True)
            print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
            return
        raise RuntimeError("Existing coordination-role cache failed its SHA256 check")

    benchmark = MatbenchBenchmark(autoload=False, subset=[TASK_NAME])
    task = benchmark.tasks_map[TASK_NAME]
    task.load()
    frame = task.df.copy()
    frame.index = frame.index.map(str)
    if len(frame) != EXPECTED_SAMPLE_COUNT or not frame.index.is_unique:
        raise RuntimeError("Unexpected Matbench perovskites inventory")

    material_ids = frame.index.tolist()
    targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    compositions = [
        str(structure.composition.reduced_formula)
        for structure in frame["structure"].tolist()
    ]
    tokens = []
    role_indices = []
    role_signatures: Counter[str] = Counter()
    cn_by_role: dict[str, Counter[int]] = {role: Counter() for role in ROLE_ORDER}
    exact_legacy_matches = 0
    ambiguous_count = 0
    ambiguous_positions = []
    score_margins = []

    for position, structure in enumerate(frame["structure"].tolist()):
        roles, audit = assign_coordination_roles(structure)
        tokens.append(grouped_role_tokens(structure, roles))
        role_indices.append(
            {
                "A": int(audit["a_index"]),
                "B": int(audit["b_index"]),
                "X": audit["x_indices"],
            }
        )
        role_signatures["".join(roles)] += 1
        exact_legacy_matches += int(audit["legacy_exact_match"])
        ambiguous_count += int(audit["ambiguous"])
        if audit["ambiguous"]:
            ambiguous_positions.append(position)
        score_margins.append(float(audit["score_margin"]))
        for site_index, role in enumerate(roles):
            cn_by_role[role][int(audit["nearest_shell_cn"][site_index])] += 1
        if (position + 1) % 1000 == 0:
            print(
                f"Classified {position + 1}/{EXPECTED_SAMPLE_COUNT} structures",
                flush=True,
            )

    payload = {
        "schema_version": "1.0.0",
        "task": TASK_NAME,
        "target_column": TARGET_COLUMN,
        "material_ids": material_ids,
        "compositions": compositions,
        "targets": targets,
        "tokens": tokens,
        "role_indices": role_indices,
        "role_assignment": {
            "method": "target_resolved_periodic_coordination_topology_assignment",
            "species_used_for_role_assignment": False,
            "input_site_order_used_for_role_assignment": False,
            "required_role_inventory": {"A": 1, "B": 1, "X": 3},
            "expected_coordination_numbers": {
                "A_to_X": 12,
                "B_to_X": 6,
                "X_to_B": 2,
                "X_to_A": 4,
            },
            "shell_quality": "log(d[k+1]/d[k])-0.02*log(d[k]/d[1])",
            "global_search": "all 20 ordered choices of A and B sites",
            "tie_break": (
                "nearest ideal parent-perovskite roles, then normalized "
                "fractional-coordinate lexical order"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=3,
            fileobj=raw_handle,
            mtime=0,
        ) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)

    margins = np.asarray(score_margins, dtype=np.float64)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK_NAME,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "cache_path": str(output),
        "cache_bytes": output.stat().st_size,
        "cache_sha256": sha256_file(output),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "material_ids_sha256": sequence_sha256(material_ids),
        "target_sequence_sha256": sequence_sha256(
            [format(float(value), ".9g") for value in targets]
        ),
        "role_assignment": payload["role_assignment"],
        "role_signature_counts": dict(sorted(role_signatures.items())),
        "legacy_index_contract": {"B": [0], "A": [1], "X": [2, 3, 4]},
        "legacy_exact_match_count": exact_legacy_matches,
        "legacy_exact_match_fraction": exact_legacy_matches / EXPECTED_SAMPLE_COUNT,
        "coordination_differs_from_legacy_count": EXPECTED_SAMPLE_COUNT
        - exact_legacy_matches,
        "ambiguous_assignment_count": ambiguous_count,
        "ambiguous_assignment_positions": ambiguous_positions,
        "score_margin_min": float(margins.min()),
        "score_margin_median": float(np.median(margins)),
        "score_margin_max": float(margins.max()),
        "nearest_shell_cn_counts_by_assigned_role": {
            role: {str(key): value for key, value in sorted(counter.items())}
            for role, counter in cn_by_role.items()
        },
    }
    write_json_atomic(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
