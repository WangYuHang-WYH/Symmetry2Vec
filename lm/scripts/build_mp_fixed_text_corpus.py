#!/usr/bin/env python3
"""Build a canonical composition-aware fixed-text corpus from official MP CIFs."""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import shutil
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import psutil
from pymatgen.core import Structure

import build_alexandria_fixed_text_corpus as base


SCHEMA_VERSION = "1.0.0"


def init_mp_worker(
    wp_info: dict[tuple[int, str], tuple[str, str, int]],
    hall_by_sg: dict[int, int],
    symprec: float,
    angle_tolerance: float,
) -> None:
    base.init_worker(wp_info, hall_by_sg, symprec, angle_tolerance)


def mp_profile_worker(job: tuple[str, int, str]) -> dict[str, Any]:
    material_id, source_index, cif_path = job
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = Structure.from_file(cif_path)
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "structure_id": material_id,
                "source_archive": "materials_project_official_api",
                "source_index": source_index,
                "error": f"CIF read failed: {exc!r}",
            },
        }
    return base.profile_worker(
        (
            material_id,
            "materials_project_official_api",
            source_index,
            structure.as_dict(),
            None,
        )
    )


def load_group_ids(source_dir: Path, group_name: str) -> list[str]:
    path = source_dir / "material_ids" / f"{group_name}.txt.gz"
    with gzip.open(path, "rt", encoding="ascii") as handle:
        identifiers = [line.strip() for line in handle if line.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"Duplicate material IDs in {path}")
    return identifiers


def process_group(
    group_name: str,
    material_ids: list[str],
    source_dir: Path,
    out_dir: Path,
    executor: ProcessPoolExecutor,
    batch_size: int,
) -> dict[str, Any]:
    outputs = {
        "texts": out_dir / "texts" / f"{group_name}.txt.gz",
        "training": out_dir / "training_shards" / f"{group_name}.txt.gz",
        "profiles": out_dir / "profiles" / f"{group_name}.jsonl.gz",
        "failures": out_dir / "failures" / f"{group_name}.jsonl.gz",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()}
    counts: Counter[str] = Counter()
    observed_sg: set[str] = set()
    observed_wp: set[str] = set()
    observed_ss: set[str] = set()
    observed_elements: set[str] = set()
    with gzip.open(temporary["texts"], "wt", encoding="utf-8", newline="\n") as texts, gzip.open(
        temporary["training"], "wt", encoding="ascii", newline="\n"
    ) as training, gzip.open(
        temporary["profiles"], "wt", encoding="utf-8", newline="\n"
    ) as profiles, gzip.open(
        temporary["failures"], "wt", encoding="utf-8", newline="\n"
    ) as failures:
        for batch_start in range(0, len(material_ids), batch_size):
            batch_ids = material_ids[batch_start : batch_start + batch_size]
            jobs = [
                (material_id, batch_start + offset, str(source_dir / "cifs" / f"{material_id}.cif"))
                for offset, material_id in enumerate(batch_ids)
            ]
            for result in executor.map(mp_profile_worker, jobs, chunksize=8):
                if not result["ok"]:
                    counts["failures"] += 1
                    failures.write(json.dumps(result["failure"], sort_keys=True) + "\n")
                    continue
                record = result["record"]
                texts.write(result["text"] + "\n")
                training.write(result["training_line"] + "\n")
                profiles.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                counts["successful_texts"] += 1
                counts["training_token_occurrences"] += int(result["token_count"])
                counts["orbit_blocks"] += len(record["orbit_blocks"])
                counts["site_symbol_mismatches"] += int(record["site_symbol_mismatches"])
                observed_sg.add(record["space_group_token"])
                for block in record["orbit_blocks"]:
                    observed_wp.add(block["wp"])
                    observed_ss.add(block["ss"])
                    observed_elements.add(block["element"])
            print(
                f"[{base.utc_now()}] {group_name}: analyzed={min(batch_start + batch_size, len(material_ids))}/"
                f"{len(material_ids)} success={counts['successful_texts']} failures={counts['failures']}",
                flush=True,
            )
    for name, destination in outputs.items():
        os.replace(temporary[name], destination)
    return {
        "group": group_name,
        "input_records": len(material_ids),
        **{key: int(value) for key, value in counts.items()},
        "observed_space_group_tokens": sorted(observed_sg),
        "observed_wp_tokens": sorted(observed_wp),
        "observed_ss_tokens": sorted(observed_ss),
        "observed_elements": sorted(observed_elements),
        "output_paths": {name: str(value) for name, value in outputs.items()},
        "completed_at": base.utc_now(),
    }


def merge_training_shards(out_dir: Path, group_names: list[str], coverage_path: Path) -> Path:
    destination = out_dir / "sentences.txt"
    temporary = destination.with_suffix(".txt.tmp")
    with temporary.open("wb") as target:
        for group_name in group_names:
            with gzip.open(out_dir / "training_shards" / f"{group_name}.txt.gz", "rb") as source:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        with gzip.open(coverage_path, "rb") as source:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/mp_official_cifs_current"))
    parser.add_argument("--kg-nodes", type=Path, default=Path("data/kg/nodes.tsv"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/corpora/mp_official_fixed_text_canonical_hall_coverage_v2"),
    )
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--coverage-repeats", type=int, default=100)
    parser.add_argument("--memory-sample-seconds", type=float, default=5.0)
    parser.add_argument("--min-available-gib", type=float, default=16.0)
    parser.add_argument("--max-failure-fraction", type=float, default=0.001)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-records-per-group", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Generation workers must be between 1 and 4")
    if args.coverage_repeats < 1 or args.batch_size < 1:
        raise ValueError("coverage-repeats and batch-size must be positive")
    if psutil.virtual_memory().available < args.min_available_gib * 1024**3:
        raise MemoryError(f"Less than {args.min_available_gib} GiB memory is available")
    source_manifest_path = args.source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_manifest.get("complete") or not source_manifest.get("full_non_deprecated_collection"):
        raise ValueError("A complete official non-deprecated MP CIF collection is required")
    all_group_names = [row["group"] for row in source_manifest["groups"]]
    group_names = all_group_names[: args.max_groups] if args.max_groups else all_group_names
    wp_info, hall_by_sg, facts = base.load_canonical_kg(args.kg_nodes)
    generator_config = {
        "source_dir": str(args.source_dir),
        "source_manifest_sha256": base.sha256_file(source_manifest_path),
        "kg_nodes": str(args.kg_nodes),
        "symprec": args.symprec,
        "angle_tolerance": args.angle_tolerance,
        "standardize_conventional": True,
        "canonical_hall": True,
        "canonical_cell_source": "spglib canonical-Hall dataset std_lattice/std_positions/std_types",
        "workers": args.workers,
        "batch_size": args.batch_size,
        "coverage_repeats": args.coverage_repeats,
        "coverage_template": "The Wyckoff position WP has site symmetry SS and belongs to space group SG",
        "coverage_token_distances": {"WP_to_SS": 4, "WP_to_SG": 10},
        "selected_groups": group_names,
        "max_records_per_group": args.max_records_per_group,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("texts", "training_shards", "profiles", "failures"):
        (args.out_dir / name).mkdir(exist_ok=True)
    state_path = args.out_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["generator_config"] != generator_config:
            raise RuntimeError("Existing state has different generator settings")
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "created_at": base.utc_now(),
            "generator_config": generator_config,
            "completed_groups": [],
            "groups": {},
        }
        base.write_json(state_path, state)

    monitor = base.MemoryMonitor(args.out_dir / "memory_usage.jsonl", args.memory_sample_seconds)
    monitor.start()
    memory_summary: dict[str, Any] | None = None
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=init_mp_worker,
            initargs=(wp_info, hall_by_sg, args.symprec, args.angle_tolerance),
        ) as executor:
            for group_name in group_names:
                if group_name in state["completed_groups"]:
                    print(f"[{base.utc_now()}] skipping completed group {group_name}", flush=True)
                    continue
                if psutil.virtual_memory().available < args.min_available_gib * 1024**3:
                    raise MemoryError(f"Available memory fell below {args.min_available_gib} GiB")
                material_ids = load_group_ids(args.source_dir, group_name)
                if args.max_records_per_group is not None:
                    material_ids = material_ids[: args.max_records_per_group]
                missing = [
                    material_id
                    for material_id in material_ids
                    if not (args.source_dir / "cifs" / f"{material_id}.cif").is_file()
                ]
                if missing:
                    raise FileNotFoundError(f"Missing {len(missing)} source CIFs; first={missing[0]}")
                stats = process_group(
                    group_name, material_ids, args.source_dir, args.out_dir, executor, args.batch_size
                )
                state["groups"][group_name] = stats
                state["completed_groups"].append(group_name)
                state["updated_at"] = base.utc_now()
                base.write_json(state_path, state)
        coverage_path = args.out_dir / "canonical_coverage.txt.gz"
        coverage_sentences, coverage_tokens = base.write_coverage(
            coverage_path, facts, args.coverage_repeats
        )
        sentence_path = merge_training_shards(args.out_dir, state["completed_groups"], coverage_path)
    finally:
        memory_summary = monitor.stop()
        base.write_json(args.out_dir / "memory_summary.json", memory_summary)

    group_stats = [state["groups"][name] for name in state["completed_groups"]]
    totals: Counter[str] = Counter()
    observed_sg: set[str] = set()
    observed_wp: set[str] = set()
    observed_ss: set[str] = set()
    observed_elements: set[str] = set()
    for row in group_stats:
        for key in (
            "input_records", "successful_texts", "failures", "training_token_occurrences",
            "orbit_blocks", "site_symbol_mismatches",
        ):
            totals[key] += int(row.get(key, 0))
        observed_sg.update(row["observed_space_group_tokens"])
        observed_wp.update(row["observed_wp_tokens"])
        observed_ss.update(row["observed_ss_tokens"])
        observed_elements.update(row["observed_elements"])
    failure_fraction = totals["failures"] / totals["input_records"] if totals["input_records"] else 0.0
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"Failure fraction {failure_fraction:.6g} exceeds {args.max_failure_fraction:.6g}"
        )
    full_release = (
        group_names == all_group_names
        and args.max_records_per_group is None
        and state["completed_groups"] == all_group_names
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_type": "mp_official_canonical_fixed_text_coverage_v2",
        "source_dataset": "Materials Project current non-deprecated Summary collection",
        "source_database_version": source_manifest.get("database_version"),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": base.sha256_file(source_manifest_path),
        "composition_aware": True,
        "pure_symmetry_only": False,
        "complete": True,
        "full_release": full_release,
        "generator_config": generator_config,
        "fixed_text_template": (
            "The chemical formula is {formula}. Its crystal structure is in the [SG] space group. "
            "{WP multiplicity} {element} at Wyckoff position [WP] with site symmetry [SS]."
        ),
        "training_tokenization": (
            "whitespace tokens with brackets and sentence punctuation removed; formulae are split into "
            "element-symbol and stoichiometric-number tokens"
        ),
        "counts": {
            **{key: int(value) for key, value in totals.items()},
            "canonical_coverage_sentences": coverage_sentences,
            "n_sentences": int(totals["successful_texts"]) + coverage_sentences,
            "token_occurrences": int(totals["training_token_occurrences"]) + coverage_tokens,
        },
        "failure_fraction": failure_fraction,
        "canonical_vocabulary": {
            "space_groups": 230,
            "wyckoff_positions": 1731,
            "site_symmetries": 78,
            "training_coverage_complete": True,
        },
        "empirical_vocabulary": {
            "space_groups": len(observed_sg),
            "wyckoff_positions": len(observed_wp),
            "site_symmetries": len(observed_ss),
            "elements": len(observed_elements),
        },
        "metadata_only_fields": ["structure_id", "source_archive", "source_index"],
        "failed_samples_in_training": 0,
        "sentence_path": str(sentence_path),
        "sentences_bytes": sentence_path.stat().st_size,
        "sentences_sha256": base.sha256_file(sentence_path),
        "memory_monitor": memory_summary,
        "completed_at": base.utc_now(),
    }
    base.write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
