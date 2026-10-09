#!/usr/bin/env python3
"""Build a canonical, composition-aware fixed-text corpus from Alexandria."""
from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil
import spglib
from pymatgen.core import Element, Structure


MAIN_ARCHIVE_COUNT = 58
SCHEMA_VERSION = "1.0.0"
_WP_INFO: dict[tuple[int, str], tuple[str, str, int]] = {}
_HALL_BY_SG: dict[int, int] = {}
_SYMPREC = 0.1
_ANGLE_TOLERANCE = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_paths(root: Path, include_convex_hull: bool) -> list[Path]:
    paths = [root / f"alexandria_{index:05d}.json.bz2" for index in range(MAIN_ARCHIVE_COUNT)]
    if include_convex_hull:
        paths.append(root / "convex_hull.json.bz2")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Alexandria archives: {missing}")
    return paths


def load_entries(path: Path) -> list[dict[str, Any]]:
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return payload["entries"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported Alexandria payload in {path}: {type(payload).__name__}")


def source_id(item: dict[str, Any], archive_name: str, source_index: int) -> str:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    value = data.get("mat_id") or data.get("material_id") or item.get("entry_id") or item.get("id")
    return str(value) if value else f"{archive_name}:{source_index:08d}"


def load_canonical_kg(
    nodes_path: Path,
) -> tuple[dict[tuple[int, str], tuple[str, str, int]], dict[int, int], dict[str, tuple[str, str]]]:
    wp_info: dict[tuple[int, str], tuple[str, str, int]] = {}
    hall_by_sg: dict[int, int] = {}
    facts: dict[str, tuple[str, str]] = {}
    site_tokens: set[str] = set()
    with nodes_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            node_type = row.get("type", "")
            if node_type == "space_group":
                sg = int(row["space_group_number"] or row["number"])
                hall_by_sg[sg] = int(row["hall_number"])
            elif node_type == "wyckoff_position":
                sg = int(row["space_group_number"])
                letter = row["letter"].strip()
                multiplicity = int(row["multiplicity"])
                site_symbol = row["site_symmetry"].strip()
                if not site_symbol:
                    raise ValueError(f"Missing site symmetry for SG {sg} WP {letter}")
                wp_token = f"WP|SG_{sg}|{multiplicity}{letter}"
                ss_token = f"SS_{site_symbol}"
                key = (sg, letter)
                value = (wp_token, ss_token, multiplicity)
                if key in wp_info and wp_info[key] != value:
                    raise ValueError(f"Conflicting canonical WP mapping for {key}")
                wp_info[key] = value
                facts[wp_token] = (f"SG_{sg}", ss_token)
                site_tokens.add(ss_token)
    if len(hall_by_sg) != 230 or len(wp_info) != 1731 or len(site_tokens) != 78:
        raise RuntimeError(
            "Canonical KG must contain 230 SG, 1731 WP, and 78 SS tokens; "
            f"found {len(hall_by_sg)}, {len(wp_info)}, and {len(site_tokens)}"
        )
    return wp_info, hall_by_sg, facts


def init_worker(
    wp_info: dict[tuple[int, str], tuple[str, str, int]],
    hall_by_sg: dict[int, int],
    symprec: float,
    angle_tolerance: float,
) -> None:
    global _WP_INFO, _HALL_BY_SG, _SYMPREC, _ANGLE_TOLERANCE
    _WP_INFO = wp_info
    _HALL_BY_SG = hall_by_sg
    _SYMPREC = symprec
    _ANGLE_TOLERANCE = angle_tolerance


def dataset_value(dataset: Any, name: str) -> Any:
    return getattr(dataset, name) if hasattr(dataset, name) else dataset[name]


def structure_cell(structure: Structure) -> tuple[Any, Any, list[int]]:
    return structure.lattice.matrix, structure.frac_coords, list(structure.atomic_numbers)


def format_text(formula: str, sg_token: str, blocks: list[dict[str, Any]]) -> str:
    parts = [
        f"The chemical formula is {formula}.",
        f"Its crystal structure is in the [{sg_token}] space group.",
    ]
    parts.extend(
        f"{block['multiplicity']} {block['element']} at Wyckoff position [{block['wp']}] "
        f"with site symmetry [{block['ss']}]."
        for block in blocks
    )
    return " ".join(parts)


def formula_training_tokens(structure: Structure) -> list[str]:
    reduced = structure.composition.reduced_composition
    tokens: list[str] = []
    for species, amount in reduced.items():
        symbol = getattr(species, "symbol", str(species))
        rounded = round(float(amount))
        amount_token = str(rounded) if math.isclose(float(amount), rounded) else format(float(amount), ".8g")
        tokens.extend([symbol, amount_token])
    if not tokens:
        raise ValueError("Composition produced no formula tokens")
    return tokens


def training_tokens(
    formula_tokens: list[str], sg_token: str, blocks: list[dict[str, Any]]
) -> list[str]:
    tokens = [
        "The", "chemical", "formula", "is", *formula_tokens,
        "Its", "crystal", "structure", "is", "in", "the", sg_token, "space", "group",
    ]
    for block in blocks:
        tokens.extend(
            [
                str(block["multiplicity"]), block["element"], "at", "Wyckoff", "position", block["wp"],
                "with", "site", "symmetry", block["ss"],
            ]
        )
    return tokens


def profile_worker(job: tuple[str, str, int, dict[str, Any], int | None]) -> dict[str, Any]:
    structure_id, archive_name, source_index, structure_dict, reported_sg = job
    try:
        structure = Structure.from_dict(structure_dict)
        formula = structure.composition.reduced_formula.replace(" ", "")
        formula.encode("ascii")
        formula_tokens = formula_training_tokens(structure)
        source_cell = structure_cell(structure)
        standardized = spglib.standardize_cell(
            source_cell,
            to_primitive=False,
            no_idealize=False,
            symprec=_SYMPREC,
            angle_tolerance=_ANGLE_TOLERANCE,
        )
        if standardized is None:
            raise ValueError("spglib.standardize_cell returned None")
        initial = spglib.get_symmetry_dataset(
            standardized,
            symprec=_SYMPREC,
            angle_tolerance=_ANGLE_TOLERANCE,
        )
        if initial is None:
            raise ValueError("spglib.get_symmetry_dataset returned None")
        sg = int(dataset_value(initial, "number"))
        canonical_hall = _HALL_BY_SG[sg]
        canonical_source = spglib.get_symmetry_dataset(
            standardized,
            symprec=_SYMPREC,
            angle_tolerance=_ANGLE_TOLERANCE,
            hall_number=canonical_hall,
        )
        if canonical_source is None or int(dataset_value(canonical_source, "number")) != sg:
            raise ValueError(f"Could not standardize SG {sg} in canonical Hall {canonical_hall}")
        canonical_cell = (
            dataset_value(canonical_source, "std_lattice"),
            dataset_value(canonical_source, "std_positions"),
            dataset_value(canonical_source, "std_types"),
        )
        dataset = spglib.get_symmetry_dataset(
            canonical_cell,
            symprec=_SYMPREC,
            angle_tolerance=_ANGLE_TOLERANCE,
            hall_number=canonical_hall,
        )
        if dataset is None or int(dataset_value(dataset, "number")) != sg:
            raise ValueError(f"Could not analyze SG {sg} in canonical Hall {canonical_hall}")

        equivalent_atoms = [int(value) for value in dataset_value(dataset, "equivalent_atoms")]
        letters = [str(value).strip() for value in dataset_value(dataset, "wyckoffs")]
        site_symbols = [str(value).strip() for value in dataset_value(dataset, "site_symmetry_symbols")]
        atomic_numbers = [int(value) for value in canonical_cell[2]]
        if not (len(equivalent_atoms) == len(letters) == len(site_symbols) == len(atomic_numbers)):
            raise ValueError("Inconsistent conventional-cell symmetry array lengths")

        blocks: list[dict[str, Any]] = []
        seen_orbits: set[int] = set()
        site_symbol_mismatches = 0
        for index, orbit in enumerate(equivalent_atoms):
            if orbit in seen_orbits:
                continue
            seen_orbits.add(orbit)
            member_indices = [i for i, value in enumerate(equivalent_atoms) if value == orbit]
            orbit_numbers = {atomic_numbers[i] for i in member_indices}
            if len(orbit_numbers) != 1:
                raise ValueError(f"Orbit {orbit} contains inconsistent atomic numbers {orbit_numbers}")
            info = _WP_INFO.get((sg, letters[index]))
            if info is None:
                raise ValueError(f"No canonical WP for SG {sg}, letter {letters[index]!r}")
            wp_token, ss_token, expected_multiplicity = info
            if len(member_indices) != expected_multiplicity:
                raise ValueError(
                    f"Orbit multiplicity {len(member_indices)} != canonical {expected_multiplicity} for {wp_token}"
                )
            observed_ss = f"SS_{site_symbols[index]}"
            site_symbol_mismatches += int(observed_ss != ss_token)
            blocks.append(
                {
                    "element": Element.from_Z(next(iter(orbit_numbers))).symbol,
                    "wp": wp_token,
                    "ss": ss_token,
                    "multiplicity": expected_multiplicity,
                }
            )
        if not blocks:
            raise ValueError("No occupied Wyckoff orbits were identified")
        blocks.sort(key=lambda value: (value["wp"], value["element"], value["ss"]))
        sg_token = f"SG_{sg}"
        tokens = training_tokens(formula_tokens, sg_token, blocks)
        return {
            "ok": True,
            "record": {
                "structure_id": structure_id,
                "source_archive": archive_name,
                "source_index": source_index,
                "formula": formula,
                "formula_training_tokens": formula_tokens,
                "space_group_number": sg,
                "space_group_token": sg_token,
                "canonical_hall_number": canonical_hall,
                "reported_space_group_number": reported_sg,
                "reported_space_group_match": reported_sg is None or int(reported_sg) == sg,
                "orbit_blocks": blocks,
                "site_symbol_mismatches": site_symbol_mismatches,
            },
            "text": format_text(formula, sg_token, blocks),
            "training_line": " ".join(tokens),
            "token_count": len(tokens),
        }
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "structure_id": structure_id,
                "source_archive": archive_name,
                "source_index": source_index,
                "error": repr(exc),
            },
        }


class MemoryMonitor:
    def __init__(self, path: Path, interval_seconds: float) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self.samples = 0
        self.peak_tree_rss = 0
        self.minimum_available = math.inf

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join()
        return {
            "samples": self.samples,
            "peak_process_tree_rss_bytes": self.peak_tree_rss,
            "peak_process_tree_rss_gib": self.peak_tree_rss / 1024**3,
            "minimum_system_available_bytes": int(self.minimum_available),
            "minimum_system_available_gib": self.minimum_available / 1024**3,
            "sample_path": str(self.path),
        }

    def _sample(self) -> dict[str, Any]:
        root = psutil.Process(os.getpid())
        processes = [root, *root.children(recursive=True)]
        rss = 0
        live_pids: list[int] = []
        for process in processes:
            try:
                rss += process.memory_info().rss
                live_pids.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        memory = psutil.virtual_memory()
        self.samples += 1
        self.peak_tree_rss = max(self.peak_tree_rss, rss)
        self.minimum_available = min(self.minimum_available, int(memory.available))
        return {
            "timestamp": utc_now(),
            "process_tree_rss_bytes": rss,
            "system_available_bytes": int(memory.available),
            "system_percent": float(memory.percent),
            "pids": live_pids,
        }

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                handle.write(json.dumps(self._sample(), sort_keys=True) + "\n")
                handle.flush()
                self.stop_event.wait(self.interval_seconds)
            handle.write(json.dumps(self._sample(), sort_keys=True) + "\n")


def load_completed_ids(out_dir: Path, completed_archives: Iterable[str]) -> set[str]:
    identifiers: set[str] = set()
    for archive_name in completed_archives:
        path = out_dir / "processed_ids" / f"{Path(archive_name).stem}.txt.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            identifiers.update(line.strip() for line in handle if line.strip())
    return identifiers


def process_archive(
    path: Path,
    out_dir: Path,
    seen_ids: set[str],
    executor: ProcessPoolExecutor,
    batch_size: int,
    max_records: int | None,
) -> dict[str, Any]:
    entries = load_entries(path)
    if max_records is not None:
        entries = entries[:max_records]
    archive_name = path.name
    stem = path.stem
    outputs = {
        "texts": out_dir / "texts" / f"{stem}.txt.gz",
        "training": out_dir / "training_shards" / f"{stem}.txt.gz",
        "profiles": out_dir / "profiles" / f"{stem}.jsonl.gz",
        "failures": out_dir / "failures" / f"{stem}.jsonl.gz",
        "processed_ids": out_dir / "processed_ids" / f"{stem}.txt.gz",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()}
    counts: Counter[str] = Counter()
    observed_sg: set[str] = set()
    observed_wp: set[str] = set()
    observed_ss: set[str] = set()
    observed_elements: set[str] = set()
    processed_ids: list[str] = []
    duplicate_count = 0

    with gzip.open(temporary["texts"], "wt", encoding="utf-8", newline="\n") as texts, gzip.open(
        temporary["training"], "wt", encoding="ascii", newline="\n"
    ) as training, gzip.open(
        temporary["profiles"], "wt", encoding="utf-8", newline="\n"
    ) as profiles, gzip.open(
        temporary["failures"], "wt", encoding="utf-8", newline="\n"
    ) as failures:
        for batch_start in range(0, len(entries), batch_size):
            jobs: list[tuple[str, str, int, dict[str, Any], int | None]] = []
            for source_index, item in enumerate(
                entries[batch_start : batch_start + batch_size], start=batch_start
            ):
                identifier = source_id(item, archive_name, source_index)
                if identifier in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(identifier)
                processed_ids.append(identifier)
                data = item.get("data") if isinstance(item.get("data"), dict) else {}
                jobs.append((identifier, archive_name, source_index, item["structure"], data.get("spg")))
            for result in executor.map(profile_worker, jobs, chunksize=8):
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
                counts["reported_sg_mismatches"] += int(not record["reported_space_group_match"])
                counts["site_symbol_mismatches"] += int(record["site_symbol_mismatches"])
                observed_sg.add(record["space_group_token"])
                for block in record["orbit_blocks"]:
                    observed_wp.add(block["wp"])
                    observed_ss.add(block["ss"])
                    observed_elements.add(block["element"])
            print(
                f"[{utc_now()}] {archive_name}: analyzed={min(batch_start + batch_size, len(entries))}/"
                f"{len(entries)} success={counts['successful_texts']} failures={counts['failures']} "
                f"duplicates={duplicate_count}",
                flush=True,
            )

    with gzip.open(temporary["processed_ids"], "wt", encoding="utf-8", newline="\n") as handle:
        for identifier in processed_ids:
            handle.write(identifier + "\n")
    for name, destination in outputs.items():
        os.replace(temporary[name], destination)
    return {
        "archive": archive_name,
        "input_records": len(entries),
        "unique_input_records": len(processed_ids),
        "duplicates_skipped": duplicate_count,
        **{key: int(value) for key, value in counts.items()},
        "observed_space_group_tokens": sorted(observed_sg),
        "observed_wp_tokens": sorted(observed_wp),
        "observed_ss_tokens": sorted(observed_ss),
        "observed_elements": sorted(observed_elements),
        "output_paths": {name: str(value) for name, value in outputs.items()},
        "completed_at": utc_now(),
    }


def write_coverage(
    path: Path,
    facts: dict[str, tuple[str, str]],
    repeats: int,
) -> tuple[int, int]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    sentences = 0
    tokens = 0
    with gzip.open(temporary, "wt", encoding="ascii", newline="\n") as handle:
        for wp_token, (sg_token, ss_token) in sorted(facts.items()):
            line_tokens = [
                "The", "Wyckoff", "position", wp_token, "has", "site", "symmetry", ss_token,
                "and", "belongs", "to", "space", "group", sg_token,
            ]
            line = " ".join(line_tokens) + "\n"
            for _ in range(repeats):
                handle.write(line)
                sentences += 1
                tokens += len(line_tokens)
    os.replace(temporary, path)
    return sentences, tokens


def merge_training_shards(out_dir: Path, archive_names: list[str], coverage_path: Path) -> Path:
    destination = out_dir / "sentences.txt"
    temporary = destination.with_suffix(".txt.tmp")
    with temporary.open("wb") as target:
        for archive_name in archive_names:
            shard = out_dir / "training_shards" / f"{Path(archive_name).stem}.txt.gz"
            with gzip.open(shard, "rb") as source:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        with gzip.open(coverage_path, "rb") as source:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, default=Path("alexandria"))
    parser.add_argument("--kg-nodes", type=Path, default=Path("data/kg/nodes.tsv"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/corpora/alexandria_fixed_text_canonical")
    )
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--coverage-repeats", type=int, default=100)
    parser.add_argument("--memory-sample-seconds", type=float, default=5.0)
    parser.add_argument("--min-available-gib", type=float, default=16.0)
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--max-records-per-archive", type=int)
    parser.add_argument("--no-convex-hull", action="store_true")
    parser.add_argument("--max-failure-fraction", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Generation workers must be between 1 and 4")
    if args.coverage_repeats < 1:
        raise ValueError("coverage-repeats must be positive")
    if psutil.virtual_memory().available < args.min_available_gib * 1024**3:
        raise MemoryError(f"Less than {args.min_available_gib} GiB memory is available")

    all_archives = archive_paths(args.archives_dir, include_convex_hull=not args.no_convex_hull)
    selected_archives = all_archives[: args.max_archives] if args.max_archives else all_archives
    wp_info, hall_by_sg, facts = load_canonical_kg(args.kg_nodes)
    generator_config = {
        "archives_dir": str(args.archives_dir),
        "kg_nodes": str(args.kg_nodes),
        "symprec": args.symprec,
        "angle_tolerance": args.angle_tolerance,
        "standardize_conventional": True,
        "canonical_hall": True,
        "canonical_cell_source": "spglib canonical-Hall dataset std_lattice/std_positions/std_types",
        "workers": args.workers,
        "batch_size": args.batch_size,
        "coverage_repeats": args.coverage_repeats,
        "selected_archives": [path.name for path in selected_archives],
        "max_records_per_archive": args.max_records_per_archive,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("texts", "training_shards", "profiles", "failures", "processed_ids"):
        (args.out_dir / name).mkdir(exist_ok=True)
    state_path = args.out_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["generator_config"] != generator_config:
            raise RuntimeError("Existing state has different generator settings")
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "generator_config": generator_config,
            "completed_archives": [],
            "archives": {},
        }
        write_json(state_path, state)

    monitor = MemoryMonitor(args.out_dir / "memory_usage.jsonl", args.memory_sample_seconds)
    monitor.start()
    memory_summary: dict[str, Any] | None = None
    try:
        seen_ids = load_completed_ids(args.out_dir, state["completed_archives"])
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=init_worker,
            initargs=(wp_info, hall_by_sg, args.symprec, args.angle_tolerance),
        ) as executor:
            for archive in selected_archives:
                if archive.name in state["completed_archives"]:
                    print(f"[{utc_now()}] skipping completed archive {archive.name}", flush=True)
                    continue
                if psutil.virtual_memory().available < args.min_available_gib * 1024**3:
                    raise MemoryError(f"Available memory fell below {args.min_available_gib} GiB")
                stats = process_archive(
                    archive,
                    args.out_dir,
                    seen_ids,
                    executor,
                    args.batch_size,
                    args.max_records_per_archive,
                )
                state["archives"][archive.name] = stats
                state["completed_archives"].append(archive.name)
                state["updated_at"] = utc_now()
                write_json(state_path, state)
        coverage_path = args.out_dir / "canonical_coverage.txt.gz"
        coverage_sentences, coverage_tokens = write_coverage(
            coverage_path, facts, args.coverage_repeats
        )
        sentence_path = merge_training_shards(
            args.out_dir, state["completed_archives"], coverage_path
        )
    finally:
        memory_summary = monitor.stop()
        write_json(args.out_dir / "memory_summary.json", memory_summary)

    archive_stats = [state["archives"][name] for name in state["completed_archives"]]
    totals: Counter[str] = Counter()
    observed_sg: set[str] = set()
    observed_wp: set[str] = set()
    observed_ss: set[str] = set()
    observed_elements: set[str] = set()
    for row in archive_stats:
        for key in (
            "input_records", "unique_input_records", "duplicates_skipped", "successful_texts",
            "failures", "training_token_occurrences", "orbit_blocks", "reported_sg_mismatches",
            "site_symbol_mismatches",
        ):
            totals[key] += int(row.get(key, 0))
        observed_sg.update(row["observed_space_group_tokens"])
        observed_wp.update(row["observed_wp_tokens"])
        observed_ss.update(row["observed_ss_tokens"])
        observed_elements.update(row["observed_elements"])
    unique_records = totals["successful_texts"] + totals["failures"]
    failure_fraction = totals["failures"] / unique_records if unique_records else 0.0
    if failure_fraction > args.max_failure_fraction:
        raise RuntimeError(
            f"Failure fraction {failure_fraction:.6g} exceeds {args.max_failure_fraction:.6g}"
        )
    full_release = (
        len(selected_archives) == len(all_archives)
        and args.max_records_per_archive is None
        and len(state["completed_archives"]) == len(all_archives)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_type": "alexandria_canonical_fixed_text",
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
        "sentences_sha256": sha256_file(sentence_path),
        "memory_monitor": memory_summary,
        "completed_at": utc_now(),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
