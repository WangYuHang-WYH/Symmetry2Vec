#!/usr/bin/env python3
"""Build an unlabeled fixed-text MP corpus for Wyckoff-position MLM."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl_gz(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_wp_vocabulary(path: Path) -> tuple[list[str], dict[str, str]]:
    rows: list[tuple[int, str, int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("type") != "wyckoff_position":
                continue
            entity_id = row["id"]
            sg_number = int(row["space_group_number"])
            multiplicity = int(row["multiplicity"])
            letter = row["letter"]
            surface = f"WP|SG_{sg_number}|{multiplicity}{letter}"
            rows.append((sg_number, letter, multiplicity, entity_id, surface))
    rows.sort(key=lambda value: (value[0], value[1], value[2]))
    entity_ids = [row[3] for row in rows]
    surface_by_entity = {row[3]: row[4] for row in rows}
    if len(rows) != 1731 or len(set(entity_ids)) != 1731:
        raise RuntimeError(f"Expected 1731 canonical WP rows, found {len(rows)}")
    if len(set(surface_by_entity.values())) != 1731:
        raise RuntimeError("Canonical WP surface tokens are not unique")
    return entity_ids, surface_by_entity


def preliminary_split(structure_id: str, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{structure_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def fixed_text_tokens(record: dict[str, Any]) -> list[str]:
    tokens = [
        "The",
        "chemical",
        "formula",
        "is",
        *[str(value) for value in record["formula_training_tokens"]],
        ".",
        "Its",
        "crystal",
        "structure",
        "is",
        "in",
        "the",
        record["space_group_token"],
        "space",
        "group",
        ".",
    ]
    for block in record["orbit_blocks"]:
        tokens.extend(
            [
                str(block["multiplicity"]),
                block["element"],
                "at",
                "Wyckoff",
                "position",
                block["wp"],
                "with",
                "site",
                "symmetry",
                block["ss"],
                ".",
            ]
        )
    if any(not token or any(character.isspace() for character in token) for token in tokens):
        raise ValueError("Each stored word must be non-empty and whitespace-free")
    return tokens


def render_text(tokens: list[str]) -> str:
    return " ".join(tokens).replace(" .", ".")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus = config["corpus"]
    source_dir = Path(corpus["source_profile_dir"])
    profile_paths = sorted(source_dir.glob("*.jsonl.gz"))
    if not profile_paths:
        raise FileNotFoundError(f"No profile shards found in {source_dir}")
    output_dir = Path(corpus["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_fraction = float(corpus["validation_fraction"])
    seed = int(config["seed"])
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    entity_ids, surface_by_entity = load_wp_vocabulary(Path(config["wp_nodes"]))
    entity_by_surface = {value: key for key, value in surface_by_entity.items()}
    preliminary_train_counts: Counter[str] = Counter()
    validation_candidates: dict[str, str] = {}
    input_records = 0
    observed_wps: set[str] = set()
    for record in iter_jsonl_gz(profile_paths):
        input_records += 1
        structure_id = str(record["structure_id"])
        split = preliminary_split(structure_id, validation_fraction, seed)
        for block in record["orbit_blocks"]:
            surface = block["wp"]
            if surface not in entity_by_surface:
                raise RuntimeError(f"Unknown canonical WP token: {surface}")
            entity_id = entity_by_surface[surface]
            observed_wps.add(entity_id)
            if split == "train":
                preliminary_train_counts[entity_id] += 1
            else:
                validation_candidates.setdefault(entity_id, structure_id)

    missing_from_train = sorted(observed_wps - set(preliminary_train_counts))
    force_train_ids = {
        validation_candidates[entity_id]
        for entity_id in missing_from_train
        if entity_id in validation_candidates
    }
    if len(force_train_ids) != len(missing_from_train):
        raise RuntimeError("Could not rescue every observed WP into the training split")

    split_paths = {
        split: output_dir / f"{split}.jsonl.gz" for split in ("train", "validation")
    }
    temporary_paths = {
        split: path.with_suffix(path.suffix + ".tmp") for split, path in split_paths.items()
    }
    handles = {
        split: gzip.open(path, "wt", encoding="utf-8", newline="\n")
        for split, path in temporary_paths.items()
    }
    split_samples: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    split_wp_occurrences: Counter[str] = Counter()
    split_wp_sets: dict[str, set[str]] = {"train": set(), "validation": set()}
    vocabulary: set[str] = set()
    try:
        for record in iter_jsonl_gz(profile_paths):
            structure_id = str(record["structure_id"])
            split = preliminary_split(structure_id, validation_fraction, seed)
            if structure_id in force_train_ids:
                split = "train"
            tokens = fixed_text_tokens(record)
            wp_entities = [entity_by_surface[block["wp"]] for block in record["orbit_blocks"]]
            payload = {
                "text": render_text(tokens),
                "tokens": tokens,
                "wp_entities": wp_entities,
            }
            handles[split].write(json.dumps(payload, separators=(",", ":")) + "\n")
            split_samples[split] += 1
            split_tokens[split] += len(tokens)
            split_wp_occurrences[split] += len(wp_entities)
            split_wp_sets[split].update(wp_entities)
            vocabulary.update(tokens)
    finally:
        for handle in handles.values():
            handle.close()
    for split, destination in split_paths.items():
        temporary_paths[split].replace(destination)

    if sum(split_samples.values()) != input_records:
        raise RuntimeError("Output sample count does not match input profiles")
    if split_wp_sets["train"] != observed_wps:
        raise RuntimeError("Training split does not cover every empirically observed WP")

    vocabulary_path = output_dir / "canonical_wp_vocabulary.tsv"
    with vocabulary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_index", "entity_id", "surface_token", "observed_in_mp"])
        for index, entity_id in enumerate(entity_ids):
            writer.writerow(
                [index, entity_id, surface_by_entity[entity_id], int(entity_id in observed_wps)]
            )

    manifest = {
        "schema_version": "1.0.0",
        "corpus_type": "mp_official_fixed_text_wp_mlm",
        "source_profile_dir": str(source_dir),
        "profile_shards": len(profile_paths),
        "labels_used": False,
        "canonical_coverage_sentences_used": False,
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "forced_train_structures_for_wp_coverage": len(force_train_ids),
        "template": (
            "The chemical formula is {element amount tokens}. Its crystal structure is "
            "in the SG space group. {multiplicity} {element} at Wyckoff position WP "
            "with site symmetry SS."
        ),
        "token_contract": "Stored tokens are exact pre-tokenized words; period is one token.",
        "counts": {
            "input_structures": input_records,
            "train_structures": split_samples["train"],
            "validation_structures": split_samples["validation"],
            "train_word_tokens": split_tokens["train"],
            "validation_word_tokens": split_tokens["validation"],
            "train_wp_occurrences": split_wp_occurrences["train"],
            "validation_wp_occurrences": split_wp_occurrences["validation"],
            "canonical_wp_rows": len(entity_ids),
            "observed_mp_wp_rows": len(observed_wps),
            "unobserved_mp_wp_rows": len(entity_ids) - len(observed_wps),
            "corpus_word_vocabulary": len(vocabulary),
        },
        "split_wp_coverage": {
            split: len(values) for split, values in split_wp_sets.items()
        },
        "files": {
            split: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for split, path in split_paths.items()
        },
        "wp_vocabulary": {
            "path": str(vocabulary_path),
            "sha256": sha256_file(vocabulary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
