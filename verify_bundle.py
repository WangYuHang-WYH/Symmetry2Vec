#!/usr/bin/env python3
"""Validate the portable bundle and regenerate its SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CODE_ROOT / "data"

TABLES = {
    "kg": (DATA_ROOT / "kg" / "entity_embeddings.tsv", 1731, 200),
    "lm": (
        DATA_ROOT / "lm" / "entity_embeddings.tsv",
        1731,
        200,
    ),
    "wren": (
        CODE_ROOT
        / "downstream"
        / "data"
        / "representations"
        / "wren444"
        / "entity_embeddings.tsv",
        1731,
        444,
    ),
    "onehot": (
        CODE_ROOT
        / "downstream"
        / "data"
        / "representations"
        / "onehot_1731"
        / "entity_embeddings.tsv",
        1731,
        1731,
    ),
}

EXPECTED_HASHES = {
    "data/kg/entity_embeddings.tsv": (
        "8672f13f2b2fdea234ddcdd5b38571e8ba03c42a512993f6beaecc4f3a5995ea"
    ),
    "data/kg/entity_embeddings_full.tsv": (
        "700849834ce3bfc4348eba4fb8393dddbd1001d9b94c7f6b06526bfcbe5afc62"
    ),
    "data/lm/entity_embeddings.tsv": (
        "dbf6431380b67535e441f7c7632e7aeca1320d85ffeedab75cc0f59489a7ed3c"
    ),
    "data/lm/wp_training_coverage.tsv": (
        "66800000b5ff44c2dfd28b49d08d00d67835003e09a8a4db2e538734f45a55df"
    ),
    "downstream/data/representations/wren444/entity_embeddings.tsv": (
        "c3051949c2ac55fab7e9f03fc444e28a981689c331634ffc88a352a0528cba99"
    ),
    "downstream/data/representations/onehot_1731/entity_embeddings.tsv": (
        "b42a493886dbe59c92c4b08756a50d62dd6133fad07c9a0a55dd7dbaf9e9e630"
    ),
    "lm/data/corpus/train.jsonl.gz": (
        "8bea52a4535552166cd5b0698c802038c36c30ef9b155ae07fc9a552bd33d23a"
    ),
    "lm/data/corpus/validation.jsonl.gz": (
        "8d383bf845962390480f81c0b93277ac89164fa66511c2b8eb6ff29dcd6da20f"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_table(path: Path, expected_rows: int, expected_dim: int) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = sum(1 for row in reader if row)
    if not header or header[0] != "id" or len(header) - 1 != expected_dim:
        raise RuntimeError(f"Invalid table header or dimension: {path}")
    if rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}, found {rows}")


def main() -> None:
    for path, rows, dimension in TABLES.values():
        validate_table(path, rows, dimension)

    registry = json.loads(
        (CODE_ROOT / "downstream" / "vector_registry.json").read_text(encoding="utf-8")
    )
    for name, metadata in registry["representations"].items():
        table = CODE_ROOT / "downstream" / metadata["path"]
        if not table.resolve().is_file():
            raise FileNotFoundError(f"Registry table for {name}: {table}")
        coverage = metadata.get("coverage")
        if coverage and not (CODE_ROOT / "downstream" / coverage).resolve().is_file():
            raise FileNotFoundError(f"Registry coverage for {name}: {coverage}")

    for relative, expected in EXPECTED_HASHES.items():
        path = CODE_ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")

    bert_weights = (
        CODE_ROOT / "lm" / "resources" / "bert-base-cased" / "model.safetensors"
    )
    if bert_weights.stat().st_size < 400_000_000:
        raise RuntimeError("Packaged bert-base-cased weights are incomplete")

    manifest_path = CODE_ROOT / "manifest.sha256"
    files = sorted(
        path
        for path in CODE_ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(CODE_ROOT).as_posix()}" for path in files]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "validated_tables": sorted(TABLES),
                "expected_hashes_checked": len(EXPECTED_HASHES),
                "manifest_files": len(files),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
