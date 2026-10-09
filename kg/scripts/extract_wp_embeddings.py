#!/usr/bin/env python3
"""Extract the 1,731 canonical Wyckoff rows from a full KG embedding table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


WP_ID = re.compile(r"^WP:\d{3}:\d+[A-Za-z]+$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row and WP_ID.fullmatch(row[0])]
    if len(rows) != 1731:
        raise RuntimeError(f"Expected 1731 Wyckoff rows, found {len(rows)}")
    rows.sort(key=lambda row: row[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
