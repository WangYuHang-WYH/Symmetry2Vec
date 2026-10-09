from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx


Triple = tuple[str, str, str]


def _numbers(*items: int | tuple[int, int]) -> frozenset[int]:
    values: set[int] = set()
    for item in items:
        if isinstance(item, int):
            values.add(item)
        else:
            start, stop = item
            values.update(range(start, stop + 1))
    return frozenset(values)


# Standard 14 Bravais lattices indexed only by International space-group type.
# A/B/C settings are one base-centered orthorhombic or monoclinic lattice.
BRAVAIS_SPACE_GROUPS: dict[str, frozenset[int]] = {
    "aP": _numbers((1, 2)),
    "mP": _numbers(3, 4, 6, 7, 10, 11, 13, 14),
    "mC": _numbers(5, 8, 9, 12, 15),
    "oP": _numbers((16, 19), (25, 34), (47, 62)),
    "oC": _numbers(20, 21, (35, 41), (63, 68)),
    "oF": _numbers(22, 42, 43, 69, 70),
    "oI": _numbers(23, 24, (44, 46), (71, 74)),
    "tP": _numbers((75, 78), 81, (83, 86), (89, 96), (99, 106), (111, 118), (123, 138)),
    "tI": _numbers(79, 80, 82, 87, 88, 97, 98, (107, 110), (119, 122), (139, 142)),
    "hP": _numbers((143, 145), 147, (149, 154), (156, 159), (162, 165), (168, 194)),
    "hR": _numbers(146, 148, 155, 160, 161, 166, 167),
    "cP": _numbers(195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, (221, 224)),
    "cF": _numbers(196, 202, 203, 209, 210, 216, 219, (225, 228)),
    "cI": _numbers(197, 199, 204, 206, 211, 214, 217, 220, 229, 230),
}

EXPECTED_BRAVAIS_COUNTS = {
    "aP": 2,
    "mP": 8,
    "mC": 5,
    "oP": 30,
    "oC": 15,
    "oF": 5,
    "oI": 9,
    "tP": 49,
    "tI": 19,
    "hP": 45,
    "hR": 7,
    "cP": 15,
    "cF": 11,
    "cI": 10,
}

BRAVAIS_METADATA = {
    "aP": ("triclinic", "P", "triclinic primitive"),
    "mP": ("monoclinic", "P", "monoclinic primitive"),
    "mC": ("monoclinic", "C", "monoclinic base-centered"),
    "oP": ("orthorhombic", "P", "orthorhombic primitive"),
    "oC": ("orthorhombic", "C", "orthorhombic base-centered"),
    "oF": ("orthorhombic", "F", "orthorhombic face-centered"),
    "oI": ("orthorhombic", "I", "orthorhombic body-centered"),
    "tP": ("tetragonal", "P", "tetragonal primitive"),
    "tI": ("tetragonal", "I", "tetragonal body-centered"),
    "hP": ("hexagonal", "P", "hexagonal primitive"),
    "hR": ("rhombohedral", "R", "rhombohedral"),
    "cP": ("cubic", "P", "cubic primitive"),
    "cF": ("cubic", "F", "cubic face-centered"),
    "cI": ("cubic", "I", "cubic body-centered"),
}


def _build_space_group_lookup() -> dict[int, str]:
    lookup: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    for symbol, numbers in BRAVAIS_SPACE_GROUPS.items():
        if len(numbers) != EXPECTED_BRAVAIS_COUNTS[symbol]:
            raise RuntimeError(f"Unexpected {symbol} count: {len(numbers)}")
        for number in numbers:
            if number in lookup:
                duplicates.setdefault(number, [lookup[number]]).append(symbol)
            lookup[number] = symbol
    expected = set(range(1, 231))
    if duplicates or set(lookup) != expected:
        missing = sorted(expected.difference(lookup))
        extra = sorted(set(lookup).difference(expected))
        raise RuntimeError(
            f"Invalid standard Bravais mapping: duplicates={duplicates}, "
            f"missing={missing}, extra={extra}"
        )
    if len(BRAVAIS_SPACE_GROUPS) != 14:
        raise RuntimeError("The mapping must contain exactly 14 Bravais lattices")
    return lookup


SPACE_GROUP_TO_BRAVAIS = _build_space_group_lookup()


def standard_bravais_lattice(space_group_number: int) -> str:
    try:
        return SPACE_GROUP_TO_BRAVAIS[int(space_group_number)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Space-group number must be in 1..230: {space_group_number}") from exc


def read_existing_graph(graph_dir: Path) -> tuple[dict[str, dict[str, Any]], list[Triple]]:
    nodes_path = graph_dir / "nodes.tsv"
    triples_path = graph_dir / "triples.tsv"
    if not nodes_path.is_file() or not triples_path.is_file():
        raise FileNotFoundError(
            f"Base KG source requires nodes.tsv and triples.tsv: {graph_dir}"
        )
    nodes: dict[str, dict[str, Any]] = {}
    with nodes_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            node_id = row.pop("id")
            if node_id in nodes:
                raise RuntimeError(f"Duplicate base KG node: {node_id}")
            nodes[node_id] = {key: value for key, value in row.items() if value != ""}
    triples: list[Triple] = []
    with triples_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["head", "relation", "tail"]:
            raise RuntimeError(f"Unexpected base KG triple header: {reader.fieldnames}")
        for row in reader:
            triples.append((row["head"], row["relation"], row["tail"]))
    if len(nodes) != 5691 or len(triples) != 49645 or len(set(triples)) != len(triples):
        raise RuntimeError(
            f"Unexpected base KG inventory: nodes={len(nodes)} triples={len(triples)} "
            f"unique_triples={len(set(triples))}"
        )
    return nodes, triples


def _jsonable(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def write_outputs(nodes: dict[str, dict[str, Any]], triples: list[Triple], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_keys = sorted({key for attrs in nodes.values() for key in attrs})
    with (out_dir / "nodes.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", *attr_keys],
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, attrs in sorted(nodes.items()):
            writer.writerow(
                {"id": node_id, **{key: _jsonable(value) for key, value in attrs.items()}}
            )
    with (out_dir / "triples.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["head", "relation", "tail"])
        writer.writerows(triples)
    relation_counts = Counter(relation for _, relation, _ in triples)
    with (out_dir / "relations.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relation", "count"])
        writer.writerows(sorted(relation_counts.items()))
    graph = nx.MultiDiGraph()
    for node_id, attrs in nodes.items():
        graph.add_node(
            node_id, **{key: _jsonable(value) for key, value in attrs.items()}
        )
    for head, relation, tail in triples:
        graph.add_edge(head, tail, relation=relation)
    nx.write_graphml(graph, out_dir / "graph.graphml")
    summary = {
        "num_nodes": len(nodes),
        "num_triples": len(triples),
        "node_types": Counter(attrs.get("type", "unknown") for attrs in nodes.values()),
        "relation_counts": relation_counts,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replace_lattice_subgraph(
    nodes: dict[str, dict[str, Any]], triples: list[Triple]
) -> tuple[dict[str, dict[str, Any]], list[Triple], dict[str, Any]]:
    legacy_lattice_nodes = {
        node_id for node_id, attrs in nodes.items() if attrs.get("type") == "lattice_type"
    }
    legacy_lattice_triples = {
        triple for triple in triples if triple[1] == "has_lattice_type"
    }
    unexpected_references = [
        triple
        for triple in triples
        if triple[1] != "has_lattice_type"
        and (triple[0] in legacy_lattice_nodes or triple[2] in legacy_lattice_nodes)
    ]
    if unexpected_references:
        raise RuntimeError("Legacy lattice nodes participate in unexpected relations")

    updated_nodes = {
        node_id: dict(attrs)
        for node_id, attrs in nodes.items()
        if node_id not in legacy_lattice_nodes
    }
    updated_triples = {
        triple for triple in triples if triple[1] != "has_lattice_type"
    }
    mapped_space_groups: set[str] = set()
    mapping_rows: list[dict[str, Any]] = []

    for number in range(1, 231):
        sg_id = f"SG:{number:03d}"
        if sg_id not in updated_nodes:
            raise RuntimeError(f"Missing space-group node: {sg_id}")
        symbol = standard_bravais_lattice(number)
        lattice_system, centering, label = BRAVAIS_METADATA[symbol]
        lattice_id = f"LATTICE_TYPE:{symbol}"
        sg_attrs = updated_nodes[sg_id]
        legacy_value = sg_attrs.get("lattice_id", "")
        sg_attrs["legacy_pyxtal_lattice_id"] = legacy_value
        sg_attrs["lattice_id"] = symbol
        sg_attrs["bravais_lattice"] = symbol
        updated_nodes.setdefault(
            lattice_id,
            {
                "type": "lattice_type",
                "label": f"{symbol}: {label}",
                "bravais_symbol": symbol,
                "lattice_system": lattice_system,
                "centering": centering,
                "standard": "14 Bravais lattices",
            },
        )
        updated_triples.add((sg_id, "has_lattice_type", lattice_id))
        mapped_space_groups.add(sg_id)
        mapping_rows.append(
            {
                "space_group_number": number,
                "space_group_id": sg_id,
                "space_group_symbol": sg_attrs.get("symbol", ""),
                "crystal_system": sg_attrs.get("crystal_system", ""),
                "bravais_lattice": symbol,
                "lattice_node_id": lattice_id,
                "legacy_pyxtal_lattice_id": legacy_value,
            }
        )

    lattice_nodes = {
        node_id for node_id, attrs in updated_nodes.items() if attrs.get("type") == "lattice_type"
    }
    lattice_triples = {
        triple for triple in updated_triples if triple[1] == "has_lattice_type"
    }
    if len(lattice_nodes) != 14 or len(lattice_triples) != 230:
        raise RuntimeError(
            f"Expected 14 lattice nodes and 230 mappings, got "
            f"{len(lattice_nodes)} and {len(lattice_triples)}"
        )
    if {head for head, _, _ in lattice_triples} != mapped_space_groups:
        raise RuntimeError("Every space group must have exactly one Bravais-lattice mapping")

    audit = {
        "legacy_lattice_node_ids": sorted(legacy_lattice_nodes),
        "legacy_lattice_nodes": len(legacy_lattice_nodes),
        "legacy_lattice_edges": len(legacy_lattice_triples),
        "legacy_unmapped_space_groups": sorted(
            {f"SG:{number:03d}" for number in range(1, 231)}
            - {head for head, _, _ in legacy_lattice_triples}
        ),
        "standard_lattice_node_ids": sorted(lattice_nodes),
        "standard_lattice_nodes": len(lattice_nodes),
        "standard_lattice_edges": len(lattice_triples),
        "mapping_counts": dict(sorted(Counter(SPACE_GROUP_TO_BRAVAIS.values()).items())),
        "non_lattice_triples_unchanged": updated_triples - lattice_triples
        == set(triples) - legacy_lattice_triples,
        "mapping_rows": mapping_rows,
    }
    return updated_nodes, sorted(updated_triples), audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_v2_audit(out_dir: Path, audit: dict[str, Any]) -> None:
    mapping_path = out_dir / "bravais_lattice_mapping.tsv"
    fieldnames = [
        "space_group_number",
        "space_group_id",
        "space_group_symbol",
        "crystal_system",
        "bravais_lattice",
        "lattice_node_id",
        "legacy_pyxtal_lattice_id",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit.pop("mapping_rows"))

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0.0",
        "kg_version": "KG",
        "parent": "base KG graph",
        "change": (
            "Replace PyXtal lattice_id categories with the standard 14 Bravais "
            "lattices and map every International space-group type 1..230."
        ),
        "bravais_mapping_basis": "International space-group type number; setting-independent",
        "graph_summary": summary,
        "audit": audit,
        "files": {},
    }
    for filename in (
        "nodes.tsv",
        "triples.tsv",
        "relations.tsv",
        "graph.graphml",
        "summary.json",
        "bravais_lattice_mapping.tsv",
    ):
        path = out_dir / filename
        manifest["files"][filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    (out_dir / "kg_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build KG with a setting-independent standard 14-Bravais-lattice mapping."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/kg_standard_bravais14")
    )
    parser.add_argument(
        "--base-graph-dir",
        type=Path,
        default=Path("data/base_graph_source"),
        help="Archived base graph to transform without regenerating PyXtal data.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    nodes, triples = read_existing_graph(args.base_graph_dir)
    nodes, triples, audit = replace_lattice_subgraph(nodes, triples)
    audit["source_graph"] = {
        "path": str(args.base_graph_dir),
        "nodes_sha256": sha256_file(args.base_graph_dir / "nodes.tsv"),
        "triples_sha256": sha256_file(args.base_graph_dir / "triples.tsv"),
    }
    write_outputs(nodes, triples, args.out_dir)
    write_v2_audit(args.out_dir, audit)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "num_nodes": len(nodes),
                "num_triples": len(triples),
                "lattice_nodes": 14,
                "lattice_edges": 230,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
