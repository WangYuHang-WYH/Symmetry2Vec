from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
from pyxtal.symmetry import Group


NodeAttrs = Dict[str, Any]
Triple = Tuple[str, str, str]


SITE_SYMM_RE = re.compile(r"^\s*(?P<label>\S+)\s+site symm:\s+(?P<site>.+?)\s*$")


def _node_id(kind: str, value: str) -> str:
    safe_value = str(value).strip().replace("\t", " ").replace("\n", " ")
    return f"{kind}:{safe_value}"


def _space_group_id(number: int) -> str:
    return f"SG:{number:03d}"


def _wyckoff_id(group_number: int, label: str) -> str:
    return f"WP:{group_number:03d}:{label}"


def _operation_id(xyz: str) -> str:
    return _node_id("OP", xyz)


def _jsonable(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _parse_site_symmetry_labels(group: Group) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for line in str(group).splitlines():
        match = SITE_SYMM_RE.match(line)
        if match:
            labels[match.group("label")] = match.group("site").strip()
    return labels


def _add_node(nodes: Dict[str, NodeAttrs], node_id: str, **attrs: Any) -> None:
    existing = nodes.setdefault(node_id, {})
    existing.update({k: v for k, v in attrs.items() if v is not None})


def _add_triple(triples: set[Triple], head: str, relation: str, tail: str) -> None:
    if head and relation and tail and head != tail:
        triples.add((head, relation, tail))


def _add_typed_link(
    nodes: Dict[str, NodeAttrs],
    triples: set[Triple],
    source: str,
    relation: str,
    kind: str,
    value: str,
    label: str | None = None,
) -> str:
    target = _node_id(kind, value)
    _add_node(nodes, target, type=kind.lower(), label=label or value)
    _add_triple(triples, source, relation, target)
    return target


def _safe_group(number: int) -> Group | None:
    try:
        return Group(number)
    except Exception as exc:
        print(f"Skipping space group {number}: {exc}")
        return None


def _operation_xyz(op: Any) -> str:
    if hasattr(op, "as_xyz_str"):
        return op.as_xyz_str()
    return str(op)


def build_knowledge_graph(
    include_operations: bool = True,
    include_wyckoff_operations: bool = True,
    include_subgroups: bool = True,
) -> tuple[Dict[str, NodeAttrs], List[Triple]]:
    nodes: Dict[str, NodeAttrs] = {}
    triples: set[Triple] = set()

    for number in range(1, 231):
        group = _safe_group(number)
        if group is None:
            continue

        sg_id = _space_group_id(number)
        _add_node(
            nodes,
            sg_id,
            type="space_group",
            label=f"{number} {group.symbol}",
            number=number,
            symbol=group.symbol,
            hall_number=getattr(group, "hall_number", ""),
            point_group=getattr(group, "point_group", ""),
            crystal_system=getattr(group, "lattice_type", ""),
            lattice_id=getattr(group, "lattice_id", ""),
            dim=getattr(group, "dim", ""),
        )

        crystal_system = str(getattr(group, "lattice_type", "") or "")
        if crystal_system:
            _add_typed_link(
                nodes,
                triples,
                sg_id,
                "has_crystal_system",
                "CRYSTAL_SYSTEM",
                crystal_system,
                crystal_system,
            )

        point_group = str(getattr(group, "point_group", "") or "")
        if point_group:
            _add_typed_link(
                nodes,
                triples,
                sg_id,
                "has_point_group",
                "POINT_GROUP",
                point_group,
                point_group,
            )

        lattice_id = str(getattr(group, "lattice_id", "") or "")
        if lattice_id:
            _add_typed_link(
                nodes,
                triples,
                sg_id,
                "has_lattice_type",
                "LATTICE_TYPE",
                lattice_id,
                f"lattice_type_{lattice_id}",
            )

        site_symmetry_by_label = _parse_site_symmetry_labels(group)
        for wp_index, wp in enumerate(group.Wyckoff_positions):
            label = wp.get_label()
            wp_id = _wyckoff_id(number, label)
            site_symmetry = site_symmetry_by_label.get(label)
            if site_symmetry is None:
                site_obj = wp.get_site_symmetry_object()
                site_symmetry = getattr(site_obj, "name", "") or "unknown"

            _add_node(
                nodes,
                wp_id,
                type="wyckoff_position",
                label=f"{number}:{label}",
                space_group_number=number,
                space_group_symbol=group.symbol,
                wyckoff_label=label,
                letter=getattr(wp, "letter", ""),
                multiplicity=getattr(wp, "multiplicity", ""),
                index=wp_index,
                site_symmetry=site_symmetry,
                hm_symbol=wp.get_hm_symbol(),
                dof=wp.get_dof(),
            )
            _add_triple(triples, sg_id, "has_wyckoff_position", wp_id)
            _add_triple(triples, wp_id, "wyckoff_in_space_group", sg_id)

            site_id = _node_id("SITE_SYMM", site_symmetry)
            _add_node(
                nodes,
                site_id,
                type="site_symmetry",
                label=site_symmetry,
            )
            _add_triple(triples, wp_id, "has_site_symmetry", site_id)
            _add_triple(triples, site_id, "site_symmetry_of_wyckoff", wp_id)

            mult_id = _node_id("MULTIPLICITY", str(wp.multiplicity))
            _add_node(
                nodes,
                mult_id,
                type="multiplicity",
                label=str(wp.multiplicity),
                value=wp.multiplicity,
            )
            _add_triple(triples, wp_id, "has_multiplicity", mult_id)

            if include_wyckoff_operations:
                for op in wp.ops:
                    op_xyz = _operation_xyz(op)
                    op_id = _operation_id(op_xyz)
                    _add_node(nodes, op_id, type="symmetry_operation", label=op_xyz)
                    _add_triple(triples, wp_id, "wyckoff_has_operation", op_id)
                    _add_triple(triples, op_id, "operation_in_wyckoff", wp_id)

        if include_operations:
            operations = group.Wyckoff_positions[0].ops if group.Wyckoff_positions else []
            for op in operations:
                op_xyz = _operation_xyz(op)
                op_id = _operation_id(op_xyz)
                _add_node(nodes, op_id, type="symmetry_operation", label=op_xyz)
                _add_triple(triples, sg_id, "space_group_has_operation", op_id)
                _add_triple(triples, op_id, "operation_in_space_group", sg_id)

        if include_subgroups:
            try:
                subgroup_numbers = sorted(set(group.get_max_subgroup_numbers()))
            except Exception:
                subgroup_numbers = []
            for subgroup_number in subgroup_numbers:
                if 1 <= int(subgroup_number) <= 230:
                    subgroup_id = _space_group_id(int(subgroup_number))
                    _add_triple(triples, sg_id, "has_max_subgroup", subgroup_id)
                    _add_triple(triples, subgroup_id, "has_min_supergroup", sg_id)

    return nodes, sorted(triples)


def write_outputs(nodes: Dict[str, NodeAttrs], triples: List[Triple], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_keys = sorted({key for attrs in nodes.values() for key in attrs})
    with (out_dir / "nodes.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", *attr_keys],
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for node_id, attrs in sorted(nodes.items()):
            row = {"id": node_id}
            row.update({key: _jsonable(value) for key, value in attrs.items()})
            writer.writerow(row)

    with (out_dir / "triples.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["head", "relation", "tail"])
        writer.writerows(triples)

    relation_counts = Counter(relation for _, relation, _ in triples)
    with (out_dir / "relations.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relation", "count"])
        for relation, count in sorted(relation_counts.items()):
            writer.writerow([relation, count])

    graph = nx.MultiDiGraph()
    for node_id, attrs in nodes.items():
        graph.add_node(node_id, **{key: _jsonable(value) for key, value in attrs.items()})
    for head, relation, tail in triples:
        graph.add_edge(head, tail, relation=relation)
    nx.write_graphml(graph, out_dir / "graph.graphml")

    summary = {
        "num_nodes": len(nodes),
        "num_triples": len(triples),
        "node_types": Counter(attrs.get("type", "unknown") for attrs in nodes.values()),
        "relation_counts": relation_counts,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a crystal symmetry knowledge graph.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/kg"))
    parser.add_argument("--no-operations", action="store_true")
    parser.add_argument("--no-wyckoff-operations", action="store_true")
    parser.add_argument("--no-subgroups", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    nodes, triples = build_knowledge_graph(
        include_operations=not args.no_operations,
        include_wyckoff_operations=not args.no_wyckoff_operations,
        include_subgroups=not args.no_subgroups,
    )
    write_outputs(nodes, triples, args.out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "num_nodes": len(nodes),
                "num_triples": len(triples),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
