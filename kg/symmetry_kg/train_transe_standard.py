from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OUT_DIR = Path("artifacts/kge_transe_standard_d200_v1")


@dataclass
class TrainConfig:
    dim: int = 200
    epochs: int = 1000
    batch_size: int = 4096
    negative_ratio: int = 4
    lr: float = 0.001
    margin: float = 1.0
    distance_norm: int = 1
    seed: int = 13
    device: str = "auto"


class StandardTransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)
        self.normalize_entities()

    def distance(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        norm: int,
    ) -> torch.Tensor:
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=norm,
            dim=1,
        )

    @torch.no_grad()
    def normalize_entities(self) -> None:
        self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=1))


def validate_config(config: TrainConfig) -> None:
    if config.dim < 1:
        raise ValueError("--dim must be >= 1")
    if config.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if config.negative_ratio < 1:
        raise ValueError("--negative-ratio must be >= 1")
    if config.lr <= 0:
        raise ValueError("--lr must be > 0")
    if config.margin <= 0:
        raise ValueError("--margin must be > 0")
    if config.distance_norm not in {1, 2}:
        raise ValueError("--distance-norm must be 1 or 2")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_nodes(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    return {row["id"]: row for row in read_tsv(path)}


def load_triples(
    path: Path,
) -> tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    rows = read_tsv(path)
    triples = [(row["head"], row["relation"], row["tail"]) for row in rows]
    entities = sorted({head for head, _, _ in triples} | {tail for _, _, tail in triples})
    relations = sorted({relation for _, relation, _ in triples})
    return triples, entities, relations


def encode_triples(
    triples: Sequence[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> torch.Tensor:
    return torch.tensor(
        [
            (entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
            for head, relation, tail in triples
        ],
        dtype=torch.long,
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def triple_keys(
    triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
) -> torch.Tensor:
    return (
        triples[:, 0] * (num_relations * num_entities)
        + triples[:, 1] * num_entities
        + triples[:, 2]
    )


def known_triple_mask(candidate_keys: torch.Tensor, sorted_positive_keys: torch.Tensor) -> torch.Tensor:
    positions = torch.searchsorted(sorted_positive_keys, candidate_keys)
    in_bounds = positions < sorted_positive_keys.numel()
    safe_positions = positions.clamp(max=max(0, sorted_positive_keys.numel() - 1))
    return in_bounds & (sorted_positive_keys[safe_positions] == candidate_keys)


def sample_filtered_negatives(
    batch: torch.Tensor,
    num_entities: int,
    num_relations: int,
    negative_ratio: int,
    sorted_positive_keys: torch.Tensor,
    max_resample_rounds: int = 64,
) -> torch.Tensor:
    negatives = batch.repeat_interleave(negative_ratio, dim=0).clone()
    replace_heads = torch.rand(negatives.shape[0], device=batch.device) < 0.5
    unresolved = torch.ones(negatives.shape[0], dtype=torch.bool, device=batch.device)

    for _ in range(max_resample_rounds):
        if not unresolved.any().item():
            break
        unresolved_indices = unresolved.nonzero(as_tuple=False).view(-1)
        corruptions = torch.randint(
            0,
            num_entities,
            (unresolved_indices.numel(),),
            device=batch.device,
        )
        replace_head_now = replace_heads[unresolved_indices]
        current = negatives[unresolved_indices]
        current[replace_head_now, 0] = corruptions[replace_head_now]
        current[~replace_head_now, 2] = corruptions[~replace_head_now]
        negatives[unresolved_indices] = current

        candidate_keys = triple_keys(negatives, num_entities, num_relations)
        unresolved = known_triple_mask(candidate_keys, sorted_positive_keys)
    if unresolved.any().item():
        raise RuntimeError(
            "Could not sample filtered negative triples after "
            f"{max_resample_rounds} rounds"
        )
    return negatives


def train_model(
    triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    config: TrainConfig,
) -> tuple[StandardTransE, List[Dict[str, float]]]:
    validate_config(config)
    if triples.ndim != 2 or triples.shape[1] != 3 or triples.shape[0] == 0:
        raise ValueError("triples must be a non-empty [N, 3] tensor")
    if num_entities < 2:
        raise ValueError("At least two entities are required")
    if num_relations < 1:
        raise ValueError("At least one relation is required")

    set_seed(config.seed)
    device = choose_device(config.device)
    model = StandardTransE(num_entities, num_relations, config.dim).to(device)
    triples = triples.to(device)
    sorted_positive_keys = torch.sort(
        triple_keys(triples, num_entities, num_relations)
    ).values
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: List[Dict[str, float]] = []
    num_triples = triples.shape[0]

    for epoch in range(1, config.epochs + 1):
        permutation = torch.randperm(num_triples, device=device)
        total_loss = 0.0
        total_pairs = 0
        total_positive_distance = 0.0
        total_negative_distance = 0.0
        total_active = 0.0

        model.train()
        for start in range(0, num_triples, config.batch_size):
            batch = triples[permutation[start : start + config.batch_size]]
            negatives = sample_filtered_negatives(
                batch,
                num_entities,
                num_relations,
                config.negative_ratio,
                sorted_positive_keys,
            )
            positive_distance = model.distance(
                batch[:, 0], batch[:, 1], batch[:, 2], config.distance_norm
            )
            negative_distance = model.distance(
                negatives[:, 0],
                negatives[:, 1],
                negatives[:, 2],
                config.distance_norm,
            )
            repeated_positive_distance = positive_distance.repeat_interleave(
                config.negative_ratio
            )
            margin_terms = config.margin + repeated_positive_distance - negative_distance
            loss = F.relu(margin_terms).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_entities()

            pair_count = negatives.shape[0]
            total_pairs += pair_count
            total_loss += float(loss.detach().cpu()) * pair_count
            total_positive_distance += float(
                repeated_positive_distance.detach().sum().cpu()
            )
            total_negative_distance += float(negative_distance.detach().sum().cpu())
            total_active += float((margin_terms.detach() > 0).sum().cpu())

        record = {
            "epoch": float(epoch),
            "loss": total_loss / total_pairs,
            "mean_positive_distance": total_positive_distance / total_pairs,
            "mean_negative_distance": total_negative_distance / total_pairs,
            "active_margin_fraction": total_active / total_pairs,
        }
        history.append(record)
        log_interval = max(1, config.epochs // 20)
        if epoch == 1 or epoch == config.epochs or epoch % log_interval == 0:
            print(
                "epoch={epoch:.0f} loss={loss:.6f} pos_d={mean_positive_distance:.4f} "
                "neg_d={mean_negative_distance:.4f} active={active_margin_fraction:.4f}".format(
                    **record
                ),
                flush=True,
            )

    return model.cpu(), history


def write_embedding_table(path: Path, ids: Sequence[str], vectors: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", *[f"x{i}" for i in range(vectors.shape[1])]])
        for item_id, vector in zip(ids, vectors):
            writer.writerow([item_id, *[f"{value:.8g}" for value in vector]])


def write_nearest_neighbors(
    path: Path,
    entity_ids: Sequence[str],
    entity_vectors: np.ndarray,
    node_metadata: Dict[str, Dict[str, str]],
    top_k: int,
    include_types: set[str] | None,
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    neighbor_count = min(top_k, max(0, len(entity_ids) - 1))
    safe_vectors = entity_vectors / np.maximum(
        np.linalg.norm(entity_vectors, axis=1, keepdims=True), 1e-12
    )
    similarities = safe_vectors @ safe_vectors.T
    np.fill_diagonal(similarities, -np.inf)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "id",
                "type",
                "label",
                "neighbor_rank",
                "neighbor_id",
                "neighbor_type",
                "neighbor_label",
                "cosine_similarity",
            ]
        )
        for index, entity_id in enumerate(entity_ids):
            metadata = node_metadata.get(entity_id, {})
            if include_types is not None and metadata.get("type", "") not in include_types:
                continue
            if neighbor_count == 0:
                continue
            neighbors = np.argpartition(
                -similarities[index], neighbor_count - 1
            )[:neighbor_count]
            neighbors = neighbors[np.argsort(-similarities[index][neighbors])]
            for rank, neighbor_index in enumerate(neighbors, start=1):
                neighbor_id = entity_ids[int(neighbor_index)]
                neighbor_metadata = node_metadata.get(neighbor_id, {})
                writer.writerow(
                    [
                        entity_id,
                        metadata.get("type", ""),
                        metadata.get("label", ""),
                        rank,
                        neighbor_id,
                        neighbor_metadata.get("type", ""),
                        neighbor_metadata.get("label", ""),
                        f"{similarities[index, neighbor_index]:.8g}",
                    ]
                )


def save_outputs(
    out_dir: Path,
    model: StandardTransE,
    entity_ids: Sequence[str],
    relation_ids: Sequence[str],
    node_metadata: Dict[str, Dict[str, str]],
    history: List[Dict[str, float]],
    config: TrainConfig,
    top_k: int,
    neighbor_types: set[str] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entity_vectors = model.entity.weight.detach().numpy()
    relation_vectors = model.relation.weight.detach().numpy()
    write_embedding_table(out_dir / "entity_embeddings.tsv", entity_ids, entity_vectors)
    write_embedding_table(out_dir / "relation_embeddings.tsv", relation_ids, relation_vectors)
    if top_k > 0:
        write_nearest_neighbors(
            out_dir / "nearest_neighbors.tsv",
            entity_ids,
            entity_vectors,
            node_metadata,
            top_k,
            neighbor_types,
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "entity_ids": list(entity_ids),
            "relation_ids": list(relation_ids),
            "config": asdict(config),
            "objective": "margin_ranking",
            "filtered_negative_sampling": True,
        },
        out_dir / "model.pt",
    )
    summary = {
        "config": asdict(config),
        "objective": "relu(margin + positive_distance - negative_distance)",
        "optimizer": "Adam",
        "filtered_negative_sampling": True,
        "entity_constraint": "L2 unit norm after every optimizer step",
        "num_entities": len(entity_ids),
        "num_relations": len(relation_ids),
        "history": history,
        "final_metrics": history[-1],
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train standard margin-ranking TransE on KG triples."
    )
    parser.add_argument("--triples", type=Path, default=Path("data/kg/triples.tsv"))
    parser.add_argument("--nodes", type=Path, default=Path("data/kg/nodes.tsv"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--distance-norm", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nearest neighbors per entity; use 0 to skip neighbor export.",
    )
    parser.add_argument(
        "--neighbor-types",
        default="space_group,wyckoff_position,site_symmetry,crystal_system,point_group",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    config = TrainConfig(
        dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_ratio=args.negative_ratio,
        lr=args.lr,
        margin=args.margin,
        distance_norm=args.distance_norm,
        seed=args.seed,
        device=args.device,
    )
    validate_config(config)
    if not args.triples.exists():
        raise FileNotFoundError(args.triples)
    if args.nodes is not None and not args.nodes.exists():
        raise FileNotFoundError(args.nodes)

    triples, entity_ids, relation_ids = load_triples(args.triples)
    entity_to_id = {entity_id: index for index, entity_id in enumerate(entity_ids)}
    relation_to_id = {relation_id: index for index, relation_id in enumerate(relation_ids)}
    encoded = encode_triples(triples, entity_to_id, relation_to_id)
    metadata = load_nodes(args.nodes)
    neighbor_types = None
    if args.neighbor_types.strip().lower() != "all":
        neighbor_types = {
            item.strip() for item in args.neighbor_types.split(",") if item.strip()
        }

    print(
        json.dumps(
            {
                "config": asdict(config),
                "num_entities": len(entity_ids),
                "num_relations": len(relation_ids),
                "num_triples": len(triples),
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    model, history = train_model(
        encoded,
        len(entity_ids),
        len(relation_ids),
        config,
    )
    save_outputs(
        args.out_dir,
        model,
        entity_ids,
        relation_ids,
        metadata,
        history,
        config,
        args.top_k,
        neighbor_types,
    )
    print(f"Saved standard TransE artifacts to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
