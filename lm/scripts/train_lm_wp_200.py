#!/usr/bin/env python3
"""Pretrain BERT with target-orbit WP, multiplicity, and site-symmetry masking."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup


WP_SURFACE_RE = re.compile(r"^WP\|SG_(\d+)\|(\d+)([A-Za-z]+)$")
MULTIPLICITY_OFFSET = -5
SITE_SYMMETRY_OFFSET = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def pack_sentences(words: list[str], maximum_words: int) -> list[list[str]]:
    sentences: list[list[str]] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if word == ".":
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    chunks: list[list[str]] = []
    chunk: list[str] = []
    for sentence in sentences:
        if len(sentence) > maximum_words:
            raise ValueError(
                f"A sentence has {len(sentence)} words, above the limit {maximum_words}"
            )
        if chunk and len(chunk) + len(sentence) > maximum_words:
            chunks.append(chunk)
            chunk = []
        chunk.extend(sentence)
    if chunk:
        chunks.append(chunk)
    if sum(map(len, chunks)) != len(words):
        raise RuntimeError("Sentence packing lost words")
    return chunks


def load_wp_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.sort(key=lambda row: int(row["row_index"]))
    if len(rows) != 1731:
        raise RuntimeError(f"Expected 1731 WP rows, found {len(rows)}")
    if [int(row["row_index"]) for row in rows] != list(range(1731)):
        raise RuntimeError("WP row indices are not contiguous from 0 to 1730")
    return rows


def token_id_to_wp_class(tokenizer_size: int, wp_rows: list[dict[str, str]]) -> np.ndarray:
    lookup = np.full(tokenizer_size, -1, dtype=np.int64)
    for row in wp_rows:
        token_id = int(row["token_id"])
        wp_class = int(row["row_index"])
        if token_id < 0 or token_id >= tokenizer_size:
            raise RuntimeError(f"WP token ID outside tokenizer: {token_id}")
        if lookup[token_id] >= 0:
            raise RuntimeError(f"Duplicate WP tokenizer ID: {token_id}")
        lookup[token_id] = wp_class
    return lookup


def validate_lm_orbits(words: list[str]) -> int:
    wp_count = 0
    for position, token in enumerate(words):
        if not token.startswith("WP|SG_"):
            continue
        wp_count += 1
        match = WP_SURFACE_RE.fullmatch(token)
        if match is None:
            raise RuntimeError(f"Malformed WP surface token: {token}")
        if position < -MULTIPLICITY_OFFSET or position + SITE_SYMMETRY_OFFSET >= len(words):
            raise RuntimeError(f"LM orbit context crosses a chunk boundary: {token}")
        if words[position - 3 : position] != ["at", "Wyckoff", "position"]:
            raise RuntimeError(f"Unexpected words before WP token: {token}")
        if words[position + 1 : position + 4] != ["with", "site", "symmetry"]:
            raise RuntimeError(f"Unexpected words after WP token: {token}")
        if words[position + MULTIPLICITY_OFFSET] != match.group(2):
            raise RuntimeError(f"Multiplicity does not match WP token: {token}")
        if not words[position + SITE_SYMMETRY_OFFSET].startswith("SS_"):
            raise RuntimeError(f"Missing site-symmetry token for: {token}")
    return wp_count


def cache_contract(
    config: dict[str, Any], tokenizer: BertTokenizerFast, split: str
) -> dict[str, Any]:
    corpus_path = Path(config["corpus"]["output_dir"]) / f"{split}.jsonl.gz"
    tokenizer_audit = Path(config["tokenizer"]["output_dir"]) / "tokenizer_audit.json"
    return {
        "schema_version": "2.0.0",
        "split": split,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "tokenizer_size": len(tokenizer),
        "tokenizer_audit_sha256": sha256_file(tokenizer_audit),
        "chunk_words": int(config["training"]["chunk_words"]),
        "masking_strategy": "target_wp_multiplicity_site_symmetry_group_mask",
        "multiplicity_offset_from_wp": MULTIPLICITY_OFFSET,
        "site_symmetry_offset_from_wp": SITE_SYMMETRY_OFFSET,
    }


def prepare_encoded_split(
    config: dict[str, Any],
    tokenizer: BertTokenizerFast,
    wp_lookup: np.ndarray,
    split: str,
) -> dict[str, Any]:
    cache_dir = Path(config["training"]["encoded_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    flat_path = cache_dir / f"{split}_input_ids.npy"
    offsets_path = cache_dir / f"{split}_offsets.npy"
    metadata_path = cache_dir / f"{split}_metadata.json"
    expected = cache_contract(config, tokenizer, split)
    if flat_path.is_file() and offsets_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract") == expected:
            print(f"Using encoded cache: {flat_path}", flush=True)
            return metadata

    corpus_path = Path(config["corpus"]["output_dir"]) / f"{split}.jsonl.gz"
    maximum_words = int(config["training"]["chunk_words"])
    chunks: list[np.ndarray] = []
    offsets = [0]
    structures = 0
    chunks_without_wp = 0
    wp_targets = 0
    maximum_length = 0
    for record in records(corpus_path):
        structures += 1
        for words in pack_sentences(record["tokens"], maximum_words):
            validated_wp_count = validate_lm_orbits(words)
            identifiers = tokenizer.convert_tokens_to_ids(words)
            if any(identifier == tokenizer.unk_token_id for identifier in identifiers):
                raise RuntimeError("UNK found after tokenizer audit")
            array = np.asarray(
                [tokenizer.cls_token_id, *identifiers, tokenizer.sep_token_id],
                dtype=np.int32,
            )
            classes = wp_lookup[array]
            count = int(np.count_nonzero(classes >= 0))
            if count != validated_wp_count:
                raise RuntimeError("Tokenizer WP count differs from LM audit")
            if count == 0:
                chunks_without_wp += 1
                continue
            chunks.append(array)
            offsets.append(offsets[-1] + len(array))
            wp_targets += count
            maximum_length = max(maximum_length, len(array))
    if not chunks:
        raise RuntimeError(f"No WP-bearing chunks encoded for {split}")
    flat = np.concatenate(chunks)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    temporary_flat = flat_path.with_suffix(".npy.tmp")
    temporary_offsets = offsets_path.with_suffix(".npy.tmp")
    with temporary_flat.open("wb") as handle:
        np.save(handle, flat, allow_pickle=False)
    with temporary_offsets.open("wb") as handle:
        np.save(handle, offsets_array, allow_pickle=False)
    temporary_flat.replace(flat_path)
    temporary_offsets.replace(offsets_path)
    metadata = {
        "contract": expected,
        "structures": structures,
        "wp_bearing_chunks": len(chunks),
        "chunks_without_wp_dropped": chunks_without_wp,
        "encoded_tokens": int(len(flat)),
        "wp_token_occurrences": wp_targets,
        "lm_validated_wp_occurrences": wp_targets,
        "maximum_encoded_length": maximum_length,
        "flat_path": str(flat_path),
        "offsets_path": str(offsets_path),
        "flat_sha256": sha256_file(flat_path),
        "offsets_sha256": sha256_file(offsets_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


class EncodedWpMlmDataset(Dataset):
    def __init__(
        self,
        metadata: dict[str, Any],
        wp_lookup: np.ndarray,
        mask_token_id: int,
        mask_probability: float,
        seed: int,
    ) -> None:
        self.flat = np.load(metadata["flat_path"], mmap_mode="r")
        self.offsets = np.load(metadata["offsets_path"], mmap_mode="r")
        self.wp_lookup = wp_lookup
        self.mask_token_id = int(mask_token_id)
        self.mask_probability = float(mask_probability)
        self.seed = int(seed)
        self.epoch = 0
        if not 0.0 < self.mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        input_ids = np.asarray(self.flat[start:stop], dtype=np.int64).copy()
        classes = self.wp_lookup[input_ids]
        positions = np.flatnonzero(classes >= 0)
        if len(positions) == 0:
            raise RuntimeError("Encoded sample unexpectedly contains no WP")
        number_masked = max(1, int(round(len(positions) * self.mask_probability)))
        number_masked = min(number_masked, len(positions))
        local_seed = (self.seed + self.epoch * 1_000_003 + index * 97_409) % (2**32)
        rng = np.random.RandomState(local_seed)
        selected = np.sort(rng.choice(positions, size=number_masked, replace=False))
        targets = classes[selected].astype(np.int64, copy=True)
        input_ids[selected] = self.mask_token_id
        input_ids[selected + MULTIPLICITY_OFFSET] = self.mask_token_id
        input_ids[selected + SITE_SYMMETRY_OFFSET] = self.mask_token_id
        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(selected.astype(np.int64, copy=False)),
            torch.from_numpy(targets),
        )


@dataclass
class WpMlmBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    masked_batch_indices: torch.Tensor
    masked_positions: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device) -> "WpMlmBatch":
        return WpMlmBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            masked_batch_indices=self.masked_batch_indices.to(device, non_blocking=True),
            masked_positions=self.masked_positions.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
        )


class WpMlmCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(
        self, samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> WpMlmBatch:
        maximum_length = max(len(sample[0]) for sample in samples)
        input_ids = torch.full(
            (len(samples), maximum_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        batch_indices: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch_index, (identifiers, selected, labels) in enumerate(samples):
            length = len(identifiers)
            input_ids[batch_index, :length] = identifiers
            attention_mask[batch_index, :length] = 1
            batch_indices.append(torch.full_like(selected, batch_index))
            positions.append(selected)
            targets.append(labels)
        return WpMlmBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            masked_batch_indices=torch.cat(batch_indices),
            masked_positions=torch.cat(positions),
            targets=torch.cat(targets),
        )


class BertWpMlm(nn.Module):
    def __init__(
        self,
        base_model: str,
        tokenizer: BertTokenizerFast,
        wp_lookup: np.ndarray,
        active_wp_classes: np.ndarray,
        wp_dimension: int,
        initialization_path: Path,
        gradient_checkpointing: bool,
        attention_implementation: str,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(
            base_model,
            local_files_only=True,
            add_pooling_layer=False,
            attn_implementation=attention_implementation,
        )
        base_vocabulary_size = self.bert.get_input_embeddings().weight.shape[0]
        base_embeddings = self.bert.get_input_embeddings().weight.detach().clone()
        self.bert.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        self._initialize_added_tokens(
            base_embeddings, base_vocabulary_size, initialization_path
        )
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
            self.bert.config.use_cache = False
        hidden_size = int(self.bert.config.hidden_size)
        self.wp_embeddings = nn.Embedding(1731, wp_dimension)
        nn.init.normal_(self.wp_embeddings.weight, mean=0.0, std=0.02)
        self.wp_input_projection = nn.Linear(wp_dimension, hidden_size, bias=False)
        self.prediction_transform = nn.Sequential(
            nn.Linear(hidden_size, wp_dimension),
            nn.GELU(),
            nn.LayerNorm(wp_dimension),
        )
        self.wp_decoder_bias = nn.Parameter(torch.zeros(1731))
        self.register_buffer(
            "wp_class_by_token_id", torch.as_tensor(wp_lookup, dtype=torch.long)
        )
        active_mask = torch.zeros(1731, dtype=torch.bool)
        active_mask[torch.as_tensor(active_wp_classes, dtype=torch.long)] = True
        self.register_buffer("active_wp_mask", active_mask)

    def _initialize_added_tokens(
        self,
        base_embeddings: torch.Tensor,
        base_vocabulary_size: int,
        initialization_path: Path,
    ) -> None:
        target = self.bert.get_input_embeddings().weight
        initialized = 0
        with initialization_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                token_id = int(row["token_id"])
                piece_ids = [
                    int(value) for value in row["base_piece_ids"].split(",") if value
                ]
                valid = [value for value in piece_ids if value < base_vocabulary_size]
                if token_id < base_vocabulary_size or not valid:
                    continue
                with torch.no_grad():
                    target[token_id].copy_(base_embeddings[valid].mean(dim=0))
                initialized += 1
        print(f"Initialized {initialized} added token rows from base WordPieces", flush=True)

    def forward(self, batch: WpMlmBatch) -> torch.Tensor:
        word_embeddings = self.bert.get_input_embeddings()(batch.input_ids)
        wp_classes = self.wp_class_by_token_id[batch.input_ids]
        visible = wp_classes >= 0
        if visible.any():
            replacement = self.wp_input_projection(self.wp_embeddings(wp_classes[visible]))
            word_embeddings = word_embeddings.clone()
            word_embeddings[visible] = replacement.to(word_embeddings.dtype)
        outputs = self.bert(
            inputs_embeds=word_embeddings,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state[
            batch.masked_batch_indices, batch.masked_positions
        ]
        query = self.prediction_transform(hidden)
        logits = F.linear(query, self.wp_embeddings.weight, self.wp_decoder_bias)
        logits = logits.masked_fill(~self.active_wp_mask.unsqueeze(0), -1.0e4)
        return logits


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def build_loader(
    dataset: EncodedWpMlmDataset,
    batch_size: int,
    collator: WpMlmCollator,
    workers: int,
    sampler: Optional[DistributedSampler],
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collator,
    )


@torch.no_grad()
def evaluate(
    model: BertWpMlm,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    targets = 0
    top1 = 0
    top5 = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            logits = model(batch)
            loss = F.cross_entropy(logits.float(), batch.targets, reduction="sum")
        count = len(batch.targets)
        predictions = logits.topk(k=5, dim=1).indices
        loss_sum += float(loss.cpu())
        targets += count
        top1 += int((predictions[:, 0] == batch.targets).sum().cpu())
        top5 += int((predictions == batch.targets.unsqueeze(1)).any(dim=1).sum().cpu())
    return {
        "loss": loss_sum / max(targets, 1),
        "perplexity": math.exp(min(loss_sum / max(targets, 1), 20.0)),
        "top1_accuracy": top1 / max(targets, 1),
        "top5_accuracy": top5 / max(targets, 1),
        "masked_wp_targets": targets,
    }


def optimizer_groups(
    model: BertWpMlm, training: dict[str, Any]
) -> list[dict[str, Any]]:
    bert_decay: list[nn.Parameter] = []
    bert_no_decay: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    wp_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "wp_embeddings.weight" or name == "wp_decoder_bias":
            wp_parameters.append(parameter)
        elif name.startswith("bert."):
            if name.endswith("bias") or "LayerNorm.weight" in name:
                bert_no_decay.append(parameter)
            else:
                bert_decay.append(parameter)
        else:
            head_parameters.append(parameter)
    return [
        {
            "params": bert_decay,
            "lr": float(training["bert_learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
        },
        {
            "params": bert_no_decay,
            "lr": float(training["bert_learning_rate"]),
            "weight_decay": 0.0,
        },
        {
            "params": head_parameters,
            "lr": float(training["wp_learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
        },
        {
            "params": wp_parameters,
            "lr": float(training["wp_learning_rate"]),
            "weight_decay": 0.0,
        },
    ]


def unwrap(model: nn.Module) -> BertWpMlm:
    return model.module if isinstance(model, DistributedDataParallel) else model


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def export_wp_table(
    model: BertWpMlm,
    wp_rows: list[dict[str, str]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = model.wp_embeddings.weight.detach().float().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1.0e-12)
    np.save(output_dir / "wp_vectors_1731x200.npy", vectors, allow_pickle=False)
    np.save(output_dir / "wp_vectors_1731x200_l2.npy", normalized, allow_pickle=False)
    table_path = output_dir / "entity_embeddings.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", *[f"x{index}" for index in range(vectors.shape[1])]])
        for row, vector in zip(wp_rows, vectors):
            writer.writerow(
                [row["entity_id"], *[format(float(value), ".9g") for value in vector]]
            )
    coverage_path = output_dir / "wp_training_coverage.tsv"
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "entity_id",
                "surface_token",
                "corpus_occurrences",
                "updated_from_initialization",
            ]
        )
        for row in wp_rows:
            occurrences = int(row["corpus_occurrences"])
            writer.writerow(
                [
                    row["row_index"],
                    row["entity_id"],
                    row["surface_token"],
                    occurrences,
                    int(occurrences > 0),
                ]
            )
    manifest = {
        **metadata,
        "shape": list(vectors.shape),
        "finite": bool(np.isfinite(vectors).all()),
        "raw_norm_min": float(norms.min()),
        "raw_norm_max": float(norms.max()),
        "entity_embeddings_sha256": sha256_file(table_path),
        "coverage_sha256": sha256_file(coverage_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def checkpoint_payload(
    model: BertWpMlm,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    bad_epochs: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "bad_epochs": bad_epochs,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rank, local_rank, world_size, device = distributed_context()
    seed = int(config["seed"])
    set_seed(seed + rank)
    tokenizer_dir = Path(config["tokenizer"]["output_dir"])
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    audit = json.loads((tokenizer_dir / "tokenizer_audit.json").read_text(encoding="utf-8"))
    if not audit["passed_one_word_one_token"]:
        raise RuntimeError("Tokenizer did not pass the one-word/one-token audit")
    wp_rows = load_wp_rows(tokenizer_dir / "wp_vocabulary_1731.tsv")
    wp_lookup = token_id_to_wp_class(len(tokenizer), wp_rows)

    if rank == 0:
        train_metadata = prepare_encoded_split(config, tokenizer, wp_lookup, "train")
        validation_metadata = prepare_encoded_split(
            config, tokenizer, wp_lookup, "validation"
        )
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        train_metadata = json.loads(
            (
                Path(config["training"]["encoded_cache_dir"])
                / "train_metadata.json"
            ).read_text(encoding="utf-8")
        )
        validation_metadata = json.loads(
            (
                Path(config["training"]["encoded_cache_dir"])
                / "validation_metadata.json"
            ).read_text(encoding="utf-8")
        )
    if args.prepare_cache_only:
        if world_size > 1:
            dist.destroy_process_group()
        return

    training = config["training"]
    train_dataset = EncodedWpMlmDataset(
        train_metadata,
        wp_lookup,
        tokenizer.mask_token_id,
        float(training["wp_mask_probability"]),
        seed,
    )
    validation_dataset = EncodedWpMlmDataset(
        validation_metadata,
        wp_lookup,
        tokenizer.mask_token_id,
        float(training["wp_mask_probability"]),
        seed + 7_919,
    )
    active_wp_classes = np.asarray(
        [int(row["row_index"]) for row in wp_rows if int(row["corpus_occurrences"]) > 0],
        dtype=np.int64,
    )
    if len(active_wp_classes) != int(
        json.loads(
            (Path(config["corpus"]["output_dir"]) / "manifest.json").read_text(
                encoding="utf-8"
            )
        )["counts"]["observed_mp_wp_rows"]
    ):
        raise RuntimeError("Active WP classes do not match corpus manifest")

    model = BertWpMlm(
        config["tokenizer"]["base_model"],
        tokenizer,
        wp_lookup,
        active_wp_classes,
        int(config["wp_dimension"]),
        tokenizer_dir / "added_token_initialization.tsv",
        bool(training["gradient_checkpointing"]),
        str(training["attention_implementation"]),
    ).to(device)
    collator = WpMlmCollator(tokenizer.pad_token_id)
    batch_size = (
        int(args.smoke_batch_size)
        if args.smoke_batch_size is not None
        else int(training["batch_size_per_gpu"])
    )
    if args.smoke_only:
        lengths = np.diff(train_dataset.offsets)
        longest_indices = np.argsort(lengths)[-batch_size:].tolist()
        smoke_dataset = Subset(train_dataset, longest_indices)
        loader = build_loader(
            smoke_dataset, batch_size, collator, workers=0, sampler=None, shuffle=False
        )
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        batch = next(iter(loader)).to(device)
        optimizer = torch.optim.AdamW(optimizer_groups(model, training))
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(batch)
            loss = F.cross_entropy(logits.float(), batch.targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        result = {
            "passed": True,
            "batch_size": batch_size,
            "sample_lengths": [int(lengths[index]) for index in longest_indices],
            "padded_length": int(batch.input_ids.shape[1]),
            "masked_wp_targets": int(len(batch.targets)),
            "loss": float(loss.detach().cpu()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "gpu_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
        output_dir = Path(training["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"memory_smoke_batch_{batch_size}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return

    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        if world_size > 1
        else None
    )
    train_loader = build_loader(
        train_dataset,
        batch_size,
        collator,
        int(training["num_workers"]),
        train_sampler,
        shuffle=True,
    )
    validation_loader = None
    if rank == 0:
        validation_loader = build_loader(
            validation_dataset,
            int(training["eval_batch_size"]),
            collator,
            int(training["num_workers"]),
            sampler=None,
            shuffle=False,
        )
    optimizer = torch.optim.AdamW(optimizer_groups(model, training))
    epochs = int(training["epochs"])
    accumulation = int(training["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(round(total_steps * float(training["warmup_fraction"])))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=str(training["precision"]) == "fp16"
    )
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "checkpoint_last.pt"
    start_epoch = 0
    best_validation_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if args.resume and last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_validation_loss = float(state["best_validation_loss"])
        bad_epochs = int(state["bad_epochs"])
        history = list(state["history"])
        print(f"Resumed after epoch {start_epoch - 1}", flush=True)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    fp16 = str(training["precision"]) == "fp16"
    patience = int(training["early_stopping_patience"])
    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = 0.0
        local_targets = 0
        local_top1 = 0
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                logits = model(batch)
                loss = F.cross_entropy(logits.float(), batch.targets)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step()
            count = len(batch.targets)
            local_loss_sum += float(loss.detach().cpu()) * count
            local_targets += count
            local_top1 += int((logits.argmax(dim=1) == batch.targets).sum().detach().cpu())

        totals = torch.tensor(
            [local_loss_sum, float(local_targets), float(local_top1)],
            dtype=torch.float64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_metrics = {
            "loss": float(totals[0].item() / max(totals[1].item(), 1.0)),
            "top1_accuracy": float(totals[2].item() / max(totals[1].item(), 1.0)),
            "masked_wp_targets": int(totals[1].item()),
        }

        validation_dataset.set_epoch(0)
        if rank == 0:
            validation_metrics = evaluate(
                unwrap(model), validation_loader, device, fp16
            )
            validation_loss_tensor = torch.tensor(
                validation_metrics["loss"], dtype=torch.float64, device=device
            )
        else:
            validation_metrics = {}
            validation_loss_tensor = torch.zeros((), dtype=torch.float64, device=device)
        if world_size > 1:
            dist.broadcast(validation_loss_tensor, src=0)
        validation_loss = float(validation_loss_tensor.item())
        improved = validation_loss < best_validation_loss - float(
            training["early_stopping_min_delta"]
        )
        if improved:
            best_validation_loss = validation_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if rank == 0:
            epoch_row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "best_validation_loss": best_validation_loss,
                "bad_epochs": bad_epochs,
            }
            history.append(epoch_row)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)
            bare_model = unwrap(model)
            payload = checkpoint_payload(
                bare_model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_validation_loss,
                bad_epochs,
                history,
            )
            atomic_torch_save(payload, last_checkpoint)
            export_metadata = {
                "schema_version": "1.0.0",
        "method": "bert_base_cased_wp_lm_tied_200d_table",
                "epoch": epoch,
                "validation": validation_metrics,
                "train": train_metrics,
                "base_model": config["tokenizer"]["base_model"],
                "wp_dimension": int(config["wp_dimension"]),
                "canonical_wp_rows": 1731,
                "active_mp_wp_rows": len(active_wp_classes),
                "inactive_rows_excluded_from_softmax": 1731 - len(active_wp_classes),
                "site_symmetry_is_context_not_prediction_target": True,
                "target_orbit_site_symmetry_is_masked": True,
                "target_orbit_multiplicity_is_masked": True,
                "formula_and_elements_are_context": True,
                "wp_input_output_weights_tied": True,
                "mask_probability": float(training["wp_mask_probability"]),
            }
            export_wp_table(
                bare_model, wp_rows, output_dir / "last_wp_vectors", export_metadata
            )
            if improved:
                atomic_torch_save(payload, output_dir / "checkpoint_best.pt")
                export_wp_table(
                    bare_model, wp_rows, output_dir / "best_wp_vectors", export_metadata
                )
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
            )
        stop_tensor = torch.tensor(
            int(bad_epochs >= patience), dtype=torch.int32, device=device
        )
        if world_size > 1:
            dist.broadcast(stop_tensor, src=0)
        if int(stop_tensor.item()):
            if rank == 0:
                print(f"Early stopping after epoch {epoch}", flush=True)
            break

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
