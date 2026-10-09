#!/usr/bin/env python3
"""Extend bert-base-cased so every stored MP word is exactly one token."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from transformers import AddedToken, BertTokenizerFast


def records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_word(tokenizer: BertTokenizerFast, word: str) -> bool:
    identifiers = tokenizer(word, add_special_tokens=False)["input_ids"]
    return len(identifiers) == 1 and identifiers[0] != tokenizer.unk_token_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_dir = Path(config["corpus"]["output_dir"])
    tokenizer_config = config["tokenizer"]
    output_dir = Path(tokenizer_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        split: corpus_dir / f"{split}.jsonl.gz" for split in ("train", "validation")
    }
    wp_vocabulary_path = corpus_dir / "canonical_wp_vocabulary.tsv"
    with wp_vocabulary_path.open("r", encoding="utf-8", newline="") as handle:
        wp_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(wp_rows) != 1731:
        raise RuntimeError(f"Expected 1731 WP rows, found {len(wp_rows)}")

    words: set[str] = {row["surface_token"] for row in wp_rows}
    word_counts: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    for split, path in split_paths.items():
        for record in records(path):
            words.update(record["tokens"])
            word_counts.update(record["tokens"])
            split_samples[split] += 1

    base_model = tokenizer_config["base_model"]
    base_tokenizer = BertTokenizerFast.from_pretrained(
        base_model, local_files_only=True, do_lower_case=False
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        base_model, local_files_only=True, do_lower_case=False
    )
    base_vocabulary_size = len(tokenizer)
    addition_words = sorted(word for word in words if not atomic_word(tokenizer, word))
    additions = [
        AddedToken(word, single_word=True, normalized=False) for word in addition_words
    ]
    added_count = tokenizer.add_tokens(additions)
    if added_count != len(addition_words):
        raise RuntimeError(
            f"Tokenizer added {added_count} tokens; expected {len(addition_words)}"
        )
    tokenizer.save_pretrained(output_dir)

    initialization_path = output_dir / "added_token_initialization.tsv"
    with initialization_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["token_id", "token", "base_piece_ids", "base_pieces"])
        for word in addition_words:
            token_id = tokenizer.convert_tokens_to_ids(word)
            base_piece_ids = base_tokenizer(
                word, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
            base_pieces = base_tokenizer.convert_ids_to_tokens(base_piece_ids)
            writer.writerow(
                [
                    token_id,
                    word,
                    ",".join(map(str, base_piece_ids)),
                    " ".join(base_pieces),
                ]
            )

    failures: Counter[str] = Counter()
    unique_non_atomic = sorted(word for word in words if not atomic_word(tokenizer, word))
    unique_unknown = sorted(
        word
        for word in words
        if tokenizer.convert_tokens_to_ids(word) == tokenizer.unk_token_id
    )
    split_audits: dict[str, dict[str, int]] = {}
    maximum_words = int(tokenizer_config["max_length"]) - 2
    for split, path in split_paths.items():
        samples = 0
        word_tokens = 0
        unknown_tokens = 0
        non_atomic_words = 0
        samples_over_limit = 0
        maximum_sample_words = 0
        for record in records(path):
            tokens = record["tokens"]
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            counts = Counter(value for value in encoding.word_ids() if value is not None)
            bad = sum(counts[index] != 1 for index in range(len(tokens)))
            unknown = sum(
                identifier == tokenizer.unk_token_id for identifier in encoding["input_ids"]
            )
            if len(encoding["input_ids"]) != len(tokens) + 2:
                failures["length_mismatch_samples"] += 1
            non_atomic_words += bad
            unknown_tokens += unknown
            samples += 1
            word_tokens += len(tokens)
            samples_over_limit += int(len(tokens) > maximum_words)
            maximum_sample_words = max(maximum_sample_words, len(tokens))
        failures["non_atomic_words"] += non_atomic_words
        failures["unknown_tokens"] += unknown_tokens
        split_audits[split] = {
            "samples": samples,
            "word_tokens": word_tokens,
            "non_atomic_words": non_atomic_words,
            "unknown_tokens": unknown_tokens,
            "samples_over_single_sequence_limit": samples_over_limit,
            "maximum_sample_words": maximum_sample_words,
        }

    wp_output_path = output_dir / "wp_vocabulary_1731.tsv"
    with wp_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "entity_id",
                "surface_token",
                "token_id",
                "observed_in_mp",
                "corpus_occurrences",
            ]
        )
        for row in wp_rows:
            surface = row["surface_token"]
            writer.writerow(
                [
                    row["row_index"],
                    row["entity_id"],
                    surface,
                    tokenizer.convert_tokens_to_ids(surface),
                    row["observed_in_mp"],
                    word_counts[surface],
                ]
            )

    positive_failures = {key: value for key, value in failures.items() if value > 0}
    audit = {
        "schema_version": "1.0.0",
        "passed_one_word_one_token": not positive_failures
        and not unique_non_atomic
        and not unique_unknown,
        "failures": positive_failures,
        "unique_non_atomic_words": unique_non_atomic,
        "unique_unknown_words": unique_unknown,
        "base_model": base_model,
        "base_vocabulary_size": base_vocabulary_size,
        "candidate_words_including_all_canonical_wp": len(words),
        "added_tokens": added_count,
        "final_vocabulary_size": len(tokenizer),
        "canonical_wp_tokens": len(wp_rows),
        "atomic_canonical_wp_tokens": sum(
            atomic_word(tokenizer, row["surface_token"]) for row in wp_rows
        ),
        "max_length": int(tokenizer_config["max_length"]),
        "maximum_words_without_chunking": maximum_words,
        "splits": split_audits,
    }
    (output_dir / "tokenizer_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if not audit["passed_one_word_one_token"]:
        raise SystemExit("Tokenizer audit failed")


if __name__ == "__main__":
    main()
