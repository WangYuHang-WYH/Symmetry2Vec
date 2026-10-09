#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
GRAPH_DIR="${ROOT}/artifacts/kg_graph"
KGE_DIR="${ROOT}/artifacts/kg_transe_d200"

cd "${ROOT}"
"${PYTHON_BIN}" -m symmetry_kg.build_graph_kg \
  --base-graph-dir data/base_graph_source \
  --out-dir "${GRAPH_DIR}"
"${PYTHON_BIN}" -m symmetry_kg.train_transe_standard \
  --triples "${GRAPH_DIR}/triples.tsv" \
  --nodes "${GRAPH_DIR}/nodes.tsv" \
  --out-dir "${KGE_DIR}" \
  --dim 200 --epochs 1000 --batch-size 4096 --negative-ratio 4 \
  --lr 0.001 --margin 1.0 --distance-norm 1 --seed 13 \
  --device "${DEVICE}" --top-k 10
"${PYTHON_BIN}" scripts/extract_wp_embeddings.py \
  --input "${KGE_DIR}/entity_embeddings.tsv" \
  --output "${KGE_DIR}/wp_entity_embeddings.tsv"
