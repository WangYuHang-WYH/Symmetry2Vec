#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TORCHRUN="${TORCHRUN:-torchrun}"

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

"${TORCHRUN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_lm_wp_200.py \
  --config configs/lm.json
