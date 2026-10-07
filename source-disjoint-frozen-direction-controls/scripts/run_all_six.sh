#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$BUNDLE_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BUNDLE_ROOT/results/frozen_direction/reproduction_v4}"
RAG_ROOT="${RAG_ROOT:-$REPO_ROOT/value-write-rag-practicality}"
EXPERIMENT_ROOT="$BUNDLE_ROOT/dependencies/controlled_task"
LEGACY_ROOT="$BUNDLE_ROOT/dependencies/legacy"
DATA_PATH="$BUNDLE_ROOT/data/hotpot_400_prompts.jsonl"
PROTOCOL_PATH="$BUNDLE_ROOT/experiments/PROTOCOL_v4.md"

if [[ ! -f "$RAG_ROOT/vw_rag/value_write.py" ]]; then
  echo "Missing sibling RAG bundle: $RAG_ROOT" >&2
  exit 2
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$BUNDLE_ROOT"

PREFLIGHT_OUTPUT="$BUNDLE_ROOT/results/frozen_direction/main_v4/preflight/gpu_norm_regression_reproduction_$(date -u +%Y%m%dT%H%M%SZ).json"
"$PYTHON_BIN" -u experiments/gpu_validate_v4_norm.py --output "$PREFLIGHT_OUTPUT"

for cell in llama_3b llama_8b qwen_7b qwen_14b mistral_7b mistral_12b; do
  "$PYTHON_BIN" -u experiments/frozen_direction_intervention_v4.py \
    --cell "$cell" \
    --output "$OUTPUT_ROOT/main_v4/$cell" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --rag-root "$RAG_ROOT" \
    --legacy-root "$LEGACY_ROOT" \
    --data "$DATA_PATH" \
    --protocol-file "$PROTOCOL_PATH"
done

"$PYTHON_BIN" -u experiments/aggregate_frozen_controls_v4.py \
  --input "$OUTPUT_ROOT/main_v4" \
  --output "$OUTPUT_ROOT/aggregate_v4" \
  --norms-only

"$PYTHON_BIN" -u experiments/aggregate_frozen_controls_v4.py \
  --input "$OUTPUT_ROOT/main_v4" \
  --output "$OUTPUT_ROOT/aggregate_v4"

"$PYTHON_BIN" -u experiments/render_frozen_report_tables.py \
  --directory "$OUTPUT_ROOT/aggregate_v4"

echo "Reproduction complete: $OUTPUT_ROOT"
