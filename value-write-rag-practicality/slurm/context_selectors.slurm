#!/bin/bash
#SBATCH --job-name=vw_ctx_select
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT=${VW_RAG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
mkdir -p logs results/context_scaling/bge results/context_scaling/scout \
  results/hybrid_calibration results/hybrid/readers/qwen3_8b \
  results/hybrid/readers/qwen_14b

# Use the active Python environment by default. Set CONDA_INIT and
# VW_RAG_CONDA_ENV if the job must initialize and activate a Conda environment.
if [[ -n "${CONDA_INIT:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_INIT}"
fi
if [[ -n "${VW_RAG_CONDA_ENV:-}" ]]; then
  conda activate "${VW_RAG_CONDA_ENV}"
fi

export HF_HOME=${HF_HOME:-$REPO_ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

PYTHON_BIN=$(command -v python)
declare -a STEP_PIDS=()
declare -a STEP_LABELS=()

python -V
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

run_step() {
  local label=$1
  shift
  echo "START ${label} $(date --iso-8601=seconds)"
  srun \
    --exclusive \
    --exact \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=60G \
    --gpus=1 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    "$@" \
    >"logs/context_select_${SLURM_JOB_ID}_${label}.out" \
    2>"logs/context_select_${SLURM_JOB_ID}_${label}.err" &
  STEP_PIDS+=("$!")
  STEP_LABELS+=("$label")
}

wait_steps() {
  local failed=0
  local index
  local return_code
  for index in "${!STEP_PIDS[@]}"; do
    if wait "${STEP_PIDS[$index]}"; then
      echo "DONE ${STEP_LABELS[$index]} $(date --iso-8601=seconds)"
    else
      return_code=$?
      echo "FAILED ${STEP_LABELS[$index]} rc=${return_code} $(date --iso-8601=seconds)" >&2
      failed=1
    fi
  done
  STEP_PIDS=()
  STEP_LABELS=()
  return "$failed"
}

run_step scout0 \
  "$PYTHON_BIN" -u scripts/score_long_scout.py \
  --shard-index 0 --num-shards 2 \
  --output results/context_scaling/scout/part-000-of-002.jsonl

run_step scout1 \
  "$PYTHON_BIN" -u scripts/score_long_scout.py \
  --shard-index 1 --num-shards 2 \
  --output results/context_scaling/scout/part-001-of-002.jsonl

run_step bge_all bash -lc \
  "'$PYTHON_BIN' -u scripts/score_long_reranker.py --output results/context_scaling/bge/selections.jsonl && '$PYTHON_BIN' -u scripts/score_long_reranker.py --data data/hotpot_hybrid_calibration.jsonl --depths 10 --output results/hybrid_calibration/bge.jsonl"

run_step value_qwen3 \
  "$PYTHON_BIN" -u scripts/score_reader_value.py \
  --cell qwen3_8b \
  --data data/hotpot_hybrid_calibration.jsonl \
  --output results/hybrid_calibration/value_qwen3_8b.jsonl

# Slurm holds this fifth step until one of the four isolated GPUs is free.
run_step value_qwen14 \
  "$PYTHON_BIN" -u scripts/score_reader_value.py \
  --cell qwen_14b \
  --data data/hotpot_hybrid_calibration.jsonl \
  --output results/hybrid_calibration/value_qwen_14b.jsonl

wait_steps

"$PYTHON_BIN" scripts/merge_jsonl.py \
  --input results/context_scaling/scout \
  --output results/context_scaling/scout/selections.jsonl \
  --expected-data data/musique_long_context_pool.jsonl

"$PYTHON_BIN" scripts/calibrate_frozen_hybrid.py

if "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

gate = json.loads(Path("results/hybrid_calibration/hybrid_gate.json").read_text())
raise SystemExit(0 if gate["gate_passed"] else 1)
PY
then
  run_step hybrid_qwen3 \
    "$PYTHON_BIN" -u scripts/run_reranker_reader.py \
    --cell qwen3_8b \
    --reranker-selections results/hybrid_calibration/musique_qwen3_8b_selections.jsonl \
    --output-dir results/hybrid/readers/qwen3_8b
  run_step hybrid_qwen14 \
    "$PYTHON_BIN" -u scripts/run_reranker_reader.py \
    --cell qwen_14b \
    --reranker-selections results/hybrid_calibration/musique_qwen_14b_selections.jsonl \
    --output-dir results/hybrid/readers/qwen_14b
  wait_steps
  "$PYTHON_BIN" scripts/aggregate_hybrid.py
else
  echo "Frozen hybrid gate failed; MuSiQue hybrid generation was not run."
fi

echo "SELECTOR AND CALIBRATION PHASE COMPLETE $(date --iso-8601=seconds)"
