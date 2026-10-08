#!/bin/bash
#SBATCH --job-name=vw_ctx_readers
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
mkdir -p logs results/context_scaling/readers/qwen3_8b \
  results/context_scaling/readers/qwen_14b \
  results/context_scaling/aggregate

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
    >"logs/context_readers_${SLURM_JOB_ID}_${label}.out" \
    2>"logs/context_readers_${SLURM_JOB_ID}_${label}.err" &
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

run_step qwen3_0 \
  "$PYTHON_BIN" -u scripts/run_context_scaling.py \
  --reader-cell qwen3_8b --shard-index 0 --num-shards 2
run_step qwen3_1 \
  "$PYTHON_BIN" -u scripts/run_context_scaling.py \
  --reader-cell qwen3_8b --shard-index 1 --num-shards 2
run_step qwen14_0 \
  "$PYTHON_BIN" -u scripts/run_context_scaling.py \
  --reader-cell qwen_14b --shard-index 0 --num-shards 2
run_step qwen14_1 \
  "$PYTHON_BIN" -u scripts/run_context_scaling.py \
  --reader-cell qwen_14b --shard-index 1 --num-shards 2
wait_steps

"$PYTHON_BIN" scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen3_8b \
  --output results/context_scaling/readers/qwen3_8b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
"$PYTHON_BIN" scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen_14b \
  --output results/context_scaling/readers/qwen_14b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
"$PYTHON_BIN" scripts/aggregate_context_scaling.py

echo "CONTEXT READER PHASE COMPLETE $(date --iso-8601=seconds)"
