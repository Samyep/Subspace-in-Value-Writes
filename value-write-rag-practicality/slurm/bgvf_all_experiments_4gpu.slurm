#!/bin/bash
#SBATCH --job-name=vw_rag_all
#SBATCH --account=bgvf-dtai-gh
#SBATCH --partition=ghx4-interactive
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
mkdir -p logs results/reranker/bge_v2_m3 results/scout/llama_3b_l17

module load default
source /sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh
conda activate "${VW_RAG_CONDA_ENV:-base}"

export HF_HOME=${HF_HOME:-$REPO_ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=hsn
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
    >"logs/all_${SLURM_JOB_ID}_${label}.out" \
    2>"logs/all_${SLURM_JOB_ID}_${label}.err" &
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

jsonl_complete() {
  local path=$1
  local summary=$2
  [[ -s "$path" && -s "$summary" ]] || return 1
  [[ $(wc -l <"$path") -eq 400 ]]
}

# The two selectors are independent and share their outputs across all readers.
if ! jsonl_complete \
  results/reranker/bge_v2_m3/selections.jsonl \
  results/reranker/bge_v2_m3/selections.summary.json; then
  run_step reranker \
    "$PYTHON_BIN" -u scripts/score_reranker.py \
    --output results/reranker/bge_v2_m3/selections.jsonl \
    --max-examples 0
fi
if ! jsonl_complete \
  results/scout/llama_3b_l17/selections.jsonl \
  results/scout/llama_3b_l17/selections.summary.json; then
  run_step scout \
    "$PYTHON_BIN" -u scripts/score_truncated_scout.py \
    --scout-cell llama_3b \
    --max-examples 0 \
    --no-require-exact-selection-agreement
fi
wait_steps

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

from vw_rag.io_utils import load_jsonl

data_ids = {
    str(row["example_id"])
    for row in load_jsonl(Path("data/musique_pooled_rag_confirm.jsonl"))
}
reranker = load_jsonl(Path("results/reranker/bge_v2_m3/selections.jsonl"))
scout = load_jsonl(Path("results/scout/llama_3b_l17/selections.jsonl"))
assert len(reranker) == 400 and {str(row["example_id"]) for row in reranker} == data_ids
assert len(scout) == 400 and {str(row["example_id"]) for row in scout} == data_ids
print("PASS: both shared selectors contain the frozen 400 IDs")
print(
    "Released-run agreement: value=",
    sum(row["value_full_model_selection_match"] for row in scout) / len(scout),
    "attention=",
    sum(row["attention_full_model_selection_match"] for row in scout) / len(scout),
)
PY

# Long jobs enter first. Later steps wait for one of the four isolated GPUs.
run_step cost_qwen14 bash -lc \
  "'$PYTHON_BIN' -u scripts/benchmark_value_scoring.py --cell qwen_14b --max-examples 0 --allow-selection-mismatch && '$PYTHON_BIN' -u scripts/benchmark_generation.py --cell qwen_14b --max-examples 0"
run_step scout_reader_qwen14 \
  "$PYTHON_BIN" -u scripts/run_scout_reader.py \
  --reader-cell qwen_14b \
  --max-examples 0
run_step cost_qwen3 bash -lc \
  "'$PYTHON_BIN' -u scripts/benchmark_value_scoring.py --cell qwen3_8b --max-examples 0 --allow-selection-mismatch && '$PYTHON_BIN' -u scripts/benchmark_generation.py --cell qwen3_8b --max-examples 0"
run_step scout_reader_qwen3 \
  "$PYTHON_BIN" -u scripts/run_scout_reader.py \
  --reader-cell qwen3_8b \
  --max-examples 0
run_step cost_qwen7 bash -lc \
  "'$PYTHON_BIN' -u scripts/benchmark_value_scoring.py --cell qwen_7b --max-examples 0 --allow-selection-mismatch && '$PYTHON_BIN' -u scripts/benchmark_generation.py --cell qwen_7b --max-examples 0"
run_step cost_llama3 bash -lc \
  "'$PYTHON_BIN' -u scripts/benchmark_value_scoring.py --cell llama_3b --max-examples 0 --allow-selection-mismatch && '$PYTHON_BIN' -u scripts/benchmark_generation.py --cell llama_3b --max-examples 0"
run_step reader_qwen14 \
  "$PYTHON_BIN" -u scripts/run_reranker_reader.py --cell qwen_14b --max-examples 0
run_step reader_qwen3 \
  "$PYTHON_BIN" -u scripts/run_reranker_reader.py --cell qwen3_8b --max-examples 0
run_step reader_qwen7 \
  "$PYTHON_BIN" -u scripts/run_reranker_reader.py --cell qwen_7b --max-examples 0
run_step reader_llama3 \
  "$PYTHON_BIN" -u scripts/run_reranker_reader.py --cell llama_3b --max-examples 0
wait_steps

"$PYTHON_BIN" -u scripts/audit_scout_prefix.py
"$PYTHON_BIN" -u scripts/aggregate_practicality.py
"$PYTHON_BIN" -u scripts/aggregate_scout.py
echo "ALL EXPERIMENTS COMPLETE $(date --iso-8601=seconds)"
