#!/bin/bash
#SBATCH --job-name=vw_ctx_repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
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

python -V
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

# Rerun Qwen2.5-14B shard 0 after an interrupted reader job.
# The JSONL writer resumes from any rows already present.
srun python -u scripts/run_context_scaling.py \
  --reader-cell qwen_14b --shard-index 0 --num-shards 2

python scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen3_8b \
  --output results/context_scaling/readers/qwen3_8b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
python scripts/merge_jsonl.py \
  --input results/context_scaling/readers/qwen_14b \
  --output results/context_scaling/readers/qwen_14b/merged.jsonl \
  --expected-data data/musique_long_context_pool.jsonl
python scripts/aggregate_context_scaling.py

echo "CONTEXT REPAIR AND AGGREGATION COMPLETE $(date --iso-8601=seconds)"
