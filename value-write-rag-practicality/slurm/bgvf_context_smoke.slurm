#!/bin/bash
#SBATCH --job-name=vw_ctx_smoke
#SBATCH --account=bgvf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT=${VW_RAG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
mkdir -p logs results/context_scaling/smoke/bge results/context_scaling/smoke/scout

module load default
source /sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh
conda activate base

export HF_HOME=${HF_HOME:-$REPO_ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16

python -V
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

srun python -u scripts/score_long_reranker.py \
  --max-examples 2 \
  --output results/context_scaling/smoke/bge/selections.jsonl

srun python -u scripts/score_long_scout.py \
  --max-examples 2 \
  --output results/context_scaling/smoke/scout/selections.jsonl

srun python -u scripts/run_context_scaling.py \
  --reader-cell qwen3_8b \
  --max-examples 2 \
  --bge-selections results/context_scaling/smoke/bge/selections.jsonl \
  --scout-selections results/context_scaling/smoke/scout/selections.jsonl \
  --output-dir results/context_scaling/smoke/qwen3_8b

srun python -u scripts/run_context_scaling.py \
  --reader-cell qwen_14b \
  --max-examples 2 \
  --bge-selections results/context_scaling/smoke/bge/selections.jsonl \
  --scout-selections results/context_scaling/smoke/scout/selections.jsonl \
  --output-dir results/context_scaling/smoke/qwen_14b
