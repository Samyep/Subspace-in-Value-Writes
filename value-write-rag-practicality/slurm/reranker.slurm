#!/bin/bash
#SBATCH --job-name=vw_rag_rerank
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
mkdir -p results/reranker/bge_v2_m3

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python -V
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

srun python -u scripts/score_reranker.py \
  --output results/reranker/bge_v2_m3/selections.jsonl \
  --max-examples 0
