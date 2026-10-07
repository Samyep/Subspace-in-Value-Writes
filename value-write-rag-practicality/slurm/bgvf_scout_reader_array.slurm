#!/bin/bash
#SBATCH --job-name=vw_rag_scout_reader
#SBATCH --account=bgvf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=${VW_RAG_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"

module load default
source /sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh
conda activate "${VW_RAG_CONDA_ENV:-base}"

export HF_HOME=${HF_HOME:-$REPO_ROOT/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

READERS=(qwen_14b qwen3_8b)
READER=${READERS[$SLURM_ARRAY_TASK_ID]}
test -s results/scout/llama_3b_l17/selections.jsonl
test -s results/reranker/bge_v2_m3/selections.jsonl
mkdir -p "results/scout_readers/${READER}"

python -V
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

srun python -u scripts/run_scout_reader.py \
  --reader-cell "$READER" \
  --max-examples 0
