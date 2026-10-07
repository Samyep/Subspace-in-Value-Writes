#!/usr/bin/env bash
#SBATCH --job-name=frozen_pc1_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

srun bash "$BUNDLE_ROOT/scripts/run_all_six.sh"
