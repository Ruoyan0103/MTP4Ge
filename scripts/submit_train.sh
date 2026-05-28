#!/bin/bash
#SBATCH --job-name=mtp_train
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load openblas

export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"

echo "=== MTP Training ==="
echo "  Node    : $(hostname)"
echo "  Time    : $(date)"
echo "  Train   : data/train.cfg"

python src/train.py --config config/training.yaml

echo "Done. Potential saved to results/potentials/pot.almtp"
