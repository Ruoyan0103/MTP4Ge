#!/bin/bash
#SBATCH --job-name=mtp_prune
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/prune.out
#SBATCH --error=logs/prune.err

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load openblas

export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"

POT="${1:-results/potentials/pot_al.almtp}"
ROW="${2:-0}"

echo "=== MTP Pruning ==="
echo "  Node      : $(hostname)"
echo "  Time      : $(date)"
echo "  Potential : $POT"
echo "  Pareto row: $ROW"

python src/prune.py --pot "$POT" --row "$ROW"

echo "Done. Pruned potential: results/potentials/pruned.almtp"
