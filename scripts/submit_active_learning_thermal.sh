#!/bin/bash
#SBATCH --job-name=mtp_al_thermal
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/al_thermal_%j.out
#SBATCH --error=logs/al_thermal_%j.err

# Thermal active learning SLURM job.
#
# Usage:
#   sbatch scripts/submit_active_learning_thermal.sh [--iter-per-temp N]
#   Initial potential is read from config/active_learning.yaml (init_train_pot).
#
# Runs 3 temperatures (1500/2000/2500 K) × N iterations each using a perfect
# bulk 2×2×2 Ge diamond supercell. Run after SIA AL (Step 3).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load vasp/6.4.3 2>/dev/null || true
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

module load gcc/11.2.0 openmpi/4.1.2
module load openblas 2>/dev/null || true
export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_NTASKS:-10}

echo "=== Thermal AL SLURM job ==="
echo "  Node     : $(hostname)"
echo "  Time     : $(date)"

bash scripts/03_active_learning_thermal.sh "$@"

echo "Done: $(date)"
