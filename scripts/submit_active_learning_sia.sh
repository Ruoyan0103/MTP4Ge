#!/bin/bash
#SBATCH --job-name=mtp_al_sia
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/al_sia_%j.out
#SBATCH --error=logs/al_sia_%j.err

# SIA-targeted active learning SLURM job.
#
# Usage:
#   sbatch scripts/submit_active_learning_sia.sh [--iter-per-structure N]
#   Initial potential is read from config/active_learning.yaml (init_train_pot).
#
# Runs 7 structure types × N iterations each.
# VASP is launched inline by active_learning.py using srun vasp_std.
# LAMMPS is launched using srun -n 1 lmp_mpi (single rank, required by MLIP selection).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load vasp/6.3.2 2>/dev/null || true
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

module load openblas 2>/dev/null || true
export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"

echo "=== SIA AL SLURM job ==="
echo "  Node     : $(hostname)"
echo "  Time     : $(date)"

bash scripts/save/03_active_learning_sia.sh "$@"

echo "Done: $(date)"
