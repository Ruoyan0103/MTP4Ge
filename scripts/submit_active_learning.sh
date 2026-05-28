#!/usr/bin/env bash
#SBATCH --job-name=MTP4Ge_AL
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/al_%j.out
#SBATCH --error=logs/al_%j.err
#
# Active learning loop for MTP4Ge.
#
# Each iteration runs LAMMPS MD (32 MPI tasks) then one or more VASP
# single-point calculations (32 MPI tasks each), both via srun.
# The srun commands are specified in config/active_learning.yaml as:
#   lammps_binary: srun -n 32 lmp
#   vasp_command:  srun -n 32 vasp_std
#
# Usage:
#   sbatch scripts/submit_active_learning.sh [pot.almtp] [--max-iter N]

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate

export OMP_NUM_THREADS=1
export VASP_PP_PATH=/appl/soft/phys/vasp/potpaw_PBE.64

echo "=== Active Learning Job ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Nodes     : ${SLURM_JOB_NUM_NODES:-1}  Tasks: ${SLURM_NTASKS:-32}"
echo "  Potential : $POT"
date

python src/active_learning.py --pot "$POT" "$@"

echo "=== Done ==="
date
