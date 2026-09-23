#!/bin/bash
#SBATCH --job-name=AL
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=36:00:00
##SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/al_%j.out
#SBATCH --error=logs/al_%j.err

# Active learning SLURM job.
#
# Usage:
#   sbatch scripts/submit_active_learning.sh [extra args passed to active_learning.py]
#   Config is read from config/active_learning.yaml (and config/training.yaml).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load vasp/6.4.3 2>/dev/null || true
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

module load gcc/15.2.0 openmpi/5.0.10
module load openblas 2>/dev/null || true
export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"
# 1 thread per rank, matching --cpus-per-task=1 above: everything in this
# pipeline (LAMMPS, mlp train, mlp select_add) parallelizes via MPI ranks
# (srun -n N), not per-rank OpenMP threading, so anything above 1 here just
# oversubscribes each rank's single core and risks OOM (see src/active_learning_structs/
# active_learning.py's select_add()/retrain() for the same reasoning).
export OMP_NUM_THREADS=1

echo "=== AL SLURM job ==="
echo "  Node     : $(hostname)"
echo "  Time     : $(date)"

python src/active_learning_structs/active_learning.py \
    --al-config config/active_learning.yaml \
    "$@"

echo "Done: $(date)"
