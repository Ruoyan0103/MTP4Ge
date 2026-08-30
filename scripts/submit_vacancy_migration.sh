#!/bin/bash
#SBATCH --job-name=mtp_vac_migration
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/vac_migration_%j.out
#SBATCH --error=logs/vac_migration_%j.err
#
# Vacancy migration barriers (1NN + 2NN hops) via climbing-image NEB, using
# MTPCalculator (mlp calculate_efs per image, one subprocess call per
# evaluation) — serial, no LAMMPS/MPI required.
#
# Usage:
#   sbatch scripts/submit_vacancy_migration.sh                                   # default pot
#   sbatch scripts/submit_vacancy_migration.sh results/potentials/pot.almtp
#   sbatch scripts/submit_vacancy_migration.sh results/potentials/pot.almtp --a0 5.779 --n-images 7

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-/scratch/project_2012355/Paper_3/00-subsets/07-short_range/pot24_08_26.almtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Vacancy Migration (NEB) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Potential : $POT"
echo "  Extra args: $@"
date

python tests/vacancy_migration.py \
    --pot "$POT" \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/vacancy_migration 2>/dev/null || echo results/tests/vacancy_migration)"
date
