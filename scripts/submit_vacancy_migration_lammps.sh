#!/bin/bash
#SBATCH --job-name=mtp_vac_migration_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/vac_migration_lammps_%j.out
#SBATCH --error=logs/vac_migration_lammps_%j.err
#
# Vacancy migration barriers (1NN + 2NN hops) via climbing-image NEB, using
# LAMMPS (pair_style hybrid/overlay mtp nlh, via src/utils/mtp_calculator.py's
# MTPLammpsCalculator — one LAMMPS subprocess call per energy/force
# evaluation) instead of `mlp calculate_efs`
# (src/physical_validation/vacancy_migration.py --backend lammps). Needed for potentials whose
# radial basis type `mlp calculate_efs` cannot load ("Wrong radial basis
# type"); see config/lammps/phonon_dispersion_mtp.in.
#
# NEB + endpoint relaxation issues many single-point evaluations (BFGS steps
# x images x 2 shells), so this can take much longer than the other LAMMPS
# tests — budget accordingly.
#
# Usage:
#   sbatch scripts/submit_vacancy_migration_lammps.sh                                  # default pot
#   sbatch scripts/submit_vacancy_migration_lammps.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_vacancy_migration_lammps.sh results/potentials/my_pot.almtp --a0 5.7547 --n-images 7

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/9527.mtp}"
shift || true

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Vacancy migration (NEB, LAMMPS pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $*"
date

# NOTE: unlike the other submit_*_lammps.sh scripts, LAMMPS is invoked
# directly (no srun) — NEB/BFGS issues thousands of single-point calls here,
# and skipping srun's per-launch overhead matters at that volume.
python src/physical_validation/vacancy_migration.py \
    --pot "$POT" \
    --backend lammps \
    --lammps "$LAMMPS_BIN" \
    --outdir results/tests/vacancy_migration_lammps \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/vacancy_migration_lammps 2>/dev/null || echo results/tests/vacancy_migration_lammps)"
date
