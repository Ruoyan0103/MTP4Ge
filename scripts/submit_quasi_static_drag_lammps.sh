#!/bin/bash
#SBATCH --job-name=mtp_qsd_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/qsd_lammps_%j.out
#SBATCH --error=logs/qsd_lammps_%j.err
#
# Quasi-static drag test (src/physical_validation/quasi_static_drag.py --backend lammps) via
# LAMMPS (pair_style hybrid/overlay mtp nlh, via src/utils/mtp_calculator.py's
# MTPLammpsCalculator — one LAMMPS subprocess call per displacement step)
# — for potentials whose radial basis type `mlp calculate_efs` cannot load
# ("Wrong radial basis type"); see config/lammps/phonon_dispersion_mtp.in.
#
# Usage:
#   sbatch scripts/submit_quasi_static_drag_lammps.sh                                  # default pot
#   sbatch scripts/submit_quasi_static_drag_lammps.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_quasi_static_drag_lammps.sh results/potentials/my_pot.almtp --a0 5.7547 --n-steps 15

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/20.mtp}"
shift || true

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Quasi-static drag (LAMMPS, pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $*"
date

# NOTE: like submit_vacancy_migration_lammps.sh, LAMMPS is invoked directly
# (no srun) — many single-point calls are issued here (steps x directions),
# and skipping srun's per-launch overhead matters at that volume.
python src/physical_validation/quasi_static_drag.py \
    --pot "$POT" \
    --backend lammps \
    --lammps "$LAMMPS_BIN" \
    --outdir results/tests/quasi_static_drag_lammps \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/quasi_static_drag_lammps 2>/dev/null || echo results/tests/quasi_static_drag_lammps)"
date
