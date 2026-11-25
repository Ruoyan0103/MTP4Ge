#!/bin/bash
#SBATCH --job-name=mtp_sia_migration
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/sia_migration_%j.out
#SBATCH --error=logs/sia_migration_%j.err
#
# Self-interstitial migration barriers (T-T via H, X-X dumbbell translation,
# X-T conversion) via climbing-image NEB, using LAMMPS (pair_style
# hybrid/overlay mtp nlh, via src/utils/mtp_calculator.py's
# MTPLammpsCalculator — one LAMMPS subprocess per energy/force evaluation)
# (src/physical_validation/interstitial_migration.py --backend lammps).
#
# Default: 3x3x3 supercell (217 atoms), 7 images, 3 paths. Endpoint relaxation
# + NEB issue a few thousand single-point calls; the 2x2x2 smoke test took
# ~1.5 min per path with 3 images, so 3 h is a generous budget for the
# full run.
#
# Usage:
#   sbatch scripts/submit_interstitial_migration.sh results/potentials/pot_1019/bal.mtp --a0 5.7609
#   sbatch scripts/submit_interstitial_migration.sh results/potentials/pot_1019/bal.mtp --a0 5.7609 \
#       --paths T-T X-T --n-images 9
# Output: results/tests/<pot parent dir>/interstitial_migration/ (unless --outdir given)

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"
mkdir -p logs

POT="${1:-results/potentials/pot_1019/bal.mtp}"
shift || true

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== SIA migration (CI-NEB, LAMMPS pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $*"
date

# LAMMPS invoked directly (no srun): thousands of single-point calls, srun's
# per-launch overhead would dominate (same choice as submit_vacancy_migration_lammps.sh).
python src/physical_validation/interstitial_migration.py \
    --pot "$POT" \
    --backend lammps \
    --lammps "$LAMMPS_BIN" \
    "$@"

echo "=== Done ==="
date
