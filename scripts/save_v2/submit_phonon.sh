#!/bin/bash
#SBATCH --job-name=mtp_phonon
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/phonon_%j.out
#SBATCH --error=logs/phonon_%j.err
#
# Phonon dispersion + total DOS via LAMMPS (pair_style mlip) single-point
# force evaluations on phonopy-displaced supercells (src/physical_validation/phonon_dispersion.py
# --method lammps), matching the ilearn LAMMPS-based workflow.
#
# Usage:
#   sbatch scripts/submit_phonon.sh                                  # default pot
#   sbatch scripts/submit_phonon.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_phonon.sh results/potentials/my_pot.almtp --supercell 4 --npoints 31

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-2026-07-19-v2.almtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Phonon dispersion (LAMMPS, pair_style mlip) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $@"
date

python src/physical_validation/phonon_dispersion.py \
    --pot "$POT" \
    --method lammps \
    --lammps "srun $LAMMPS_BIN" \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/phonon_dispersion 2>/dev/null || echo results/tests/phonon_dispersion)"
date
