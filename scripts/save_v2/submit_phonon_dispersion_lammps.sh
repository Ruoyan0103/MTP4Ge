#!/bin/bash
#SBATCH --job-name=mtp_phonon_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/phonon_lammps_%j.out
#SBATCH --error=logs/phonon_lammps_%j.err
#
# Phonon dispersion + total DOS via LAMMPS (src/physical_validation/phonon_dispersion.py
# --method lammps-nlh) using pair_style hybrid/overlay mtp nlh — for
# potentials whose radial basis type neither `mlp calculate_efs` nor
# `pair_style mlip` (--method lammps-mlip) can load ("Wrong radial basis
# type"); see config/lammps/phonon_dispersion_mtp.in.
#
# Usage:
#   sbatch scripts/submit_phonon_dispersion_lammps.sh                                  # default pot
#   sbatch scripts/submit_phonon_dispersion_lammps.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_phonon_dispersion_lammps.sh results/potentials/my_pot.almtp --supercell 4 --npoints 31

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

echo "=== Phonon dispersion (LAMMPS, pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $*"
date

python src/physical_validation/phonon_dispersion.py \
    --pot "$POT" \
    --method lammps-nlh \
    --lammps "srun $LAMMPS_BIN" \
    --outdir results/tests/phonon_dispersion_lammps \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/phonon_dispersion_lammps 2>/dev/null || echo results/tests/phonon_dispersion_lammps)"
date
