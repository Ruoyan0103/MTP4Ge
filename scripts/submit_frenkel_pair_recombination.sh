#!/bin/bash
#SBATCH --job-name=mtp_frenkel_pair
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/frenkel_pair_%j.out
#SBATCH --error=logs/frenkel_pair_%j.err
#
# Close Frenkel-pair stability and recombination barrier
# (src/physical_validation/frenkel_pair_recombination.py --backend lammps):
# vacancy + SIA (X dumbbell by default) at increasing separation shells in a
# 3x3x3 (216-site) cell, fixed-cell LAMMPS relaxation, Wigner-Seitz
# classification, then CI-NEB (one LAMMPS subprocess per image evaluation,
# MTPLammpsCalculator) from the closest stable FP shells to the perfect crystal.
#
# The relaxations take seconds; the NEB dominates (images x BFGS steps x
# subprocess launches), hence the 1 h budget.
#
# Usage:
#   sbatch scripts/submit_frenkel_pair_recombination.sh                                     # defaults
#   sbatch scripts/submit_frenkel_pair_recombination.sh results/potentials/pot_1019/bal.mtp --a0 5.7609
#   sbatch scripts/submit_frenkel_pair_recombination.sh results/potentials/pot_1019/bal.mtp --a0 5.7609 \
#       --interstitial T --outdir results/tests/pot_1019/frenkel_pair_recombination_T

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

echo "=== Frenkel-pair recombination (LAMMPS pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Potential : $POT"
echo "  Extra args: $*"
date

# LAMMPS invoked directly (no srun): NEB issues thousands of single-point calls.
python src/physical_validation/frenkel_pair_recombination.py \
    --pot "$POT" \
    --backend lammps \
    --lammps "$LAMMPS_BIN" \
    "$@"

echo "=== Done ==="
date
