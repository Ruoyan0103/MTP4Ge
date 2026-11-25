#!/bin/bash
#SBATCH --job-name=mtp_threshold_disp
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=384
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --output=logs/threshold_disp_%j.out
#SBATCH --error=logs/threshold_disp_%j.err
#
# Threshold displacement energy Ed of diamond Ge (recoil MD + Wigner-Seitz),
# src/physical_validation/threshold_displacement.py, LAMMPS pair_style
# hybrid/overlay mtp nlh.
#
# One node-exclusive medium node is packed with many small LAMMPS runs:
# WORKERS concurrent runs x RANKS MPI ranks each (default 96 x 4 = 384 cores),
# each launched as its own `srun --exact` job step. Every (direction, energy,
# repeat) is an independent run directory with a result.json, so a job that
# hits the time limit can simply be resubmitted with the same arguments and
# continues where it stopped.
#
# Defaults: 6x6x6 cell (1728 atoms), 10 K, 3 thermal repeats, 8 lattice
# directions, 4->60 eV in 2 eV steps (4 coarse energies in flight per series),
# bisection to 0.5 eV, 3 ps NVE + dt/reset. Measured on the login node with
# pot_1019/bal.mtp: ~9e-5 s per atom-step serial, i.e. ~4-5 min per serial
# 1728-atom, 3 ps run (~1600 adaptive steps), ~1-1.5 min on 4 ranks.
#
# Usage:
#   sbatch scripts/submit_threshold_displacement.sh                                       # default pot
#   sbatch scripts/submit_threshold_displacement.sh results/potentials/pot_1019/bal.mtp
#   sbatch scripts/submit_threshold_displacement.sh results/potentials/pot_1019/bal.mtp --n-random 20 --repeats 1
#   WORKERS=48 RANKS=8 sbatch scripts/submit_threshold_displacement.sh results/potentials/pot_1019/bal.mtp
# Check the queue first (see shared memory: compare partitions with --test-only).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot_1019/bal.mtp}"
shift || true

WORKERS="${WORKERS:-96}"
RANKS="${RANKS:-4}"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Threshold displacement energy (LAMMPS pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Potential : $POT"
echo "  Packing   : $WORKERS runs x $RANKS ranks"
echo "  Extra args: $*"
date

python src/physical_validation/threshold_displacement.py \
    --pot "$POT" \
    --supercell 6 \
    --temperature 10 \
    --repeats 3 \
    --batch 4 \
    --workers "$WORKERS" \
    --lammps "srun --exact -n $RANKS -c 1 --mem-per-cpu=1500M $LAMMPS_BIN" \
    "$@"

echo "=== Done ==="
echo "  Output : results/tests/$(basename "$(dirname "$(realpath "$POT")")")/threshold_displacement/"
date
