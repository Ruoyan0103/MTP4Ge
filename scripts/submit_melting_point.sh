#!/bin/bash
#SBATCH --job-name=mtp_melting
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/melting_%j.out
#SBATCH --error=logs/melting_%j.err
#
# Two-phase coexistence melting point for diamond-cubic Ge (a0 = 5.7567 A, MTP equilibrated).
#
# Protocol:
#   1. NPT 500 K, 1 bar, 20 ps
#   2. NVT bottom half fixed, top half melted at 1500 K, 20 ps
#   3. NpH coexistence 1 bar, 200 ps — T evolves freely to Tm
#   4. 1 repeat (single test run; increase with --n-repeats for statistics)
#
# Usage:
#   sbatch scripts/submit_melting_point.sh                                    # default pot
#   sbatch scripts/submit_melting_point.sh results/potentials/pot-init.almtp
#   sbatch scripts/submit_melting_point.sh results/potentials/pot-init.almtp --n-repeats 12
#
# Quick test (4×4×8 cell, 1 repeat, ~10 min):
#   sbatch scripts/submit_melting_point.sh results/potentials/pot-init.almtp --quick

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-init.almtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"

echo "================================================================"
echo "  Melting Point: Two-Phase Coexistence (NpH method)"
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-8}"
echo "  Potential : $POT"
echo "  Extra args: $@"
echo "  Started   : $(date)"
echo "================================================================"

python tests/melting_point.py \
    --pot "$POT" \
    --lammps "srun $LAMMPS_BIN" \
    --np 1 \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "  Results: results/tests/melting_point/"
