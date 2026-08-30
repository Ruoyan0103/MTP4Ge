#!/bin/bash
#SBATCH --job-name=mtp_melting
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/melting_%A_%a.out
#SBATCH --error=logs/melting_%A_%a.err
#SBATCH --array=0-9   # overridden by submit command to match --n-repeats
#
# Two-phase coexistence melting point for diamond-cubic Ge (a0 = 5.7567 A, MTP equilibrated).
#
# Protocol:
#   1. NPT 500 K, 1 bar, 20 ps
#   2. NVT bottom half fixed, top half melted at 1500 K, 20 ps
#   3. NpH coexistence 1 bar, 200 ps — T evolves freely to Tm
#   4. Each repeat runs as its own independent SLURM array task (10 MPI ranks each).
#
# Usage:
#   # Step 1 — submit one array task per repeat (10 repeats by default):
#   sbatch scripts/submit_melting_point.sh                                    # default pot, 10 repeats
#   sbatch scripts/submit_melting_point.sh results/potentials/pot-init.almtp
#   sbatch --array=0-15 scripts/submit_melting_point.sh results/potentials/pot-init.almtp --n-repeats 16
#
#   # Step 2 — after all array tasks finish, collect Tm and write the summary:
#   python tests/melting_point.py --pot results/potentials/pot-init.almtp --aggregate
#
# Quick test (4×4×8 cell, 1 repeat, ~10 min):
#   sbatch --array=0 scripts/submit_melting_point.sh results/potentials/pot-init.almtp --quick

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
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"

echo "================================================================"
echo "  Melting Point: Two-Phase Coexistence (NpH method)"
echo "  Array Job : ${SLURM_ARRAY_JOB_ID:-local}  Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-10}"
echo "  Potential : $POT"
echo "  Extra args: $@"
echo "  Started   : $(date)"
echo "================================================================"

# One repeat per array task: --repeat-index pins this task to a single seed
# and skips aggregation (run `--aggregate` separately once all tasks finish).
python tests/melting_point.py \
    --pot "$POT" \
    --lammps "srun $LAMMPS_BIN" \
    --np "${SLURM_NTASKS:-10}" \
    --repeat-index "${SLURM_ARRAY_TASK_ID:-0}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "  Results: results/tests/melting_point/repeat_$(printf '%02d' "${SLURM_ARRAY_TASK_ID:-0}")/"
