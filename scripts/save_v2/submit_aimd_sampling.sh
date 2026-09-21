#!/bin/bash
#SBATCH --job-name=mtp_aimd_sampling
#SBATCH --account=project_2001625
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/aimd_sampling_%j.out
#SBATCH --error=logs/aimd_sampling_%j.err

# Liquid Ge MD sampling + high-fidelity DFT SLURM job.
#
# Usage:
#   sbatch scripts/submit_aimd_sampling.sh
#   sbatch scripts/submit_aimd_sampling.sh --pot results/potentials/aimd.almtp
#   sbatch scripts/submit_aimd_sampling.sh --temperatures 1000 1800
#
# Step 1: runs every liquid MD trajectory in data/aimd-sampling/config.yaml in
# parallel (one LAMMPS process per temperature; ntasks=8 covers all 8 by
# default). Step 2: samples frames from each trajectory at a fixed interval.
# Step 3: submits ONE VASP array job (sbatch) covering every sampled
# structure across all temperatures, then blocks until it finishes (polls
# squeue hourly).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

module load vasp/6.4.3 2>/dev/null || true
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

module load gcc/11.2.0 openmpi/4.1.2
module load openblas 2>/dev/null || true
export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"
export OMP_NUM_THREADS=1

echo "=== AIMD sampling SLURM job ==="
echo "  Node : $(hostname)"
echo "  Time : $(date)"

python data/aimd-sampling/run_sampling.py "$@"

echo "Done: $(date)"
