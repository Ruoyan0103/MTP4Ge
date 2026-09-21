#!/bin/bash
#SBATCH --job-name=vasp_aimd
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/aimd_%j.out
#SBATCH --error=logs/aimd_%j.err

# Run one VASP AIMD calculation (one temperature directory).
#
# Usage:
#   sbatch scripts/submit_aimd.sh data/aimd_vasp/T300K
#   sbatch scripts/submit_aimd.sh data/aimd_vasp/T600K
#   sbatch scripts/submit_aimd.sh data/aimd_vasp/T1000K

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

CALC_DIR="${1:?Usage: sbatch submit_aimd.sh <calc_dir>}"

echo "=== VASP AIMD ==="
echo "  Node     : $(hostname)"
echo "  Time     : $(date)"
echo "  Calc dir : $CALC_DIR"

module load vasp/6.3.2 2>/dev/null || true
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

cd "$CALC_DIR"

if [[ -f vasprun.xml ]] && grep -q "finalpos" vasprun.xml 2>/dev/null; then
    echo "  Already completed — skipping."
    exit 0
fi

srun vasp_std

echo "  Done: $(date)"
echo "  Collect snapshots with:"
echo "    python $REPO_ROOT/scripts/collect_aimd_snapshots.py --indir $(dirname $CALC_DIR) --out $REPO_ROOT/data/aimd.xyz"
