#!/bin/bash
#SBATCH --job-name=mtp_errors
#SBATCH --account=project_2001625
#SBATCH --partition=test
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/errors_%j.out
#SBATCH --error=logs/errors_%j.err

# Evaluate MTP errors on train/val/test sets.
#
# Usage:
#   sbatch scripts/submit_test_errors.sh <pot.almtp> [date]
#
# date defaults to today (YYYY-MM-DD). Pass the run date if evaluating
# a potential from a previous day, e.g.:
#   sbatch scripts/submit_test_errors.sh results/potentials/pot-2026-06-04.almtp 2026-06-04

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-init.mtp}"
DATE="${2:-$(date +%Y-%m-%d)}"

source /usr/share/lmod/lmod/init/bash
export MODULEPATH="/appl/spack/v017/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v020/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v023/modulefiles/Core:/appl/modulefiles"
module load openblas

export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"

echo "=== Error Evaluation ==="
echo "  Node : $(hostname)"
echo "  Time : $(date)"
echo "  Pot  : $POT"
echo "  Date : $DATE"

bash scripts/save/02_test_errors.sh "$POT" \
    "data/train-${DATE}.cfg" \
    "data/val-${DATE}.cfg" \
    "data/test-${DATE}.cfg"

echo "Done: $(date)"