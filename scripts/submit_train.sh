#!/bin/bash
#SBATCH --job-name=mtp_train
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

source /usr/share/lmod/lmod/init/bash
export MODULEPATH="/appl/spack/v017/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v020/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v023/modulefiles/Core:/appl/modulefiles"
module load openblas

export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"

echo "=== MTP Training ==="
echo "  Node    : $(hostname)"
echo "  Time    : $(date)"

python src/train.py --config config/training.yaml

echo "Done."
