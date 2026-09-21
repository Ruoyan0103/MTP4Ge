#!/bin/bash
#SBATCH --job-name=vasp_dft
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/vasp_%A_%a.out
#SBATCH --error=logs/vasp_%A_%a.err
#SBATCH --array=0-999%20   # overridden by submit command; %20 = max 20 concurrent

# Submit VASP single-point calculations for all dirs under $INDIR.
#
# Usage:
#   # Step 1 — generate the list of calc dirs:
#   python scripts/save/generate_strained_diamond.py
#   python scripts/save/generate_sia_configs.py
#
#   # Step 2 — build the dir list and submit array job:
#   find data/strained_dft data/sia_dft -name POSCAR -exec dirname {} \; | sort > data/vasp_dirs.txt
#   N=$(wc -l < data/vasp_dirs.txt)
#   sbatch --array=0-$((N-1))%20 scripts/submit_vasp_array.sh data/vasp_dirs.txt
#
#   # Step 3 — collect results after all jobs finish:
#   python scripts/save/collect_dft_results.py --indir data/strained_dft --out data/strained_diamond.xyz
#   python scripts/save/collect_dft_results.py --indir data/sia_dft      --out data/sia_configs.xyz

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

DIRLIST="${1:-data/vasp_dirs.txt}"

# Read the specific calculation directory for this array task
CALC_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DIRLIST")

if [[ -z "$CALC_DIR" ]]; then
    echo "ERROR: No directory for task $SLURM_ARRAY_TASK_ID in $DIRLIST"
    exit 1
fi

echo "=== VASP single-point ==="
echo "  Node     : $(hostname)"
echo "  Time     : $(date)"
echo "  Calc dir : $CALC_DIR"

# VASP environment — initialize Lmod then load VASP dependencies via env.sh
source /usr/share/lmod/lmod/init/bash
export MODULEPATH="/appl/spack/v017/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v020/modulefiles/linux-rhel8-x86_64/Core:/appl/spack/v023/modulefiles/Core:/appl/modulefiles"
source /appl/soft/phys/vasp/6.4.3/intel-2021.4.0/env.sh
VASP_BIN=/appl/soft/phys/vasp/6.4.3/intel-2021.4.0/bin/vasp_std
export VASP_PP_PATH="/appl/soft/phys/vasp/potpaw_PBE.64"

cd "$CALC_DIR"

# Skip if already successfully completed (check for closing </modeling> tag)
if [[ -f vasprun.xml ]] && grep -q "</modeling>" vasprun.xml 2>/dev/null; then
    echo "  Already completed — skipping."
    exit 0
fi

srun "$VASP_BIN"

echo "  Done: $(date)"
