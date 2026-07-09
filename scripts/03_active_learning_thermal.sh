#!/usr/bin/env bash
# Active learning targeting both solid and liquid configurational space.
#
# Runs AL iterations with N parallel LAMMPS trajectories per iteration (defined
# in config/active_learning.yaml under the 'trajectories' key).  Each trajectory
# uses a distinct temperature and lattice parameter; liquid trajectories use the
# 3-step melt→cool→equil protocol (config/lammps/md_nvt_liquid.in).
#
# All N preselected.cfg files are merged before select_add runs MaxVol
# selection on the combined pool, so one DFT + retrain cycle covers all
# sampled conditions simultaneously.
#
# Usage:
#   bash scripts/03_active_learning_thermal.sh [--iter-per-temp N]
#   sbatch scripts/submit_active_learning_thermal.sh
#   Initial potential is read from config/active_learning.yaml (init_train_pot).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ITER=$(python3 -c "import yaml; print(yaml.safe_load(open('config/active_learning.yaml')).get('max_iterations', 5))")
POT_STEM=$(python3 -c "import yaml, pathlib; print(pathlib.Path(yaml.safe_load(open('config/active_learning.yaml')).get('init_train_pot', '')).stem)")
MAX_DFT_PER_ITER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iter-per-temp)    ITER="$2";           shift 2 ;;
        --max-dft-per-iter) MAX_DFT_PER_ITER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate 2>/dev/null || true

AL_POT="results/potentials/pot-$(date +%Y-%m-%d).almtp"

echo "=== Thermal + Liquid Active Learning (parallel trajectories) ==="
echo "  AL potential  : $AL_POT"
echo "  Pot stem      : $POT_STEM"
echo "  Iterations    : $ITER"
echo "  Trajectories  : defined in config/active_learning.yaml"
echo ""

python src/active_learning.py \
    --al-pot "$AL_POT" \
    --max-iter "$ITER" \
    --dft-dir "data/${POT_STEM}/thermal_dft" \
    --label "${POT_STEM}/thermal_parallel" \
    ${MAX_DFT_PER_ITER:+--max-dft-per-iter "$MAX_DFT_PER_ITER"}

echo ""
echo "=== AL complete ==="
echo "  Final potential : $AL_POT"
N_TRAIN=$(grep -c "^BEGIN_CFG" data/train-"$(date +%Y-%m-%d)".cfg 2>/dev/null || echo 0)
echo "  Training set    : $N_TRAIN configs"

echo ""
echo "=== Evaluating errors on final potential ==="
DATE=$(date +%Y-%m-%d)
bash scripts/02_test_errors.sh "$AL_POT" errors \
    "data/train-${DATE}.cfg" \
    "data/val-${DATE}.cfg" \
    "data/test-${DATE}.cfg"
