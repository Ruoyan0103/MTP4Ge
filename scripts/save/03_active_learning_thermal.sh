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
#   bash scripts/save/03_active_learning_thermal.sh [--iter-per-temp N]
#   sbatch scripts/submit_active_learning_thermal.sh
#   Initial potential is read from config/active_learning.yaml (init_train_pot).
#
#   To resume a run that was interrupted (e.g. hit the SLURM time limit) on a
#   later day, pass --resume-pot <path>. This pins AL_POT (and the dated
#   train/test CFGs, which must already exist for that same date) to the
#   in-progress potential instead of deriving a fresh one from today's date,
#   which would otherwise silently restart from init_train_pot and lose all
#   accumulated AL training data:
#     bash scripts/save/03_active_learning_thermal.sh --resume-pot results/potentials/pot-2026-08-05.almtp

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ITER=$(python3 -c "import yaml; print(yaml.safe_load(open('config/active_learning.yaml')).get('max_iterations', 5))")
POT_STEM=$(python3 -c "import yaml, pathlib; print(pathlib.Path(yaml.safe_load(open('config/active_learning.yaml')).get('init_train_pot', '')).stem)")
MAX_DFT_PER_ITER=""
RESUME_POT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iter-per-temp)    ITER="$2";           shift 2 ;;
        --max-dft-per-iter) MAX_DFT_PER_ITER="$2"; shift 2 ;;
        --resume-pot)       RESUME_POT="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate 2>/dev/null || true

TODAY="$(date +%Y-%m-%d)"
POT_ARGS=()

if [[ -n "$RESUME_POT" ]]; then
    AL_POT="$RESUME_POT"
    POT_ARGS=(--pot "$AL_POT")
    # run_active_learning() always names the training/test CFGs after *today's*
    # date, regardless of --pot/--al-pot. If this resume crosses a day boundary,
    # copy the resumed run's dated train/test CFGs forward to today's filenames
    # so the loop keeps accumulating on the real dataset instead of falling back
    # to init_train_cfg/init_test_cfg.
    RESUME_DATE="$(basename "$RESUME_POT" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')"
    if [[ -n "$RESUME_DATE" && "$RESUME_DATE" != "$TODAY" ]]; then
        for kind in train test; do
            src="data/${kind}-${RESUME_DATE}.cfg"
            dst="data/${kind}-${TODAY}.cfg"
            if [[ -f "$src" && ! -f "$dst" ]]; then
                cp "$src" "$dst"
                echo "  Carried forward $src -> $dst ($(grep -c '^BEGIN_CFG' "$dst") configs)"
            fi
        done
    fi
else
    AL_POT="results/potentials/pot-${TODAY}.almtp"
fi

echo "=== Thermal + Liquid Active Learning (parallel trajectories) ==="
echo "  AL potential  : $AL_POT"
echo "  Pot stem      : $POT_STEM"
echo "  Iterations    : $ITER"
echo "  Trajectories  : defined in config/active_learning.yaml"
echo ""

python src/active_learning.py \
    "${POT_ARGS[@]}" \
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
bash scripts/save/02_test_errors.sh "$AL_POT" \
    "data/train-${DATE}.cfg" \
    "data/val-${DATE}.cfg" \
    "data/test-${DATE}.cfg"
