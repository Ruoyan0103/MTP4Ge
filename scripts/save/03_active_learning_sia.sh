#!/usr/bin/env bash
# Active learning targeting SIA and strained-bulk configurational space.
#
# Runs AL iterations sequentially for each target structure type:
#   1. Vacancy           (data/vac.data          — existing)
#   2. Tetrahedral SIA   (data/sia/tet.lammps)
#   3. Hexagonal SIA     (data/sia/hex.lammps)
#   4. <110> dumbbell    (data/sia/split.lammps)
#   5. Bond-centre SIA   (data/sia/bond.lammps)
#   6. Compressed bulk   (data/strained/vol_m02pct.lammps)
#   7. Expanded bulk     (data/strained/vol_p02pct.lammps)
#
# Each structure type runs for --iter-per-structure iterations (default 5).
# The same potential accumulates DFT data across all types.
#
# Prerequisites:
#   python scripts/save/generate_sia_lammps.py   # creates data/sia/*.lammps
#
# Usage:
#   bash scripts/save/03_active_learning_sia.sh [--iter-per-structure N]
#   sbatch scripts/submit_active_learning_sia.sh
#   Initial potential is read from config/active_learning.yaml (init_train_pot).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

POT=$(python3 -c "import yaml; print(yaml.safe_load(open('config/active_learning.yaml')).get('init_train_pot', 'results/potentials/pot-aimd-init.almtp'))")
ITER_PER_STRUCT=5   # default iterations per structure type

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iter-per-structure) ITER_PER_STRUCT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate 2>/dev/null || true

echo "=== SIA-targeted Active Learning ==="
echo "  Potential           : $POT"
echo "  Iterations per type : $ITER_PER_STRUCT"
echo ""

# Step 0: generate SIA and strained LAMMPS structures if not already present
if [[ ! -f data/sia/tet.lammps ]]; then
    echo "--- Generating SIA LAMMPS structures ---"
    python scripts/save/generate_sia_lammps.py
fi

# Target structures: label → LAMMPS data file
declare -A STRUCTURES
STRUCTURES["vacancy"]="data/vac.data"
STRUCTURES["tet"]="data/sia/tet.lammps"
STRUCTURES["hex"]="data/sia/hex.lammps"
STRUCTURES["split"]="data/sia/split.lammps"
STRUCTURES["bond"]="data/sia/bond.lammps"
STRUCTURES["vol_compressed"]="data/strained/vol_m02pct.lammps"
STRUCTURES["vol_expanded"]="data/strained/vol_p02pct.lammps"

# Ordered list (bash 3 compatible)
ORDER=(vacancy tet hex split bond vol_compressed vol_expanded)

# All structure types evolve the same AL potential so training accumulates.
AL_POT="results/potentials/pot_al.almtp"

for LABEL in "${ORDER[@]}"; do
    STRUCT="${STRUCTURES[$LABEL]}"
    if [[ ! -f "$STRUCT" ]]; then
        echo "WARNING: $STRUCT not found — skipping $LABEL"
        continue
    fi

    echo "========================================"
    echo "Structure: $LABEL  ($STRUCT)"
    echo "========================================"

    python src/active_learning.py \
        --pot "$POT" \
        --al-pot "$AL_POT" \
        --max-iter "$ITER_PER_STRUCT" \
        --structure "$STRUCT" \
        --dft-dir data/defect_dft \
        --label "sia_${LABEL}"

    # After the first structure type, the AL pot is the new base for all subsequent ones.
    POT="$AL_POT"

    echo ""
    echo "  Done with $LABEL. Training set size:"
    grep -c "^BEGIN_CFG" data/train-"$(date +%Y-%m-%d)".cfg 2>/dev/null || echo 0
    echo ""
done

echo "=== SIA-targeted AL complete ==="
echo "  Final potential : $POT"
N_TRAIN=$(grep -c "^BEGIN_CFG" data/train-"$(date +%Y-%m-%d)".cfg 2>/dev/null || echo 0)
echo "  Training set    : $N_TRAIN configs"

echo ""
echo "=== Evaluating errors on final potential ==="
TRAIN_CFG="data/train-$(date +%Y-%m-%d).cfg"
bash scripts/save/02_test_errors.sh "$AL_POT" errors "$TRAIN_CFG"
