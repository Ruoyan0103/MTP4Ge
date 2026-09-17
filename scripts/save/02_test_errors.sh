#!/usr/bin/env bash
# Evaluate potential accuracy on train, val, and test sets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
MODE="${2:-errors}"   # errors | efs
TRAIN_CFG="${3:-data/train.cfg}"
VAL_CFG="${4:-data/val.cfg}"
TEST_CFG="${5:-data/test.cfg}"

echo "=== Error Evaluation (mode: $MODE) ==="

for SPLIT in train val test; do
    if [[ "$SPLIT" == "train" ]]; then
        CFG="$TRAIN_CFG"
    elif [[ "$SPLIT" == "val" ]]; then
        CFG="$VAL_CFG"
    else
        CFG="$TEST_CFG"
    fi
    OUTDIR="results/errors/${SPLIT}"
    if [[ -f "$CFG" ]] && grep -q "BEGIN_CFG" "$CFG" 2>/dev/null; then
        echo "--- $SPLIT ---"
        python src/test_errors.py --pot "$POT" --cfg "$CFG" \
            --mode "$MODE" --outdir "$OUTDIR"
    else
        echo "  Skipping $SPLIT (no configs in $CFG)"
    fi
done

echo "Done. Reports in results/errors/"
