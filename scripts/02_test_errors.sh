#!/usr/bin/env bash
# Evaluate potential accuracy on train, val, and test sets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
MODE="${2:-errors}"   # errors | efs

echo "=== Error Evaluation (mode: $MODE) ==="

for SPLIT in train val test; do
    CFG="data/${SPLIT}.cfg"
    OUTDIR="results/errors/${SPLIT}"
    if [[ -f "$CFG" ]]; then
        echo "--- $SPLIT ---"
        python src/test_errors.py --pot "$POT" --cfg "$CFG" \
            --mode "$MODE" --outdir "$OUTDIR"
    else
        echo "  Skipping $SPLIT (no $CFG found)"
    fi
done

echo "Done. Reports in results/errors/"
