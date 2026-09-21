#!/usr/bin/env bash
# Evaluate potential accuracy against one or more reference datasets.
# Reports land in results/errors/<pot_stem>/<cfg_stem>.txt (see
# src/utils/check_errors.py).
#
# Usage:
#   bash scripts/save/02_test_errors.sh <pot.mtp> [cfg1] [cfg2] ...
#   bash scripts/save/02_test_errors.sh results/potentials/pot.almtp data/val.cfg data/test.cfg
#   bash scripts/save/02_test_errors.sh results/potentials/pot.almtp   # defaults to train/val/test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
shift || true
if [[ $# -eq 0 ]]; then
    set -- data/train.cfg data/val.cfg data/test.cfg
fi

echo "=== Error Evaluation ==="

for CFG in "$@"; do
    if [[ -f "$CFG" ]] && grep -q "BEGIN_CFG" "$CFG" 2>/dev/null; then
        echo "--- $CFG ---"
        python src/utils/check_errors.py --pot "$POT" --cfg "$CFG"
    else
        echo "  Skipping $CFG (not found or no configs)"
    fi
done

echo "Done. Reports in results/errors/"
