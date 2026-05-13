#!/usr/bin/env bash
# Run the full MTP pruning pipeline:
#   1. extract_problem  — compute XᵀWX/XᵀWy matrices
#   2. prune            — NSGA-II optimisation (produces Pareto front CSV)
#   3. mask_inherited   — apply selected Pareto row → pruned potential
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
ROW="${2:-0}"    # Pareto front row: 0 = most accurate, higher = sparser

echo "=== MTP Pruning Pipeline ==="
echo "  Full potential: $POT"
echo "  Pareto row    : $ROW"

python src/prune.py --pot "$POT" --row "$ROW"

echo "Done. Pruned potential: results/potentials/pruned.almtp"
