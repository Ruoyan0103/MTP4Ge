#!/usr/bin/env bash
# Run active learning loop to iteratively refine the potential.
# Requires a preselected.cfg of candidate structures in data/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
MAX_ITER="${2:-}"

echo "=== Active Learning ==="
echo "  Potential: $POT"
echo "  Preselected candidates: data/preselected.cfg"

ARGS="--pot $POT"
[[ -n "$MAX_ITER" ]] && ARGS="$ARGS --max-iter $MAX_ITER"

python src/active_learning.py $ARGS

echo "Done. Updated potential: $POT"
