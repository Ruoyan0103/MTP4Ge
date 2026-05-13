#!/usr/bin/env bash
# Run active learning loop.
#
# Auto mode (default): candidate pool is already DFT-labeled.
#   select_add → merge → retrain, no human pause.
#   Requires: data/candidate_pool.cfg  (from scripts/00_convert.sh pool)
#
# Interactive mode (--no-auto): pauses each iteration for DFT labeling.
#
# Usage:
#   bash scripts/03_active_learning.sh [pot.almtp] [--no-auto] [--max-iter N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
shift || true   # remaining args passed through to python

echo "=== Active Learning ==="
echo "  Potential: $POT"
echo "  Pool     : data/candidate_pool.cfg"

python src/active_learning.py --pot "$POT" "$@"
