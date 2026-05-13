#!/usr/bin/env bash
# Train MTP potential on the prepared training data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-config/training.yaml}"
SEED="${2:-}"

echo "=== MTP Training ==="
echo "  Config: $CONFIG"

ARGS="--config $CONFIG"
[[ -n "$SEED" ]] && ARGS="$ARGS --seed $SEED"

python src/train.py $ARGS

echo "Done. Potential saved to results/potentials/pot.almtp"
