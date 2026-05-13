#!/usr/bin/env bash
# Convert train_liquid.xyz to MLIP-3 CFG.
#
# Usage:
#   bash scripts/00_convert.sh pool   [input.xyz]   # stratified seed + candidate pool (default)
#   bash scripts/00_convert.sh split  [input.xyz]   # random 80/10/10 split (for held-out test)
#
# Pool mode produces:
#   data/seed.cfg           — all bulk-type configs + 1 per other type; dimer excluded
#   data/candidate_pool.cfg — remaining labeled configs for active learning
#   data/train.cfg          — copy of seed.cfg (AL loop appends to this)
#
# Split mode produces:
#   data/train.cfg, data/val.cfg, data/test.cfg
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-pool}"
INPUT="${2:-/scratch/project_2012355/Paper_3/train_dataset/train_liquid.xyz}"
OUTDIR="data"

echo "=== Data Conversion (mode: $MODE) ==="
echo "  Input : $INPUT"
echo "  Output: $OUTDIR/"

if [[ "$MODE" == "pool" ]]; then
    python src/convert.py --input "$INPUT" --outdir "$OUTDIR" \
        --mode pool --seed-per-type 1 \
        --always-include-contains bulk \
        --always-include-contains-except distorted_bulk \
        --exclude dimer
    echo ""
    echo "Done."
    echo "  Seed set      : $OUTDIR/seed.cfg"
    echo "  Candidate pool: $OUTDIR/candidate_pool.cfg"
    echo "  Training start: $OUTDIR/train.cfg  (= seed.cfg; AL loop appends to this)"
elif [[ "$MODE" == "split" ]]; then
    python src/convert.py --input "$INPUT" --outdir "$OUTDIR" \
        --mode split --split 0.8 0.1 0.1
    echo "Done. CFG files written to $OUTDIR/"
else
    echo "ERROR: unknown mode '$MODE'. Use 'pool' or 'split'." >&2
    exit 1
fi
