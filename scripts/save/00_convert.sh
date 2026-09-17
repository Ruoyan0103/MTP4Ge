#!/usr/bin/env bash
# Convert train.xyz to MLIP-3 CFG format.
#
# Produces:
#   data/train.cfg — all DFT-labeled configs
#
# Usage:
#   bash scripts/save/00_convert.sh [input.xyz]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT="${1:-/scratch/project_2012355/Paper_3/MTP4Ge/data/val.xyz}"
OUTDIR="data"

echo "=== Data Conversion ==="
echo "  Input : $INPUT"
OUTNAME="$(basename "$INPUT" .xyz).cfg"
echo "  Output: $OUTDIR/$OUTNAME"

python src/utils/convert.py --from xyz --to cfg --input "$INPUT" --outdir "$OUTDIR"

echo ""
echo "Done."
echo "  Training data: $OUTDIR/$OUTNAME"
