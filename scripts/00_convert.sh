#!/usr/bin/env bash
# Convert train_liquid.xyz → MLIP-3 CFG (train/val/test split)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT="${1:-/scratch/project_2012355/Paper_3/train_dataset/train_liquid.xyz}"
OUTDIR="${2:-data}"
SPLIT="${3:-0.8 0.1 0.1}"

echo "=== Data Conversion ==="
echo "  Input : $INPUT"
echo "  Output: $OUTDIR/"
echo "  Split : $SPLIT"

python src/convert.py --input "$INPUT" --outdir "$OUTDIR" --split $SPLIT

echo "Done. CFG files written to $OUTDIR/"
