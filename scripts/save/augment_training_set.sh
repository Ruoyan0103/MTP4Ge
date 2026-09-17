#!/usr/bin/env bash
# Append new DFT-labeled XYZ files to the training set (train/val/test CFGs).
#
# Converts each XYZ file to CFG, then merges into data/train.cfg at 80/10/10
# split using scripts/save/split_test.py (which handles train/val/test assignment).
#
# Usage:
#   bash scripts/save/augment_training_set.sh data/strained_diamond.xyz [data/sia_configs.xyz ...]
#
# After running, retrain:
#   sbatch scripts/submit_train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <file1.xyz> [file2.xyz ...]"
    exit 1
fi

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate 2>/dev/null || true

for XYZ in "$@"; do
    if [[ ! -f "$XYZ" ]]; then
        echo "WARNING: $XYZ not found — skipping"
        continue
    fi

    STEM="$(basename "$XYZ" .xyz)"
    CFG="data/${STEM}.cfg"

    echo "=== Converting $XYZ → $CFG ==="
    python src/utils/convert.py --from xyz --to cfg --input "$XYZ" --outdir data/

    echo "=== Splitting $CFG into train/val/test (80/10/10) ==="
    python - "$CFG" data/train.cfg data/val.cfg data/test.cfg <<'PYEOF'
import re, random, sys
from pathlib import Path

cfg_path, train_path, val_path, test_path = (Path(a) for a in sys.argv[1:5])

blocks = re.split(r"(?=BEGIN_CFG\b)", cfg_path.read_text())
blocks = [b.strip() for b in blocks if b.strip().startswith("BEGIN_CFG")]
if not blocks:
    print(f"  No configs found in {cfg_path} — skipping split.")
    sys.exit(0)

rng = random.Random(42)
rng.shuffle(blocks)
n = len(blocks)
n_test = max(1, n // 10)
n_val  = max(1, n // 10)
test_blocks  = blocks[:n_test]
val_blocks   = blocks[n_test:n_test + n_val]
train_blocks = blocks[n_test + n_val:]

def append_blocks(path, blks):
    if not blks:
        return
    with open(path, "a") as f:
        f.write("\n\n".join(blks) + "\n")

append_blocks(train_path, train_blocks)
append_blocks(val_path,   val_blocks)
append_blocks(test_path,  test_blocks)
print(f"  Split {n} configs: {len(train_blocks)} train | {len(val_blocks)} val | {len(test_blocks)} test")
PYEOF

    echo "  Done: $STEM"
    echo ""
done

echo "=== Training set sizes ==="
for f in data/train.cfg data/val.cfg data/test.cfg; do
    N=$(grep -c "^BEGIN_CFG" "$f" 2>/dev/null || echo 0)
    printf "  %-20s : %d configs\n" "$f" "$N"
done

echo ""
echo "Next: sbatch scripts/submit_train.sh"
