#!/usr/bin/env bash
# Run active learning loop (follows MLIP-2 tutorial-2 workflow).
#
# Each iteration:
#   A. calculate_grade  — initialize/update active set in pot.almtp
#   B. LAMMPS MD        — run MD with MLIP selection; extrapolative configs
#                         written to results/active_learning/iter_NNN/preselected.cfg
#   C. select_add       — pick most informative subset → data/selected.cfg
#   D/E. VASP DFT       — single-point DFT on selected structures (ASE Vasp)
#                         results written to data/labelled.cfg
#   F. retrain          — 80 % of labeled configs added to train.cfg, retrain
#                         10 % appended to data/val.cfg, 10 % to data/test.cfg
#   A. calculate_grade  — update active set with newly labeled configs
#
# Prerequisites:
#   - data/train_diamond.cfg must exist (run scripts/00_convert.sh first)
#   - results/potentials/pot.almtp must exist (run scripts/01_train.sh first)
#   - lmp (LAMMPS with MLIP pair style) in PATH
#   - vasp_std (or configured vasp_command) in PATH
#   - VASP_PP_PATH environment variable set for ASE Vasp
#
# Usage:
#   bash scripts/03_active_learning.sh [pot.almtp] [--max-iter N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot.almtp}"
shift || true   # remaining args passed through to python

echo "=== Active Learning ==="
echo "  Potential   : $POT"
echo "  Training set: data/train_diamond.cfg"
echo "  Selected    : data/selected.cfg  (written each iteration)"
echo "  Labelled    : data/labelled.cfg  (DFT results each iteration)"
echo "  Val/Test    : data/val.cfg, data/test.cfg  (10 % each, accumulated)"

python src/active_learning.py --pot "$POT" "$@"
