#!/bin/bash
# KPAR/NCORE benchmark for 64-atom VASP single points on 64 MPI tasks.
#
# Usage (from the repo root, with plain bash — NOT sbatch):
#   bash scripts/submit_vasp_parallel_test.sh <input.cfg> <outdir>
#   e.g.
#   bash scripts/submit_vasp_parallel_test.sh \
#       results/potentials/pot_660277/AL-1/iter_1/new_added.cfg \
#       results/potentials/pot_660277/AL-1/parallel_test
#
# Writes <outdir>/struct_1..5 from the first 5 CFG blocks (via
# src/tagging_structs/VASP_settings.py), each with a different KPAR/NCORE
# pair from COMBOS below, then submits them as a 5-task array using the
# run stage of scripts/submit_vasp_cfg_array.sh (same #SBATCH resources,
# modules and `srun vasp_std` as production).
#
# The structures differ, so compare the average time per SCF step (LOOP
# lines in OUTCAR), not total run time:
#   for d in <outdir>/struct_*; do
#     echo "$d $(grep -E '^ *(KPAR|NCORE)' $d/INCAR | tr '\n' ' ')" \
#          "$(grep 'LOOP:' $d/OUTCAR | awk '{s+=$NF; n++} END {printf "%d steps, %.2f s/step", n, s/n}')"
#   done

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

INPUT_CFG="${1:?usage: bash $0 <input.cfg> <outdir>}"
OUTDIR="${2:?usage: bash $0 <input.cfg> <outdir>}"

# "KPAR NCORE" per struct_N, in order. Round 1 (4/2, 4/4, 4/8, 2/8, 8/2;
# s per SCF step over steps 1-3: 149, 89, 75, 78, 143) showed NCORE >= 8
# wins, so round 2 probes that region (NCORE must divide 64/KPAR).
COMBOS=("4 8" "4 16" "2 16" "2 32" "8 8")
N=${#COMBOS[@]}

if compgen -G "$OUTDIR/struct_*" > /dev/null; then
    echo "ERROR: $OUTDIR already contains struct_* folders; use a fresh outdir." >&2
    exit 1
fi
mkdir -p "$OUTDIR" logs

# First N CFG blocks only
SUBSET_CFG="$OUTDIR/first_${N}.cfg"
awk -v n="$N" '{print} /END_CFG/ {if (++k == n) exit}' "$INPUT_CFG" > "$SUBSET_CFG"

export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"
python src/tagging_structs/VASP_settings.py --input "$SUBSET_CFG" --outdir "$OUTDIR"

for ((i = 1; i <= N; i++)); do
    read -r kpar ncore <<< "${COMBOS[i-1]}"
    incar="$OUTDIR/struct_$i/INCAR"
    sed -i -E "s/^KPAR = .*/KPAR = $kpar/; s/^NCORE = .*/NCORE = $ncore/" "$incar"
    echo "struct_$i: $(grep -E '^(KPAR|NCORE)' "$incar" | tr '\n' ' ')"
done

OUTDIR_ABS="$(cd "$OUTDIR" && pwd)"
sbatch --job-name=vasp_partest --array=1-"$N" \
    --export=ALL,VASP_OUTDIR="$OUTDIR_ABS" \
    scripts/submit_vasp_cfg_array.sh
