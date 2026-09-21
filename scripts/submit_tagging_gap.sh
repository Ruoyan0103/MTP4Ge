#!/bin/bash
#SBATCH --job-name=gap_tagging
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/gap_tagging_%j.out
#SBATCH --error=logs/gap_tagging_%j.err
#
# Re-evaluate a set of structures with the TurboGAP GAP potential
# (src/tagging_structs/GAP_settings.py, `turbogap predict`): single-point
# energy, forces, virial and stress for every config in the input extxyz
# file. Counterpart of the VASP single-point tagging step, against the GAP
# reference instead of DFT. The turbogap binary is cluster-specific and is
# supplied here, not hardcoded in GAP_settings.py. If the input carries DFT
# labels (free_energy/forces/virial/stress), a GAP-vs-DFT error report is
# also written to results/gap_tagging/<input_stem>/errors.txt, alongside
# that run's atoms_in.xyz/trajectory_out.xyz/turbogap.log.
#
# Usage:
#   sbatch scripts/submit_tagging_gap.sh <input.xyz> [outdir]
#   sbatch scripts/submit_tagging_gap.sh data/VASP_tagged/extxyz/train-tgap.xyz
#   sbatch scripts/submit_tagging_gap.sh data/VASP_tagged/extxyz/train-tgap.xyz data/GAP_tagged/extxyz

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

INPUT="${1:?Usage: sbatch scripts/submit_tagging_gap.sh <input.xyz> [outdir]}"
OUTDIR="${2:-data/GAP_tagged/extxyz}"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate

TURBOGAP_BIN="/projappl/project_2012355/CODE/TurboGAP_EPH/bin/turbogap"
if [ ! -x "$TURBOGAP_BIN" ]; then
    echo "ERROR: turbogap not found at $TURBOGAP_BIN" >&2
    exit 1
fi

echo "=== GAP Tagging (turbogap predict) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Input     : $INPUT"
echo "  Outdir    : $OUTDIR"
echo "  turbogap  : $TURBOGAP_BIN"
date

python src/tagging_structs/GAP_settings.py \
    --input "$INPUT" \
    --outdir "$OUTDIR" \
    --turbogap "srun $TURBOGAP_BIN"

echo "=== Done: $(date) ==="
