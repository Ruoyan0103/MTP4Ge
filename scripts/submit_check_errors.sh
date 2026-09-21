#!/bin/bash
#SBATCH --job-name=mtp_check_errors_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/check_errors_lammps_%j.out
#SBATCH --error=logs/check_errors_lammps_%j.err
#
# Evaluate MTP potential accuracy (src/utils/check_errors.py --backend lammps)
# via LAMMPS (pair_style hybrid/overlay mtp nlh, config/lammps/phonon_dispersion_mtp.in)
# instead of `mlp check_errors` — for potentials trained with a radial basis
# type `mlp` can't load ("Wrong radial basis type"). Runs one LAMMPS
# single-point evaluation per config in each dataset given; reports energy,
# force, and stress errors (stress is derived from LAMMPS's pressure tensor
# after the same single-point run). The LAMMPS binary and input template are
# cluster-specific and are supplied here, not hardcoded in check_errors.py.
#
# Usage:
#   sbatch scripts/submit_check_errors.sh <pot.almtp> [cfg1] [cfg2] ...
#   sbatch scripts/submit_check_errors.sh results/potentials/20.mtp                       # defaults to train/val/test
#   sbatch scripts/submit_check_errors.sh results/potentials/20.mtp data/cfg/val.cfg data/cfg/test.cfg
#   sbatch scripts/submit_check_errors.sh results/potentials/20.mtp data/cfg/test.cfg     # a single dataset
#
# Any number of CFG files, in any order, are accepted — one report is
# written per file to results/errors/<pot_stem>/<cfg_stem>.txt.

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-init.mtp}"
shift || true
if [[ $# -eq 0 ]]; then
    set -- data/train.cfg data/val.cfg data/test.cfg
fi

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
LAMMPS_TEMPLATE="config/lammps/phonon_dispersion_mtp.in"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Error Evaluation (LAMMPS, pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Template  : $LAMMPS_TEMPLATE"
echo "  Datasets  : $*"
date

for CFG in "$@"; do
    if [[ -f "$CFG" ]] && grep -q "BEGIN_CFG" "$CFG"; then
        echo "--- $CFG ---"
        python src/utils/check_errors.py --pot "$POT" --cfg "$CFG" \
            --backend lammps \
            --lammps "srun $LAMMPS_BIN" \
            --lammps-template "$LAMMPS_TEMPLATE"
    else
        echo "  Skipping $CFG (not found or no configs)"
    fi
done

echo "=== Done: $(date) ==="
