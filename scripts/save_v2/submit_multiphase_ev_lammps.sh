#!/bin/bash
#SBATCH --job-name=mtp_multiphase_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/multiphase_lammps_%j.out
#SBATCH --error=logs/multiphase_lammps_%j.err
#
# Multi-phase E-V curves (8 Ge phases) via LAMMPS
# (src/physical_validation/multiphase_ev.py --backend lammps) — for potentials whose radial
# basis type `mlp calculate_efs` cannot load ("Wrong radial basis type"); see
# config/lammps/energy_volume.in.
#
# Usage:
#   sbatch scripts/submit_multiphase_ev_lammps.sh                                  # default pot
#   sbatch scripts/submit_multiphase_ev_lammps.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_multiphase_ev_lammps.sh results/potentials/my_pot.almtp --n-volumes 15

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/bal_prune_val.mtp}"
shift || true

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Multi-phase E-V curves (LAMMPS, pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $*"
date

python src/physical_validation/multiphase_ev.py \
    --pot "$POT" \
    --backend lammps \
    --lammps "srun $LAMMPS_BIN" \
    --outdir results/tests/multiphase_ev_lammps \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/multiphase_ev_lammps 2>/dev/null || echo results/tests/multiphase_ev_lammps)"
date
