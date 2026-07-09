#!/bin/bash
#SBATCH --job-name=mtp_liquid
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/liquid_%j.out
#SBATCH --error=logs/liquid_%j.err
#
# Liquid Ge RDF via LAMMPS NPT→NVT + MTP.
#
# Usage:
#   sbatch scripts/submit_liquid.sh                                    # default pot
#   sbatch scripts/submit_liquid.sh results/potentials/my_pot.almtp    # custom pot
#   sbatch scripts/submit_liquid.sh results/potentials/my_pot.almtp --steps-npt 40000 --steps-nvt 20000
#   sbatch scripts/submit_liquid.sh results/potentials/my_pot.almtp --nx 6 --ny 6 --nz 6
#
# Protocol:  NPT equil (1500 K / 1 bar) → NVT production → RDF

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-2026-06-18.almtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -x "/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi" ]; then
    LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"
else
    echo "ERROR: lmp_mpi not found" >&2
    exit 1
fi

echo "=== Liquid Ge RDF (NPT→NVT) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $@"
date

# srun handles MPI parallelism via SLURM_NTASKS
python tests/liquid_rdf/liquid_rdf.py \
    --pot "$POT" \
    --lammps "srun $LAMMPS_BIN" \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/liquid_rdf 2>/dev/null || echo results/tests/liquid_rdf)"
date
