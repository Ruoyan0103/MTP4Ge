#!/bin/bash
#SBATCH --job-name=mtp_liquid_nlh
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/liquid_nlh_%j.out
#SBATCH --error=logs/liquid_nlh_%j.err
#
# Liquid Ge RDF via LAMMPS NPT→NVT + MTP, using pair_style hybrid/overlay
# mtp nlh (src/physical_validation/liquid_rdf.py --backend lammps-nlh) — for
# potentials whose radial basis type `pair_style mlip` (--backend
# lammps-mlip, the default, see submit_liquid_mlip.sh) cannot load
# ("Wrong radial basis type").
#
# Usage:
#   sbatch scripts/submit_liquid_nlh.sh                                    # default pot
#   sbatch scripts/submit_liquid_nlh.sh results/potentials/my_pot.almtp    # custom pot
#   sbatch scripts/submit_liquid_nlh.sh results/potentials/my_pot.almtp --steps-npt 40000 --steps-nvt 20000
#   sbatch scripts/submit_liquid_nlh.sh results/potentials/my_pot.almtp --nx 6 --ny 6 --nz 6
#
# Protocol:  NVT pre-heat → NPT equil (0 bar) → NVT production → RDF

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-results/potentials/20.mtp}"
shift || true

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Liquid Ge RDF (NVT->NPT->NVT, LAMMPS pair_style mtp+nlh) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
echo "  Extra args: $@"
date

# srun handles MPI parallelism via SLURM_NTASKS
python src/physical_validation/liquid_rdf.py \
    --pot "$POT" \
    --backend lammps-nlh \
    --lammps "srun $LAMMPS_BIN" \
    --outdir results/tests/liquid_rdf_nlh \
    "$@"

echo "=== Done ==="
echo "  Output : $(realpath results/tests/liquid_rdf_nlh 2>/dev/null || echo results/tests/liquid_rdf_nlh)"
date
