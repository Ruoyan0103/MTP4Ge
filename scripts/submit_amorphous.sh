#!/bin/bash
#SBATCH --job-name=mtp_amorphous
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/amorphous_%j.out
#SBATCH --error=logs/amorphous_%j.err
#
# Melt-quench amorphous Ge RDF via LAMMPS + MTP.
#
# Usage:
#   sbatch scripts/submit_amorphous.sh                              # default pot
#   sbatch scripts/submit_amorphous.sh results/potentials/my_pot.almtp
#   sbatch scripts/submit_amorphous.sh results/potentials/my_pot.almtp --T-melt 2000 --quench-rate 1e12
#
# Quick test (faster quench, ~15 min):
#   sbatch scripts/submit_amorphous.sh results/potentials/my_pot.almtp --quick

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-/scratch/project_2012355/Paper_3/MTP4Ge/results/potentials/pot_1028/28vib.mtp}"
shift || true

# --- Environment ---
module load gcc/15.2.0 openmpi/5.0.10   # lmp_mpi is built against these
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1

echo "=== Amorphous Ge RDF ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  Args      : $@"
date

#LAMMPS_BIN=$(which lmp_mpi 2>/dev/null || echo "/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi")
LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
# Use srun directly (--lammps as srun + binary, --np 1 avoids mpirun wrapper)
python src/physical_validation/amorphous_rdf.py \
    --pot "$POT" \
    --lammps "srun $LAMMPS_BIN" \
    --np 1 \
    --backend lammps-nlh \
    --quick \
    "$@"

echo "=== Done ==="
echo "  Output : results/tests/<pot parent dir>/amorphous_rdf/ (unless --outdir)"
date
