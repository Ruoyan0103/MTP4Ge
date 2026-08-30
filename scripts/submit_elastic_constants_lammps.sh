#!/bin/bash
#SBATCH --job-name=mtp_elastic_lammps
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/elastic_lammps_%j.out
#SBATCH --error=logs/elastic_lammps_%j.err
#
# Relaxed-ion elastic constants via LAMMPS (tests/elastic_constant/in.elastic).
# Unlike tests/elastic_constants.py's default MTP path, this minimizes atomic
# positions at each fixed strain before reading the stress.
#
# Usage:
#   sbatch scripts/submit_elastic_constants_lammps.sh                                  # default pot
#   sbatch scripts/submit_elastic_constants_lammps.sh results/potentials/my_pot.almtp

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

POT="${1:-/scratch/project_2012355/Paper_3/00-subsets/07-short_range/pot24_08_26.almtp}"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"
if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

echo "=== Elastic constants (LAMMPS, relaxed-ion) ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Tasks     : ${SLURM_NTASKS:-1}"
echo "  Potential : $POT"
echo "  LAMMPS    : $LAMMPS_BIN"
date

python tests/elastic_constants.py \
    --pot "$POT" \
    --method lammps \
    --lammps "srun $LAMMPS_BIN"

echo "=== Done ==="
date
