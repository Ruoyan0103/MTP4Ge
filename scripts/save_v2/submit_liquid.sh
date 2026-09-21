#!/bin/bash
#SBATCH --job-name=mtp_liquid
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=5
#SBATCH --ntasks=120
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/liquid_%j.out
#SBATCH --error=logs/liquid_%j.err
#
# Liquid Ge RDF via LAMMPS + MTP. Pair style is selected with --backend
# (see src/physical_validation/liquid_rdf.py):
#   mlip - pair_style mlip (old LAMMPS-MLIP build). Protocol: NPT equil
#          (1500 K / 1 bar) -> NVT production -> RDF.
#   nlh  - pair_style hybrid/overlay mtp nlh (new LAMMPS-MTP build), for
#          potentials whose radial basis type `pair_style mlip` cannot load
#          ("Wrong radial basis type"). Protocol: NVT pre-heat -> NPT equil
#          (0 bar) -> NVT production -> RDF.
#
# The #SBATCH allocation above is sized for the heavier nlh case; mlip runs
# finish well within it.
#
# Usage:
#   sbatch scripts/submit_liquid.sh --backend mlip                                   # default pot
#   sbatch scripts/submit_liquid.sh --backend mlip results/potentials/my_pot.almtp   # custom pot
#   sbatch scripts/submit_liquid.sh --backend nlh                                    # default pot
#   sbatch scripts/submit_liquid.sh --backend nlh results/potentials/my_pot.almtp --steps-npt 40000 --steps-nvt 20000

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

if [ "${1:-}" != "--backend" ] || { [ "${2:-}" != "mlip" ] && [ "${2:-}" != "nlh" ]; }; then
    echo "ERROR: first argument must be --backend mlip|nlh" >&2
    exit 1
fi
BACKEND="$2"
shift 2

EXTRA_OUTDIR_ARGS=()
case "$BACKEND" in
    mlip)
        POT="${1:-/scratch/project_2012355/Paper_3/00-subsets/07-short_range/pot24_08_26.almtp}"
        LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi"
        LIQUID_BACKEND_FLAG="lammps-mlip"
        # No --outdir override: liquid_rdf.py derives results/tests/<pot_dir>/liquid_rdf
        ;;
    nlh)
        POT="${1:-results/potentials/20.mtp}"
        LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
        LIQUID_BACKEND_FLAG="lammps-nlh"
        EXTRA_OUTDIR_ARGS=(--outdir results/tests/liquid_rdf_nlh)
        ;;
esac
shift || true

if [ ! -x "$LAMMPS_BIN" ]; then
    echo "ERROR: lmp_mpi not found at $LAMMPS_BIN" >&2
    exit 1
fi

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== Liquid Ge RDF (backend: $BACKEND) ==="
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
    --backend "$LIQUID_BACKEND_FLAG" \
    --lammps "srun $LAMMPS_BIN" \
    "${EXTRA_OUTDIR_ARGS[@]}" \
    "$@"

echo "=== Done ==="
date
