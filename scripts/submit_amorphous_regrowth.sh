#!/bin/bash
#SBATCH --job-name=mtp_regrowth
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000M
#SBATCH --output=logs/regrowth_%j.out
#SBATCH --error=logs/regrowth_%j.err
#
# Solid-phase epitaxial regrowth of a-Ge on Ge(001) (src/physical_validation/amorphous_regrowth.py).
#   1. prep    : slab nx×ny×nz (default 6×6×24 = 6912 atoms), central half melted/quenched,
#                all $SLURM_NTASKS ranks
#   2. anneal  : one independent NPT run per temperature, in parallel,
#                $SLURM_NTASKS / N_T ranks each (srun --exact)
#   3. analyze : velocities, Arrhenius fit -> regrowth_summary.txt, regrowth.png
#
# Usage:
#   sbatch scripts/submit_amorphous_regrowth.sh                                   # default pot, 800 900 1000 K
#   sbatch scripts/submit_amorphous_regrowth.sh results/potentials/pot_1019/bal.mtp
#   TEMPS="750 850 950" sbatch scripts/submit_amorphous_regrowth.sh POT --steps-anneal 1000000
#   SKIP_PREP=1 sbatch ...   # reuse an existing <outdir>/prep/a_c_slab.lammps
#   SW: MEM_PER_CPU=4500M sbatch -p medium --ntasks=128 --mem-per-cpu=1900M ... results/potentials/ref/sw_nordlund1998_nlh --backend lammps-sw
#
# Extra args are passed to every stage (e.g. --nx/--ny/--nz, --barostat z, --outdir, --delete-dump);
# they must be valid for amorphous_regrowth.py (the stage-specific ones are ignored by the others).
# Output: results/tests/<pot parent dir>/amorphous_regrowth/ (unless --outdir).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"
mkdir -p logs

POT="${1:-results/potentials/pot_1019/bal.mtp}"
shift || true
TEMPS="${TEMPS:-800 900 1000}"
MEM_PER_CPU="${MEM_PER_CPU:-1000M}"   # per LAMMPS rank (prep and anneals); SW (--backend lammps-sw) needs ~4500M

# --- Environment ---
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LAMMPS_BIN="/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
PY=src/physical_validation/amorphous_regrowth.py

NTASKS="${SLURM_NTASKS:-4}"
read -r -a T_ARR <<< "$TEMPS"
NT=${#T_ARR[@]}
NP_T=$(( NTASKS / NT ))
(( NP_T >= 1 )) || { echo "need at least one task per temperature"; exit 1; }

echo "=== a/c regrowth ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}   Node: $(hostname)"
echo "  Potential : $POT"
echo "  Temps     : $TEMPS  ($NP_T ranks each)"
echo "  Args      : $*"
date

if [[ -z "${SKIP_PREP:-}" ]]; then
    python $PY --pot "$POT" --stage prep --lammps "srun --exact --mem-per-cpu=$MEM_PER_CPU $LAMMPS_BIN" --np "$NTASKS" "$@"
fi

pids=()
for T in "${T_ARR[@]}"; do
    python $PY --pot "$POT" --stage anneal --T "$T" \
        --lammps "srun --exact --mem-per-cpu=$MEM_PER_CPU $LAMMPS_BIN" --np "$NP_T" "$@" &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || echo "WARNING: at least one anneal failed; analyzing what exists"

python $PY --pot "$POT" --stage analyze "$@"

echo "=== Done: $(date) ==="
