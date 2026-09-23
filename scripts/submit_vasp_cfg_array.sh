#!/bin/bash
#SBATCH --job-name=vasp_cfg
#SBATCH --account=project_2012355
#SBATCH --partition=medium
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
# No --mem*: 'medium' is node-exclusive, so --mem-per-cpu is multiplied by
# all 384 cores (2G -> 768G > the ~745G node, "Requested node configuration
# is not available"); the job already gets the whole node's memory by default.
#SBATCH --output=logs/vasp_cfg_%A_%a.out
#SBATCH --error=logs/vasp_cfg_%A_%a.err

# VASP single points for every structure in a CFG file, one array task each.
#
# Usage (from the repo root, with plain bash — NOT sbatch):
#   bash scripts/submit_vasp_cfg_array.sh <input.cfg> <outdir> [max_concurrent]
#   e.g.
#   bash scripts/submit_vasp_cfg_array.sh \
#       results/potentials/pot_660277/AL-1/iter_1/new_added.cfg \
#       results/potentials/pot_660277/AL-1/iter_1/vasp
#
# Stage 1 (bash, login node): writes <outdir>/struct_N/{POSCAR,INCAR,POTCAR}
#   via src/tagging_structs/VASP_settings.py (KPAR=4, NCORE=16 for 64 tasks:
#   fastest per SCF step in scripts/submit_vasp_parallel_test.sh benchmarks),
#   then submits this same file as `sbatch --array=...`, skipping structures
#   whose OUTCAR already finished (so re-running resubmits only failures).
# Stage 2 (array task): runs `srun vasp_std` inside <outdir>/struct_$SLURM_ARRAY_TASK_ID.
# [max_concurrent] caps simultaneously running array tasks (sbatch --array=...%N).

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

_is_done() {  # $1 = struct dir
    [[ -f "$1/OUTCAR" ]] && grep -q "General timing" "$1/OUTCAR"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # ---- Stage 1: prepare folders and submit the array ----
    INPUT_CFG="${1:?usage: bash $0 <input.cfg> <outdir> [max_concurrent]}"
    OUTDIR="${2:?usage: bash $0 <input.cfg> <outdir> [max_concurrent]}"
    MAX_CONC="${3:-}"

    export PATH="/scratch/project_2012355/Paper_3/mtp-env/bin:$PATH"
    python src/tagging_structs/VASP_settings.py \
        --input "$INPUT_CFG" \
        --outdir "$OUTDIR" \
        --kpar 4 --ncore 16

    OUTDIR_ABS="$(cd "$OUTDIR" && pwd)"
    N=$(ls -d "$OUTDIR_ABS"/struct_* | wc -l)
    pending=()
    for ((i = 1; i <= N; i++)); do
        _is_done "$OUTDIR_ABS/struct_$i" || pending+=("$i")
    done
    if [[ ${#pending[@]} -eq 0 ]]; then
        echo "All $N structures already finished — nothing to submit."
        exit 0
    fi

    ARRAY_SPEC="$(IFS=,; echo "${pending[*]}")${MAX_CONC:+%$MAX_CONC}"
    mkdir -p logs
    echo "Submitting ${#pending[@]}/$N structures from $OUTDIR_ABS"
    sbatch --array="$ARRAY_SPEC" --export=ALL,VASP_OUTDIR="$OUTDIR_ABS" "$0"
    exit 0
fi

# ---- Stage 2: one VASP run per array task ----
module load gcc/15.2.0 openmpi/5.0.10 hdf5/1.14.6 netlib-scalapack/2.2.2 fftw/3.3.10 vasp/6.4.3
export OMP_NUM_THREADS=1

STRUCT_DIR="${VASP_OUTDIR:?VASP_OUTDIR not set — submit via 'bash $0', not sbatch}/struct_${SLURM_ARRAY_TASK_ID}"
cd "$STRUCT_DIR"
echo "=== VASP struct_${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) ==="

if _is_done "$STRUCT_DIR"; then
    echo "Already finished — skipping."
    exit 0
fi

srun vasp_std > vasp.out 2> vasp.err
echo "Done: $(date)"
