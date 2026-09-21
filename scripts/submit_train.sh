#!/bin/bash
#SBATCH --job-name=mtp_train
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Train an MTP potential (src/train.py) on a compute node. mlp_binary in
# config/training.yaml is a plain (serial) launch command; this script wraps
# it with `srun -n $SLURM_NTASKS` (matching --ntasks above) via --mlp-binary,
# since running srun directly on the login node hits the login shell's
# zero-job association limit (AssocMaxSubmitJobLimit).
#
# Usage:
#   sbatch scripts/submit_train.sh <train.cfg> [template.almtp] [config.yaml]
#   sbatch scripts/submit_train.sh data/train.cfg
#   sbatch scripts/submit_train.sh data/train.cfg mtp_templates/20.almtp
#
# Output: results/potentials/pot_<random_id>/{pot.almtp, pot_training_info.txt}

set -euo pipefail

REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

TRAIN_CFG="${1:?Usage: sbatch scripts/submit_train.sh <train.cfg> [template.almtp] [config.yaml]}"
TEMPLATE="${2:-}"
CONFIG="${3:-config/training.yaml}"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MLP_BIN=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['mlp_binary'])")
MLP_LAUNCHER="srun -n ${SLURM_NTASKS:-20} $MLP_BIN"

ARGS=(--config "$CONFIG" --train-cfg "$TRAIN_CFG" --mlp-binary "$MLP_LAUNCHER")
[[ -n "$TEMPLATE" ]] && ARGS+=(--template "$TEMPLATE")

echo "=== MTP Training ==="
echo "  Job ID    : ${SLURM_JOB_ID:-local}"
echo "  Node      : $(hostname)"
echo "  Config    : $CONFIG"
echo "  Train CFG : $TRAIN_CFG"
echo "  MLP       : $MLP_LAUNCHER"
[[ -n "$TEMPLATE" ]] && echo "  Template  : $TEMPLATE"
date

python src/train.py "${ARGS[@]}"

echo "=== Done: $(date) ==="
