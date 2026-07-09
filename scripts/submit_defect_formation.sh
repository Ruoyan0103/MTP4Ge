#!/bin/bash
#SBATCH --job-name=mtp_defects
#SBATCH --account=project_2012355
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/defect_%j.out
#SBATCH --error=logs/defect_%j.err

set -euo pipefail
REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate

POT="${1:-results/potentials/pot_al.almtp}"
A0="${2:-5.779}"
FMAX="${3:-0.0001}"
MAXSTEPS="${4:-500}"

echo "Potential: $POT"
echo "a0 = $A0 Å | fmax = $FMAX eV/Å | max_steps = $MAXSTEPS"

python -m tests.defect_formation \
    --pot "$POT" \
    --a0 "$A0" \
    --fmax "$FMAX" \
    --max-steps "$MAXSTEPS"

echo "Done. Results: results/tests/defect_formation/"
