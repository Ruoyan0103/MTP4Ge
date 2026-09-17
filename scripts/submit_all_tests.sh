#!/usr/bin/env bash
#SBATCH --job-name=MTP4Ge_tests
#SBATCH --account=project_2012355
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/all_tests_%j.out
#SBATCH --error=logs/all_tests_%j.err
#
# Heavy MTP4Ge tests needing HPC memory — vacancy NEB + thermal properties.

set -euo pipefail
REPO_ROOT="/scratch/project_2012355/Paper_3/MTP4Ge"
cd "$REPO_ROOT"
POT="results/potentials/pot-init.almtp"

source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== MTP4Ge Heavy Tests ==="
echo "Potential: $POT"
echo "Started:   $(date)"

echo ""; echo "--- [1/2] vacancy_migration ---"
#python src/physical_validation/vacancy_migration.py --pot "$POT" && echo "PASS" || echo "FAIL"

echo ""; echo "--- [2/2] RDF ---"
#python src/physical_validation/amorphous_rdf.py --pot "$POT" && echo "PASS" || echo "FAIL"
python src/physical_validation/liquid_rdf.py --pot "$POT" && echo "PASS" || echo "FAIL"

echo ""; echo "=== DONE: $(date) ==="
