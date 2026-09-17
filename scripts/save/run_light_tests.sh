#!/usr/bin/env bash
# ============================================================================
# run_light_tests.sh — Run lightweight MTP4Ge tests using pot-init.almtp.
#
# energy_volume runs first to determine the equilibrium lattice constant (a0).
# That a0 is then fed to defect_formation, elastic_constants, and
# quasi_static_drag.  multiphase_ev fits its own per-phase EOS so it doesn't
# need a0.  phonon_dispersion uses a fixed experimental lattice constant
# (5.6524 Å) as requested.
#
# Usage:
#   bash scripts/save/run_light_tests.sh [pot.almtp]
#
# Output under results/tests/<name>/ (defaults used, no --outdir override).
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

POT="${1:-results/potentials/pot-init.almtp}"
PHONON_ALAT="5.6524"

# ---- Activate environment ----
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MTP_POT="${REPO_ROOT}/${POT}"

PASS=0
FAIL=0
FAILED_TESTS=()

# ---------------------------------------------------------------------------
# Helper: run a test, track pass/fail
# ---------------------------------------------------------------------------
run_test() {
    local test_path="$1"
    local test_name="$2"
    shift 2
    echo ""
    echo "============================================================"
    echo "  $test_name"
    echo "  python $test_path $*"
    echo "  Started: $(date)"
    echo "============================================================"
    if python "$test_path" "$@"; then
        echo "  ==> PASSED: $test_name"
        PASS=$((PASS + 1))
    else
        echo "  ==> FAILED: $test_name (exit code $?)"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$test_name")
    fi
    echo "  Finished: $(date)"
}

# ===========================================================================
# Step 1: energy_volume — extract a0
# ===========================================================================
#run_test src/physical_validation/energy_volume.py "energy_volume" --pot "$POT"

EOS_FILE="results/tests/energy_volume/eos_fit.txt"
if [[ -f "$EOS_FILE" ]]; then
    A0=$(grep "^a0 " "$EOS_FILE" | grep -oP '=\s*\K[0-9.]+')
    echo ""
    echo "  Extracted a0 = $A0 Å from $EOS_FILE"
else
    echo "  ERROR: $EOS_FILE not found — cannot extract a0"
    exit 1
fi

# ===========================================================================
# Step 2: Tests using a0 from energy_volume
# ===========================================================================
#run_test src/physical_validation/defect_formation.py  "defect_formation"  --pot "$POT" --a0 "$A0"
run_test src/physical_validation/elastic_constants.py "elastic_constants" --pot "$POT" --lattice-constant "$A0"
#run_test src/physical_validation/quasi_static_drag.py "quasi_static_drag" --pot "$POT" --a0 "$A0"

# ===========================================================================
# Step 3: multiphase_ev — no lattice constant needed
# ===========================================================================
#run_test src/physical_validation/multiphase_ev.py "multiphase_ev" --pot "$POT"

# ===========================================================================
# Step 4: phonon_dispersion — fixed experimental lattice constant
# ===========================================================================
#run_test src/physical_validation/phonon_dispersion.py "phonon_dispersion" --pot "$POT" --alat "$PHONON_ALAT"

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "============================================================"
echo "  ALL LIGHT TESTS COMPLETE"
echo "  Finished: $(date)"
echo "  PASS: $PASS  |  FAIL: $FAIL"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed: ${FAILED_TESTS[*]}"
fi
echo "============================================================"

exit $FAIL
