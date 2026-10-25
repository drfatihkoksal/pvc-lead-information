#!/bin/bash
# Full pipeline, in dependency order. Paths come from the environment variables
# described in src/config.py and README.md.
set -euo pipefail
cd "$(dirname "$0")"
WORK=$(python3 -c "import sys; sys.path.insert(0, 'src'); import config; print(config.WORK)")
mkdir -p "$WORK/logs"
run() { echo "== $1"; python3 "src/$1.py" "${@:2}" > "$WORK/logs/$1.log" 2>&1; }

run beat_classifier
run beat_extraction
run quality_control
run figure1
run train_eval 12 8
run cpsc_internal_cv
run cpsc_internal_subsets
run sinus_comparison
run pareto_analysis
run figures
run selection_stability
run montage_retention
run paired_tests
run separability
echo "done -> $WORK"
