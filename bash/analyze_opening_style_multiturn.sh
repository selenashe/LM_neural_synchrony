#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Analyze multi-turn opening style experiment results.
# Produces: all_results.csv, behavioral_summary.csv, accuracy plots,
#           turn distributions, quality flags.
#
# Usage: bash bash/analyze_opening_style_multiturn.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

echo "=== Analyze multi-turn opening style experiment: started at $(date) ==="
python analysis/analyze_opening_style_multiturn.py \
    --results_dir opening_style_multiturn_results \
    --scenarios_csv summary_plots_false_belief_controlled_deception_prompt_gemini/all_results.csv \
    --output_dir summary_plots_opening_style_multiturn
echo ""

echo "=== Done at $(date) ==="
