#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run multi-turn opening style experiment with Gemini locally.
# Mia's Turn 0 is fixed to each of 13 patterns; subsequent turns are
# generated normally by Gemini for both agents.
#
# Usage: bash bash/run_opening_style_multiturn.sh
#        bash bash/run_opening_style_multiturn.sh --max_episodes 2 --pattern "The X is [prep] Y"
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/selena/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

echo "=== Multi-turn opening style experiment: started at $(date) ==="
python simulation/sample_gemini_opening_style.py \
    --scenarios_csv summary_plots_false_belief_controlled_deception_prompt_gemini/all_results.csv \
    --output_dir opening_style_multiturn_results \
    "$@"
echo ""

echo "=== Done at $(date) ==="
