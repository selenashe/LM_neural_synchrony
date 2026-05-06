#!/usr/bin/env bash
# Run the controlled opening-style experiment (local, API-based).
# Usage: bash bash/run_opening_style_experiment.sh


REPO_ROOT="${REPO_ROOT:-/Users/selena/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

echo "=== Opening style experiment: started at $(date) ==="
python analysis/run_opening_style_experiment.py \
    --scenarios_csv summary_plots_false_belief_controlled_deception_prompt_gemini/all_results.csv \
    --output_dir opening_style_experiment_results
echo ""

echo "=== Done at $(date) ==="
