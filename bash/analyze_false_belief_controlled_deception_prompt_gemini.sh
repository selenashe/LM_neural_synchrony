#!/usr/bin/env bash
#SBATCH --job-name=fb_cdp_analyze_gem
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/fb_cdp_analyze_gemini_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/fb_cdp_analyze_gemini_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

# Analysis for the controlled deception prompt variant.
# Uses the same v1 analysis scripts (same 6 conditions), different data.
# Usage: sbatch bash/analyze_false_belief_controlled_deception_prompt_gemini.sh
#    or: bash bash/analyze_false_belief_controlled_deception_prompt_gemini.sh

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

RESULTS_DIR="logit_lens_results_false_belief_controlled_deception_prompt_gemini"

echo "=== Summarize results: started at $(date) ==="
python analysis/summarize_false_belief_results.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir summary_plots_false_belief_controlled_deception_prompt_gemini
echo ""

echo "=== Quality audit: started at $(date) ==="
python analysis/audit_conversation_quality.py \
    --results_dir "${RESULTS_DIR}"
echo ""

echo "=== Error case extraction: started at $(date) ==="
python analysis/extract_error_cases.py \
    --results_dir "${RESULTS_DIR}"
echo ""

echo "=== All done at $(date) ==="
