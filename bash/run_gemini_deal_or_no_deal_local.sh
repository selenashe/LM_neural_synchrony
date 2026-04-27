#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run deal-or-no-deal Sotopia simulations with two Gemini agents locally.
#
# Two gemini-3.1-pro-preview agents converse for up to 15 turns per episode
# across 100 deal-or-no-deal negotiation scenarios.
#
# Usage (from repo root or anywhere):
#   bash bash/run_gemini_deal_or_no_deal_local.sh
#   bash bash/run_gemini_deal_or_no_deal_local.sh --max_episodes 5   # smoke test
# ---------------------------------------------------------------------------
set -euo pipefail

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_deal_or_no_deal.json
export SOTOPIA_ENVS_BASENAME=envs_deal_or_no_deal.json
export SOTOPIA_RESULTS_DIR=sotopia_results_deal_or_no_deal_gemini
export SOTOPIA_RUN_LABEL=deal_or_no_deal

# Conda activation (optional - skip if already in correct env)
if [ -f /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh ]; then
    source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
    conda activate neural_sync
fi

REPO_ROOT="${REPO_ROOT:-/Users/selena/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs"

model_1="gemini-3.1-pro-preview"
model_2="gemini-3.1-pro-preview"

echo "=== Gemini deal-or-no-deal local run started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_ENVS_BASENAME=${SOTOPIA_ENVS_BASENAME}"
echo "SOTOPIA_RESULTS_DIR=${SOTOPIA_RESULTS_DIR}"
echo "SOTOPIA_RUN_LABEL=${SOTOPIA_RUN_LABEL}"
echo "Pair: ${model_1}  x  ${model_2}"
echo ""

python simulation/sample_gemini_false_belief.py \
    --model_1 "${model_1}" \
    --model_2 "${model_2}" \
    "$@"

echo ""
echo "=== Gemini deal-or-no-deal local run finished at $(date) ==="
