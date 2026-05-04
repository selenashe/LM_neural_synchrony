#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run false-belief Sotopia simulations with two Gemini agents locally.
# Controlled deception prompt variant: Mia's deceive goal uses
# "Guide them to the wrong location." (no "without being obvious").
#
# Usage (from repo root or anywhere):
#   bash bash/run_gemini_false_belief_controlled_deception_prompt_local.sh
#   bash bash/run_gemini_false_belief_controlled_deception_prompt_local.sh --max_episodes 5
# ---------------------------------------------------------------------------
set -euo pipefail

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_controlled_deception_prompt_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief_controlled_deception_prompt.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_controlled_deception_prompt_gemini
export SOTOPIA_RUN_LABEL=false_belief_controlled_deception_prompt_fixed_two_agents

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

echo "=== Gemini false-belief (controlled deception prompt) local run started at $(date) ==="
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
echo "=== Gemini false-belief (controlled deception prompt) local run finished at $(date) ==="
