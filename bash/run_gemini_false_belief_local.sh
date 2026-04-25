#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run false-belief Sotopia simulations with two Gemini agents locally.
#
# Two gemini-3.1-pro-preview agents converse for up to 15 turns per episode
# across 600 false-belief scenarios. No per-turn token limit.
#
# Follows the same env-var protocol as run_simulate_and_save_states.sh:
#   SOTOPIA_ENV_AGENT_COMBOS_BASENAME  – combos file under sotopia_data/
#   SOTOPIA_ENVS_BASENAME              – envs file under sotopia_data/
#   SOTOPIA_RESULTS_DIR                – output directory name
#   SOTOPIA_RUN_LABEL                  – suffix for output folder
#
# Usage (from repo root or anywhere):
#   bash bash/run_gemini_false_belief_local.sh
#   bash bash/run_gemini_false_belief_local.sh --max_episodes 5   # smoke test
# ---------------------------------------------------------------------------
set -euo pipefail

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_100
export SOTOPIA_RUN_LABEL=false_belief_fixed_two_agents

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs"

model_1="gemini-3.1-pro-preview"
model_2="gemini-3.1-pro-preview"

echo "=== Gemini false-belief local run started at $(date) ==="
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
echo "=== Gemini false-belief local run finished at $(date) ==="
