#!/usr/bin/env bash
#SBATCH --job-name=sim_states
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --array=0-8

# One array task = one (model_1, model_2) pair (same 3×3 grid as before).
# Submit: sbatch bash/run_simulate_and_save_states.sh
# Rerun one pair only: sbatch --array=3 bash/run_simulate_and_save_states.sh

# Override the default environment agent combos file.
export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_fixed_two_agents.json
# Set separate sotopia_results/... folder from the default run.
export SOTOPIA_RUN_LABEL=mia_ava_fixed_two_agents

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit 1

# Row-major order: outer model_1, inner model_2 (matches previous nested loops).
MODEL_1=(
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Mistral-7B-Instruct-v0.2"
  "Mistral-7B-Instruct-v0.2"
  "Meta-Llama-3-8B-Instruct"
  "Meta-Llama-3-8B-Instruct"
  "Meta-Llama-3-8B-Instruct"
)
MODEL_2=(
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Meta-Llama-3-8B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Meta-Llama-3-8B-Instruct"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Meta-Llama-3-8B-Instruct"
)

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-8)}
model_1=${MODEL_1[$i]}
model_2=${MODEL_2[$i]}

echo "=== Job array task ${i} / 0-8 started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_RUN_LABEL=${SOTOPIA_RUN_LABEL}"
echo "Pair: ${model_1}  x  ${model_2}"
echo ""

python sample_normal_agent.py --model_1 "${model_1}" --model_2 "${model_2}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
