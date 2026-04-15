#!/usr/bin/env bash
#SBATCH --job-name=sim_states
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --array=0-3

# One array task = one (model_1, model_2) pair (2×2 Mistral grid).
# Submit: sbatch bash/run_simulate_and_save_states.sh
# Rerun one pair only: sbatch --array=1 bash/run_simulate_and_save_states.sh

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief
export SOTOPIA_RUN_LABEL=false_belief_fixed_two_agents

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit 1

MODEL_1=(
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Mistral-7B-Instruct-v0.2"
)
MODEL_2=(
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
  "Mistral-7B-Instruct-v0.3"
  "Mistral-7B-Instruct-v0.2"
)

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-3)}
model_1=${MODEL_1[$i]}
model_2=${MODEL_2[$i]}

echo "=== Job array task ${i} / 0-3 started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_ENVS_BASENAME=${SOTOPIA_ENVS_BASENAME}"
echo "SOTOPIA_RUN_LABEL=${SOTOPIA_RUN_LABEL}"
echo "Pair: ${model_1}  x  ${model_2}"
echo ""

python sample_normal_agent.py --model_1 "${model_1}" --model_2 "${model_2}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
