#!/usr/bin/env bash
#SBATCH --job-name=sim_states
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00

# Override the default environment agent combos file.
export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_fixed_two_agents.json
# Set separate sotopia_results/... folder from the default run.
export SOTOPIA_RUN_LABEL=mia_ava_fixed_two_agents

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

# Optional: Mia+Ava on all 90 scenarios (see env_agent_combos_fixed_two_agents.json).
# export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_fixed_two_agents.json
# Optional: separate sotopia_results/... folder from the default run.
# export SOTOPIA_RUN_LABEL=mia_ava_fixed

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo ""

model_1_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2 Meta-Llama-3-8B-Instruct"
model_2_list="Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.2 Meta-Llama-3-8B-Instruct"

for model_1 in $model_1_list; do
    for model_2 in $model_2_list; do
        echo "===== ${model_1} x ${model_2} ====="
        python sample_normal_agent.py --model_1 "${model_1}" --model_2 "${model_2}"
        echo ""
    done
done

echo "=== Job finished at $(date) ==="
