#!/usr/bin/env bash
#SBATCH --job-name=sim_dond
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_dond_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_dond_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --array=0-7

# One array task = one self-pair (modelA x modelA). 8 models.
# Submit: sbatch bash/run_deal_or_no_deal.sh

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_deal_or_no_deal.json
export SOTOPIA_ENVS_BASENAME=envs_deal_or_no_deal.json
export SOTOPIA_RESULTS_DIR=sotopia_results_deal_or_no_deal
export SOTOPIA_RUN_LABEL=deal_or_no_deal

MODELS=(
  'allenai_OLMo-3-7B-Instruct'
  'DeepSeek-R1-Distill-Llama-8B'
  'Meta-Llama-3-8B-Instruct'
  'Mistral-7B-Instruct-v0.2'
  'Mistral-7B-Instruct-v0.3'
  'Qwen2.5-7B-Instruct'
  'Qwen3-8B'
  'Qwen_Qwen3-14B'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-7)}
model_1=${MODELS[$i]}
model_2=${MODELS[$i]}
if [[ -z "${model_1}" ]]; then
  echo "ERROR: empty model at array index ${i}" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-7 started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_ENVS_BASENAME=${SOTOPIA_ENVS_BASENAME}"
echo "SOTOPIA_RUN_LABEL=${SOTOPIA_RUN_LABEL}"
echo "Pair: ${model_1}  x  ${model_2}"
echo ""

python simulation/sample_normal_agent.py --model_1 "${model_1}" --model_2 "${model_2}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
