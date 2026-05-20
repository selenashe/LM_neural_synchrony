#!/usr/bin/env bash
#SBATCH --job-name=v45_robust
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/v45_robust_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/v45_robust_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-31

# 4 models × 8 v4.5 conditions = 32 array tasks.
# Pilot env (520 trials per condition).
# Submit: sbatch bash/run_v4_5_robustness_check.sh

MODELS=(
  'Qwen3-30B-A3B'
  'Qwen3-8B'
  'Mistral-7B-Instruct-v0.3'
  'DeepSeek-R1-Distill-Llama-8B'
)

CONDITIONS=(
  'v4_5_single_sys_quote'
  'v4_5_single_sys_report'
  'v4_5_single_uo_quote'
  'v4_5_single_uo_report'
  'v4_5_multi_sys_quote'
  'v4_5_multi_sys_report'
  'v4_5_multi_uo_quote'
  'v4_5_multi_uo_report'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-31)}
model_idx=$((i / 8))
cond_idx=$((i % 8))
model=${MODELS[$model_idx]}
condition=${CONDITIONS[$cond_idx]}

if [[ -z "${model}" || -z "${condition}" ]]; then
  echo "ERROR: empty model/condition at array index ${i} (model_idx=${model_idx}, cond_idx=${cond_idx})" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-31 started at $(date) ==="
echo "Node: $(hostname)"
echo "Model: ${model}"
echo "Condition:  ${condition}"
echo ""

python simulation/sample_v4_5_robustness_check.py \
    --model_name "${model}" \
    --condition "${condition}" \
    --envs envs_false_belief_v4_1_pilot.json

echo ""
echo "=== Task ${i} finished at $(date) ==="
