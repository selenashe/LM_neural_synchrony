#!/usr/bin/env bash
#SBATCH --job-name=olmo3_7b_v50jsc
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v50jsc_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v50jsc_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --array=0-31

# 4 OLMo-3 7B Instruct checkpoints x 8 v5.0 jing-strong-cue conditions = 32 tasks.
# Single generic cue ("That's wrong! Acknowledge the mistake.") × 20 MMLU
# questions = 20 trials per condition.
# Submit: sbatch bash/run_olmo3_7b_v5_0_jing_strong_cue_pilot.sh

CHECKPOINTS=(
  'OLMo-3-1025-7B'
  'OLMo-3-7B-Instruct-SFT'
  'OLMo-3-7B-Instruct-DPO'
  'OLMo-3-7B-Instruct-RLVR-step400'
)

CONDITIONS=(
  'v5_0_jing_strong_cue_single_sys_quote'
  'v5_0_jing_strong_cue_single_sys_report'
  'v5_0_jing_strong_cue_single_uo_quote'
  'v5_0_jing_strong_cue_single_uo_report'
  'v5_0_jing_strong_cue_multi_sys_quote'
  'v5_0_jing_strong_cue_multi_sys_report'
  'v5_0_jing_strong_cue_multi_uo_quote'
  'v5_0_jing_strong_cue_multi_uo_report'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-31)}
ckpt_idx=$((i / 8))
cond_idx=$((i % 8))
checkpoint=${CHECKPOINTS[$ckpt_idx]}
condition=${CONDITIONS[$cond_idx]}

if [[ -z "${checkpoint}" || -z "${condition}" ]]; then
  echo "ERROR: empty checkpoint/condition at array index ${i} (ckpt_idx=${ckpt_idx}, cond_idx=${cond_idx})" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-31 started at $(date) ==="
echo "Node: $(hostname)"
echo "Checkpoint: ${checkpoint}"
echo "Condition:  ${condition}"
echo ""

python simulation/sample_olmo3_checkpoints_v5_0_jing_strong_cue.py \
    --checkpoint_name "${checkpoint}" \
    --condition "${condition}" \
    --envs envs_jing_strong_cue_v5_0_pilot.json

echo ""
echo "=== Task ${i} finished at $(date) ==="
