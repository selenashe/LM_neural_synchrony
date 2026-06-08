#!/usr/bin/env bash
#SBATCH --job-name=olmo3_7b_v50jnos
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v50jnos_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v50jnos_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-3

# 4 OLMo-3 7B Instruct checkpoints × 1 v5.0 jing-non-option-suggested-cue
# condition (multi_sys_report) = 4 array tasks. Each task runs 20 MMLU
# questions × 38 cues = 760 multi-turn trials (T2 + T4 + cascade
# T6/8/10/12/14 = 7 generate calls per trial). v6.0 wall (9500 trials /
# 48h) scales to ~4h here; bumped to 12h for safety on slow nodes.
# Submit: sbatch bash/run_olmo3_7b_v5_0_jing_nonoption_suggested_cue_pilot.sh

CHECKPOINTS=(
  'OLMo-3-1025-7B'
  'OLMo-3-7B-Instruct-SFT'
  'OLMo-3-7B-Instruct-DPO'
  'OLMo-3-7B-Instruct-RLVR-step400'
)

CONDITION='v5_0_jing_nonoption_suggested_cue_multi_sys_report'

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-3)}
checkpoint=${CHECKPOINTS[$i]}

if [[ -z "${checkpoint}" ]]; then
  echo "ERROR: empty checkpoint at array index ${i}" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-3 started at $(date) ==="
echo "Node: $(hostname)"
echo "Checkpoint: ${checkpoint}"
echo "Condition:  ${CONDITION}"
echo ""

python simulation/sample_olmo3_checkpoints_v5_0_jing_nonoption_suggested_cue.py \
    --checkpoint_name "${checkpoint}" \
    --condition "${CONDITION}" \
    --envs envs_jing_nonoption_suggested_cue_v5_0_pilot.json

echo ""
echo "=== Task ${i} finished at $(date) ==="
