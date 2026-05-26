#!/usr/bin/env bash
#SBATCH --job-name=olmo3_7b_v60_base_rerun
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v60_base_rerun_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_7b_v60_base_rerun_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --array=0-11

# Re-run of OLMo-3-1025-7B (base only) with v5.0-style "\n\nAnswer:" prompt.
# The v6.0 verbose instruction prompt drove base to 58.9% null on multi_sys_report
# (v5.0 was 0% on the same condition); the simulation wrapper now selects the
# v5.0 prompt automatically for base checkpoints.
#
# 9500 trials × 7 generations split into 12 array tasks of 800 trials each
# (last task: 700). Each task should finish in ~2-4h; 5h walltime gives buffer.
#
# Submit:  sbatch bash/run_olmo3_7b_v6_0_pilot_base_rerun.sh
# Merge:   python simulation/merge_v6_0_base_batches.py   (after all tasks finish)

CHECKPOINT='OLMo-3-1025-7B'
CONDITION='v6_0_multi_sys_report'
BATCH_SIZE=800
TOTAL_TRIALS=9500

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-11)}

START=$(( i * BATCH_SIZE ))
END=$(( START + BATCH_SIZE ))
if (( END > TOTAL_TRIALS )); then END=${TOTAL_TRIALS}; fi
COUNT=$(( END - START ))

if (( COUNT <= 0 )); then
  echo "Task ${i}: nothing to do (start=${START} >= total=${TOTAL_TRIALS})"
  exit 0
fi

LABEL=$(printf "base_batch%02d" "${i}")

echo "=== Base re-run task ${i} / 0-11 started at $(date) ==="
echo "Node:        $(hostname)"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Condition:   ${CONDITION}"
echo "Episodes:    [${START}, ${END})  (${COUNT} trials)"
echo "Label:       ${LABEL}"
echo ""

python simulation/sample_olmo3_checkpoints_v6_0.py \
    --checkpoint_name "${CHECKPOINT}" \
    --condition "${CONDITION}" \
    --envs envs_v6_0_pilot.json \
    --start_idx ${START} \
    --max_episodes ${COUNT} \
    --label "${LABEL}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
