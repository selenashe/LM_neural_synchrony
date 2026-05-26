#!/usr/bin/env bash
#SBATCH --job-name=olmo31_32b_v60
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_32b_v60_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_32b_v60_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-3

# 4 OLMo-3.1 32B checkpoints × 1 v6.0 condition (multi_sys_report)
# = 4 array tasks. Each task runs 100 questions × 19 cues = 1900 multi-turn
# trials (~7 generate calls per trial, so ~13.3k forward+gen passes/task).
# Submit: sbatch bash/run_olmo31_32b_v6_0_pilot.sh

CHECKPOINTS=(
  'Olmo-3-1125-32B'
  'Olmo-3.1-32B-Instruct-SFT'
  'Olmo-3.1-32B-Instruct-DPO'
  'Olmo-3.1-32B-Instruct'
)

CONDITION='v6_0_multi_sys_report'

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

python simulation/sample_olmo3_checkpoints_v6_0.py \
    --checkpoint_name "${checkpoint}" \
    --condition "${CONDITION}" \
    --envs envs_v6_0_pilot.json

echo ""
echo "=== Task ${i} finished at $(date) ==="
