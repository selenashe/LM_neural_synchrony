#!/usr/bin/env bash
#SBATCH --job-name=olmo31_32b_v4_full
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_32b_v4_full_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_32b_v4_full_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-3

# One array task = one OLMo-3.1 32B Instruct-track checkpoint (4 total).
# Full 100-scenario v4 3-choice forced-choice (4,800 trials per checkpoint).
# Submit: sbatch bash/run_olmo31_32b_v4_3choice_full.sh

CHECKPOINTS=(
  'Olmo-3-1125-32B'
  'Olmo-3.1-32B-Instruct-SFT'
  'Olmo-3.1-32B-Instruct-DPO'
  'Olmo-3.1-32B-Instruct'
)

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
echo ""

python simulation/sample_olmo3_checkpoints_v4_3choice.py \
    --checkpoint_name "${checkpoint}" \
    --envs envs_false_belief_v4.json

echo ""
echo "=== Task ${i} finished at $(date) ==="
