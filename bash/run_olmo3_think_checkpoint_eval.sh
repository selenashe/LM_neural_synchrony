#!/usr/bin/env bash
#SBATCH --job-name=olmo3_think_fb
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_think_fb_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo3_think_fb_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-2

# One array task = one OLMo-3 Think-track checkpoint (3 total).
# Think models produce <think>...</think> reasoning blocks; the simulation
# script strips them before parsing Ava's response.
# Longer time limit (24h) because Think models generate more tokens per episode.
# Submit: sbatch bash/run_olmo3_think_checkpoint_eval.sh

CHECKPOINTS=(
  'OLMo-3-7B-Think-SFT'
  'OLMo-3-7B-Think-DPO'
  'OLMo-3-7B-Think-RLVR-step1375'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-2)}
checkpoint=${CHECKPOINTS[$i]}
if [[ -z "${checkpoint}" ]]; then
  echo "ERROR: empty checkpoint at array index ${i}" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-2 started at $(date) ==="
echo "Node: $(hostname)"
echo "Checkpoint: ${checkpoint}"
echo ""

python simulation/sample_olmo3_checkpoints.py --checkpoint_name "${checkpoint}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
