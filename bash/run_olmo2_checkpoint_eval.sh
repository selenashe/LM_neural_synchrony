#!/usr/bin/env bash
#SBATCH --job-name=olmo2_fb_v3
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo2_fb_v3_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo2_fb_v3_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-9

# One array task = one OLMo-2 checkpoint (10 total).
# Evaluates false-belief v3 scenarios with fixed Mia opening.
# Submit: sbatch bash/run_olmo2_checkpoint_eval.sh

CHECKPOINTS=(
  'OLMo-2-1124-7B-Base'
  'OLMo-2-1124-7B-SFT'
  'OLMo-2-1124-7B-DPO'
  'OLMo-2-1124-7B-Instruct'
  'OLMo-2-1124-7B-RLVR-step60'
  'OLMo-2-1124-7B-RLVR-step120'
  'OLMo-2-1124-7B-RLVR-step180'
  'OLMo-2-1124-7B-RLVR-step240'
  'OLMo-2-1124-7B-RLVR-step300'
  'OLMo-2-1124-7B-RLVR-step360'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-9)}
checkpoint=${CHECKPOINTS[$i]}
if [[ -z "${checkpoint}" ]]; then
  echo "ERROR: empty checkpoint at array index ${i}" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-9 started at $(date) ==="
echo "Node: $(hostname)"
echo "Checkpoint: ${checkpoint}"
echo ""

python simulation/sample_olmo2_checkpoints.py --checkpoint_name "${checkpoint}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
