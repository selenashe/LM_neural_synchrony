#!/usr/bin/env bash
#SBATCH --job-name=ll_fb
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --array=0-9999
# NOTE: 64 model pairs × 600 episodes = 38400 tasks total.
# SLURM MaxArraySize is typically ~10001, so submit multiple batches:
#   BATCH_OFFSET=0     sbatch run_logit_lens_false_belief.sh   (tasks 0-9999)
#   BATCH_OFFSET=10000 sbatch run_logit_lens_false_belief.sh   (tasks 10000-19999)
#   BATCH_OFFSET=20000 sbatch run_logit_lens_false_belief.sh   (tasks 20000-29999)
#   BATCH_OFFSET=30000 sbatch --array=0-8399 run_logit_lens_false_belief.sh (tasks 30000-38399)
# Pair order must match bash/pairs_false_belief_45.bash (see scripts/list_false_belief_pairs.py).

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

PAIRS_FILE="${REPO_ROOT}/bash/pairs_false_belief_45.bash"
if [[ ! -f "${PAIRS_FILE}" ]]; then
  echo "ERROR: missing ${PAIRS_FILE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${PAIRS_FILE}"

N_EPISODES=600
GLOBAL_ID=$(( ${BATCH_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
PAIR_IDX=$(( GLOBAL_ID / N_EPISODES ))
EPISODE=$(( GLOBAL_ID % N_EPISODES ))

M1="${MODEL_1[$PAIR_IDX]}"
M2="${MODEL_2[$PAIR_IDX]}"
if [[ -z "${M1}" || -z "${M2}" ]]; then
  echo "ERROR: empty model pair at pair_idx ${PAIR_IDX} (MODEL_1/MODEL_2 not set?)" >&2
  exit 1
fi

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Pair ${PAIR_IDX}: ${M1} × ${M2}  |  Episode: ${EPISODE}"
echo ""

python analysis/logit_lens_false_belief.py \
    --model_1 "${M1}" \
    --model_2 "${M2}" \
    --episode "${EPISODE}" \
    --seed 0 \
    --temp 0.7 \
    --hf-cache "${HF_HOME}" \
    --results_dir sotopia_results_false_belief_100

echo ""
echo "=== Job finished at $(date) ==="
