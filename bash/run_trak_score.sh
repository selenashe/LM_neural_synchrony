#!/usr/bin/env bash
#SBATCH --job-name=trak_score
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_score_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_score_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# Final TRAK step: stack SFT shards, form Gram matrix (D x D), Cholesky-solve,
# and produce (n_trials, n_sft) attribution scores. Pure linear algebra; should
# finish in well under 30 min on a single 80G GPU.
#
# Submit AFTER both featurize jobs are done.

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

# Conda's `python` can be shadowed by ~/.venv_brain (auto-activated in .bashrc).
# Use the env's python explicitly. Override via PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-/nlp/scr/jshe/miniconda3/envs/neural_sync/bin/python}"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1
mkdir -p "${REPO_ROOT}/logs"

NUM_SHARDS="${NUM_SHARDS:-15}"
NUM_TRIAL_SHARDS="${NUM_TRIAL_SHARDS:-15}"
LAM="${LAM:-1e7}"
NORMALIZE="${NORMALIZE:-0}"
OUT_NAME="${OUT_NAME:-attribution_scores_all_lam1e7.npy}"
TRIAL_GRADS_DIR="${TRIAL_GRADS_DIR:-/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads_all}"
SFT_GRADS_DIR="${SFT_GRADS_DIR:-/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/sft_grads}"
OUT_DIR_SCRATCH="${OUT_DIR_SCRATCH:-/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/scores}"

echo "=== TRAK score start $(date) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: ${PYTHON_BIN}"
echo "num_shards=${NUM_SHARDS}  num_trial_shards=${NUM_TRIAL_SHARDS}"
echo "lambda=${LAM}  normalize=${NORMALIZE}  out_name=${OUT_NAME}"
echo "trial_grads_dir=${TRIAL_GRADS_DIR}"
echo "sft_grads_dir=${SFT_GRADS_DIR}"
echo "out_dir=${OUT_DIR_SCRATCH}"
echo ""

EXTRA=""
if [ "${NORMALIZE}" = "1" ]; then
    EXTRA="--normalize"
fi

"${PYTHON_BIN}" analysis/trak/score_trak.py \
    --sft_grads_dir   "${SFT_GRADS_DIR}" \
    --trial_grads_dir "${TRIAL_GRADS_DIR}" \
    --out_dir         "${OUT_DIR_SCRATCH}" \
    --num_shards "${NUM_SHARDS}" \
    --num_trial_shards "${NUM_TRIAL_SHARDS}" \
    --lam "${LAM}" \
    --out_name "${OUT_NAME}" \
    ${EXTRA}

echo ""
echo "=== Score finished at $(date) ==="
