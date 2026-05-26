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

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1
mkdir -p "${REPO_ROOT}/logs"

NUM_SHARDS="${NUM_SHARDS:-15}"
LAM="${LAM:-1e-2}"
NORMALIZE="${NORMALIZE:-0}"
OUT_NAME="${OUT_NAME:-attribution_scores.npy}"

echo "=== TRAK score start $(date) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "num_shards=${NUM_SHARDS}  lambda=${LAM}  normalize=${NORMALIZE}  out_name=${OUT_NAME}"
echo ""

EXTRA=""
if [ "${NORMALIZE}" = "1" ]; then
    EXTRA="--normalize"
fi

python analysis/trak/score_trak.py \
    --num_shards "${NUM_SHARDS}" \
    --lam "${LAM}" \
    --out_name "${OUT_NAME}" \
    ${EXTRA}

echo ""
echo "=== Score finished at $(date) ==="
