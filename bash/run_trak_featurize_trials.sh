#!/usr/bin/env bash
#SBATCH --job-name=trak_trials
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_trials_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_trials_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

# Single-GPU job: 300 trial gradients. At ~5 s/trial → ~25 min.
# Submit AFTER SFT featurize finishes (needs same projector seed).
#
# Smoke-test recipe (recommended first run, ~10-15 min, no array dep):
#   SMOKE=120 sbatch bash/run_trak_featurize_trials.sh    # ~10 min
#   SMOKE=180 sbatch bash/run_trak_featurize_trials.sh    # ~15 min
# Smoke outputs land in trial_grads_smoke/ (separate from real run).

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1
mkdir -p "${REPO_ROOT}/logs"

CHECKPOINT="${CHECKPOINT:-OLMo-3-7B-Instruct-SFT}"
PROJ_DIM="${PROJ_DIM:-4096}"
PROJ_SEED="${PROJ_SEED:-0}"
SMOKE_ARG=""
if [[ -n "${SMOKE:-}" ]]; then
  SMOKE_ARG="--smoke ${SMOKE}"
fi

echo "=== TRAK trial featurize start $(date) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Smoke: ${SMOKE:-none}"
echo ""

python analysis/trak/featurize_trials.py \
    --checkpoint "${CHECKPOINT}" \
    --proj_dim "${PROJ_DIM}" \
    --proj_seed "${PROJ_SEED}" \
    ${SMOKE_ARG}

echo ""
echo "=== Trial featurize finished at $(date) ==="
