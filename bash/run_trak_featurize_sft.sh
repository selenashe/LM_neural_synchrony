#!/usr/bin/env bash
#SBATCH --job-name=trak_sft
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_sft_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_sft_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-14

# 15-way shard of SFT featurization for TRAK.
#   50,000 SFT examples / 15 shards = ~3,334 per task.
#   At ~3 s/example (bf16, 7B, seq_len≤1024, proj_dim=4096), each task
#   should finish in 2.5-5h. Wall time cap is 12h for safety.
#
# Submit: sbatch bash/run_trak_featurize_sft.sh
# Smoke-test one shard first:
#   SMOKE=16 sbatch --array=0 bash/run_trak_featurize_sft.sh

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1
mkdir -p "${REPO_ROOT}/logs"

NUM_SHARDS=15
i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch --array=0-14}

CHECKPOINT="${CHECKPOINT:-OLMo-3-7B-Instruct-SFT}"
PROJ_DIM="${PROJ_DIM:-4096}"
PROJ_SEED="${PROJ_SEED:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
SMOKE_ARG=""
if [[ -n "${SMOKE:-}" ]]; then
  SMOKE_ARG="--smoke ${SMOKE}"
fi

echo "=== TRAK SFT shard ${i}/$((NUM_SHARDS-1)) start $(date) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Checkpoint: ${CHECKPOINT}  proj_dim=${PROJ_DIM}  max_seq_len=${MAX_SEQ_LEN}"
echo "Smoke: ${SMOKE:-none}"
echo ""

python analysis/trak/featurize_sft.py \
    --shard_idx "${i}" \
    --num_shards "${NUM_SHARDS}" \
    --checkpoint "${CHECKPOINT}" \
    --proj_dim "${PROJ_DIM}" \
    --proj_seed "${PROJ_SEED}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    ${SMOKE_ARG}

echo ""
echo "=== Shard ${i} finished at $(date) ==="
