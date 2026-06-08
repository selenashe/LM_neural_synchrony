#!/usr/bin/env bash
#SBATCH --job-name=trak_trials_array
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_trials_array_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/trak_trials_array_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --array=0-14

# 15-way shard of TRIAL-side TRAK featurization for the 4,432-trial expansion.
#   4,432 trials / 15 shards = ~295 per task.
#   At ~7.5 s/trial → each task finishes in ~37 min. Wall time cap 2h for safety.
#
# Defaults to the all-trials manifest and a separate out_dir so the legacy
# 300-trial outputs aren't clobbered.
#
# Submit:        sbatch bash/run_trak_featurize_trials_array.sh
# Emitted target: USE_EMITTED=1 sbatch bash/run_trak_featurize_trials_array.sh
# 300-trial pilot (single job, no array): use bash/run_trak_featurize_trials.sh
# True smoke (small N global): use bash/run_trak_featurize_trials.sh with SMOKE=N

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

# Use the conda env's python explicitly. `python` on its own can resolve to
# ~/.venv_brain/bin/python (which doesn't have pandas/torch) because that
# venv's auto-activation in .bashrc shadows the conda env even after
# `conda activate`. Override via PYTHON_BIN if needed.
PYTHON_BIN="${PYTHON_BIN:-/nlp/scr/jshe/miniconda3/envs/neural_sync/bin/python}"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1
mkdir -p "${REPO_ROOT}/logs"

NUM_SHARDS=15
i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch --array=0-14}

TRIAL_MANIFEST="${TRIAL_MANIFEST:-analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv}"
OUT_DIR="${OUT_DIR:-/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads_all}"
CHECKPOINT="${CHECKPOINT:-OLMo-3-7B-Instruct-SFT}"
PROJ_DIM="${PROJ_DIM:-4096}"
PROJ_SEED="${PROJ_SEED:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
USE_EMITTED_ARG=""
if [[ -n "${USE_EMITTED:-}" ]]; then
  USE_EMITTED_ARG="--use_emitted"
fi

echo "=== TRAK trial shard ${i}/$((NUM_SHARDS-1)) start $(date) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: ${PYTHON_BIN}"
echo "Manifest: ${TRIAL_MANIFEST}"
echo "Out dir:  ${OUT_DIR}"
echo "Checkpoint: ${CHECKPOINT}  proj_dim=${PROJ_DIM}  max_seq_len=${MAX_SEQ_LEN}"
echo "Use emitted: ${USE_EMITTED:-no}"
echo ""

"${PYTHON_BIN}" analysis/trak/featurize_trials.py \
    --trial_manifest "${TRIAL_MANIFEST}" \
    --out_dir "${OUT_DIR}" \
    --shard_idx "${i}" \
    --num_shards "${NUM_SHARDS}" \
    --checkpoint "${CHECKPOINT}" \
    --proj_dim "${PROJ_DIM}" \
    --proj_seed "${PROJ_SEED}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    ${USE_EMITTED_ARG}

echo ""
echo "=== Shard ${i} finished at $(date) ==="
