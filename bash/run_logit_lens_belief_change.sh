#!/usr/bin/env bash
#SBATCH --job-name=ll_bc
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_bc_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_bc_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --array=0-3290
# ---------------------------------------------------------------------------
# Logit-lens analysis restricted to "belief change" episodes:
#   - false_help  where correct == 1  (seeker persuaded to correct location)
#   - false_deceive where correct == 0 (seeker persuaded to wrong location)
#
# Reads from a pre-generated manifest (one line per task):
#   model_1 <TAB> model_2 <TAB> episode_index
#
# Generate the manifest first:
#   cd /juice6/u/jshe/nlp/LM_neural_synchrony
#   python bash/generate_belief_change_manifest.py
#
# Then submit:
#   sbatch bash/run_logit_lens_belief_change.sh
#
# Total tasks: 3291 (fits in one SLURM array submission)
# Estimated wall time per task: ~15 min median, ~19 min mean
# Estimated total GPU-hours: ~1040 (with full parallelism, wall ~4h)
# ---------------------------------------------------------------------------
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

MANIFEST="${REPO_ROOT}/bash/logit_lens_belief_change_manifest.tsv"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found at ${MANIFEST}" >&2
  echo "Run: python bash/generate_belief_change_manifest.py" >&2
  exit 1
fi

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
if [[ -z "${LINE}" ]]; then
  echo "ERROR: no entry at line $((SLURM_ARRAY_TASK_ID + 1)) in manifest" >&2
  exit 1
fi

M1=$(echo "${LINE}" | cut -f1)
M2=$(echo "${LINE}" | cut -f2)
EPISODE=$(echo "${LINE}" | cut -f3)

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Pair: ${M1} × ${M2}  |  Episode: ${EPISODE}"
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
