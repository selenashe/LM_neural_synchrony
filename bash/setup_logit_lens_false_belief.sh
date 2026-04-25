#!/usr/bin/env bash
#SBATCH --job-name=ll_fb_setup
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_setup_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_setup_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=3:30:00
#SBATCH --array=0-63

# One array task = one (model_1, model_2) pair.
# Pairs without all 600 episodes are skipped automatically.
# Re-submitting after more simulation data is safe: already-created
# behavioral summaries are skipped by the Python script.

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

PAIRS_FILE="${REPO_ROOT}/bash/pairs_false_belief_45.bash"
if [[ ! -f "${PAIRS_FILE}" ]]; then
  echo "ERROR: missing ${PAIRS_FILE}" >&2
  exit 1
fi
source "${PAIRS_FILE}"

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-63)}
model_1=${MODEL_1[$i]}
model_2=${MODEL_2[$i]}
if [[ -z "${model_1}" || -z "${model_2}" ]]; then
  echo "ERROR: empty model pair at array index ${i}" >&2
  exit 1
fi

echo "=== Task ${i} / 0-63 started at $(date) ==="
echo "Pair: ${model_1}  x  ${model_2}"

pair_dir="${REPO_ROOT}/sotopia_results_false_belief_100/dialogs/${model_1}_None_0_${model_2}_None_0_false_belief_fixed_two_agents"
n_complete=$(find "${pair_dir}" -maxdepth 1 -name '*_temp*.csv' -type f 2>/dev/null | wc -l)
if [[ "${n_complete}" -lt 600 ]]; then
  echo "Skipping: only ${n_complete}/600 episodes complete for this pair."
  exit 0
fi

python analysis/setup_logit_lens_false_belief.py \
    --model_1 "${model_1}" \
    --model_2 "${model_2}" \
    --results_dir sotopia_results_false_belief_100

echo ""
echo "=== Task ${i} finished at $(date) ==="
