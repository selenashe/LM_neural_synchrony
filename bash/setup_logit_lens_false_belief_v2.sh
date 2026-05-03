#!/usr/bin/env bash
#SBATCH --job-name=ll_fb_v2_setup
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v2_setup_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v2_setup_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --array=0-7

# One array task = one self-pair (modelA x modelA). 8 models.
# V2 false-belief scenarios (C1-C6 redesigned conditions).
# Re-submitting is safe: already-created behavioral summaries are skipped.

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

MODELS=(
  'allenai_OLMo-3-7B-Instruct'
  'DeepSeek-R1-Distill-Llama-8B'
  'Meta-Llama-3-8B-Instruct'
  'Mistral-7B-Instruct-v0.2'
  'Mistral-7B-Instruct-v0.3'
  'Qwen2.5-7B-Instruct'
  'Qwen3-8B'
  'Qwen_Qwen3-14B'
)

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-7)}
model_1=${MODELS[$i]}
model_2=${MODELS[$i]}
if [[ -z "${model_1}" ]]; then
  echo "ERROR: empty model at array index ${i}" >&2
  exit 1
fi

pair_dir="${REPO_ROOT}/sotopia_results_false_belief_v2/dialogs/${model_1}_None_0_${model_2}_None_0_false_belief_v2"
n_complete=$(find "${pair_dir}" -maxdepth 1 -name '*_temp*.csv' -type f 2>/dev/null | wc -l)
echo "Found ${n_complete} episode files for ${model_1} x ${model_2}"

if [[ "${n_complete}" -lt 600 ]]; then
  echo "WARNING: only ${n_complete}/600 episodes complete."
fi

echo "=== Task ${i} / 0-7 started at $(date) ==="
echo "Pair: ${model_1}  x  ${model_2}"

python analysis/setup_logit_lens_false_belief_v2.py \
    --model_1 "${model_1}" \
    --model_2 "${model_2}" \
    --results_dir sotopia_results_false_belief_v2 \
    --output_dir logit_lens_results_false_belief_v2

echo ""
echo "=== Task ${i} finished at $(date) ==="
