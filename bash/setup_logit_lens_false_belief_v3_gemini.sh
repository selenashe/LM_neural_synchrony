#!/usr/bin/env bash
#SBATCH --job-name=ll_fb_v3_gem
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v3_gemini_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v3_gemini_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

# Single model pair: gemini-3.1-pro-preview x gemini-3.1-pro-preview
# V3 false-belief scenarios.
# No array needed — one job processes all 2400 episodes.

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

model_1="gemini-3.1-pro-preview"
model_2="gemini-3.1-pro-preview"

pair_dir="${REPO_ROOT}/sotopia_results_false_belief_v3_gemini/dialogs/${model_1}_None_0_${model_2}_None_0_false_belief_v3"
n_complete=$(find "${pair_dir}" -maxdepth 1 -name '*_temp*.csv' -type f 2>/dev/null | wc -l)
echo "Found ${n_complete} episode files for ${model_1} x ${model_2}"

if [[ "${n_complete}" -lt 2400 ]]; then
  echo "WARNING: only ${n_complete}/2400 episodes complete."
fi

echo "=== Started at $(date) ==="
echo "Pair: ${model_1}  x  ${model_2}"

python analysis/setup_logit_lens_false_belief_v2.py \
    --model_1 "${model_1}" \
    --model_2 "${model_2}" \
    --results_dir sotopia_results_false_belief_v3_gemini \
    --output_dir logit_lens_results_false_belief_v3_gemini \
    --envs_basename envs_false_belief_v3.json \
    --combos_basename env_agent_combos_false_belief_v3_fixed_two_agents.json \
    --run_label false_belief_v3

echo ""
echo "=== Finished at $(date) ==="
