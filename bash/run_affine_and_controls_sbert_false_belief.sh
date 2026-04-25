#!/usr/bin/env bash
#SBATCH --job-name=affine_sbert_fb
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_sbert_false_belief_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_sbert_false_belief_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
# Must match simulation outputs (sample_normal_agent / affine read states & dialogs here).
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_100

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs
cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit 1

echo "=== Affine + SBERT (false-belief 600 ep, Mistral-only) started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_ENVS_BASENAME=${SOTOPIA_ENVS_BASENAME}"
echo "SOTOPIA_RESULTS_DIR=${SOTOPIA_RESULTS_DIR}"
python -c "from experiment_config import RESULTS_DIR; print('RESULTS_DIR (resolved):', RESULTS_DIR); import os; print('Dialogs:', os.path.join(RESULTS_DIR, 'dialogs'))"
echo ""

SUF="_false_belief_fixed_two_agents"

echo "--- affine_transformation.py (2×2 Mistral pairs) ---"
for pair in \
  "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0" \
  "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0" \
  "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0" \
  "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0"
do
  echo "===== ${pair}${SUF} ====="
  python analysis/affine_transformation.py --model "${pair}${SUF}" --setting A_forward --method cka_cca
done

echo ""
echo "--- analyze_controls_sbert.py (--all_pairs --mistral_only_pairs) ---"
python analysis/analyze_controls_sbert.py \
  --all_pairs \
  --mistral_only_pairs \
  --results_postfix "${SUF}" \
  --output_tag false_belief_600ep

echo ""
echo "=== Finished at $(date) ==="
