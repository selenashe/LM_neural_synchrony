#!/usr/bin/env bash
#SBATCH --job-name=affine_sbert_mia_ava
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_sbert_mia_ava_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_sbert_mia_ava_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Aligns with env_agent_combos_fixed_two_agents.json (90 episodes) + Mia/Ava result folders.
export SOTOPIA_ENV_AGENT_COMBOS_BASENAME="${SOTOPIA_ENV_AGENT_COMBOS_BASENAME:-env_agent_combos_fixed_two_agents.json}"
# Prefer *mia_ava_fixed_two_agents under sotopia_results/ for episode listing (optional; also auto when EPISODES_NUM<=120).
export SOTOPIA_REFERENCE_SUFFIX="${SOTOPIA_REFERENCE_SUFFIX:-_mia_ava_fixed_two_agents}"

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs
cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit 1

echo "=== Affine + SBERT (Mia/Ava 90 ep) started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_REFERENCE_SUFFIX=${SOTOPIA_REFERENCE_SUFFIX}"
echo ""

echo "--- affine_transformation.py (same 3×3 model pairs as run_simulate_and_save_states) ---"
SUF="_mia_ava_fixed_two_agents"
for pair in \
  "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0" \
  "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0" \
  "Mistral-7B-Instruct-v0.3_None_0_Meta-Llama-3-8B-Instruct_None_0" \
  "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0" \
  "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0" \
  "Mistral-7B-Instruct-v0.2_None_0_Meta-Llama-3-8B-Instruct_None_0" \
  "Meta-Llama-3-8B-Instruct_None_0_Mistral-7B-Instruct-v0.3_None_0" \
  "Meta-Llama-3-8B-Instruct_None_0_Mistral-7B-Instruct-v0.2_None_0" \
  "Meta-Llama-3-8B-Instruct_None_0_Meta-Llama-3-8B-Instruct_None_0"
do
  echo "===== ${pair}${SUF} ====="
  python affine_transformation.py --model "${pair}${SUF}" --setting A_forward
done

echo ""
echo "--- analyze_controls_sbert.py (--all_pairs) ---"
python analyze_controls_sbert.py --all_pairs --output_tag mia_ava_90ep

echo ""
echo "=== Finished at $(date) ==="
