#!/usr/bin/env bash
#SBATCH --job-name=sim_fb_v3_resume
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_fb_v3_resume_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_fb_v3_resume_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-2

# Resume the three slow model pairs for false-belief v3.
# The simulation script auto-skips existing episodes, so this just
# picks up where the previous run left off.
#
# Estimated time per model (single GPU):
#   DeepSeek-R1-Distill-Llama-8B : ~298s/ep × 2218 remaining ≈ 183 hrs
#   Qwen3-8B                     : ~147s/ep × 2035 remaining ≈  83 hrs
#   Qwen_Qwen3-14B               : ~288s/ep × 2216 remaining ≈ 177 hrs
#
# With --time=48:00:00 each batch completes roughly:
#   DeepSeek : ~580 episodes/batch  → 4 batches to finish
#   Qwen3-8B : ~1170 episodes/batch → 2 batches to finish
#   Qwen3-14B: ~600 episodes/batch  → 4 batches to finish
#
# Re-submit this script after each batch until all 2400 episodes exist:
#   sbatch bash/run_simulate_false_belief_v3_resume.sh
#
# Check progress:
#   for d in DeepSeek-R1-Distill-Llama-8B Qwen3-8B Qwen_Qwen3-14B; do
#     echo "$d: $(ls sotopia_results_false_belief_v3/dialogs/${d}_None_0_${d}_None_0_false_belief_v3/ | wc -l) / 2400"
#   done

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_v3_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief_v3.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_v3
export SOTOPIA_RUN_LABEL=false_belief_v3

MODELS=(
  'DeepSeek-R1-Distill-Llama-8B'
  'Qwen3-8B'
  'Qwen_Qwen3-14B'
)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-2)}
model=${MODELS[$i]}
if [[ -z "${model}" ]]; then
  echo "ERROR: empty model at array index ${i}" >&2
  exit 1
fi

DIALOG_DIR="${REPO_ROOT}/sotopia_results_false_belief_v3/dialogs/${model}_None_0_${model}_None_0_false_belief_v3"
done_count=$(ls "${DIALOG_DIR}" 2>/dev/null | wc -l)

echo "=== Resume job array task ${i} / 0-2 started at $(date) ==="
echo "Node: $(hostname)"
echo "Model: ${model}"
echo "Episodes completed so far: ${done_count} / 2400"
echo ""

python simulation/sample_normal_agent.py --model_1 "${model}" --model_2 "${model}"

done_after=$(ls "${DIALOG_DIR}" 2>/dev/null | wc -l)
echo ""
echo "=== Task ${i} finished at $(date) ==="
echo "Episodes now: ${done_after} / 2400"
