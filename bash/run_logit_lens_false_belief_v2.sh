#!/usr/bin/env bash
#SBATCH --job-name=ll_fb_v2
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v2_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_v2_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --array=0-4799
# NOTE: 8 self-pairs × 600 episodes = 4800 tasks total.
# If MaxArraySize < 4800, submit in batches:
#   BATCH_OFFSET=0    sbatch run_logit_lens_false_belief_v2.sh   (tasks 0-4799)

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

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

N_EPISODES=600
GLOBAL_ID=$(( ${BATCH_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
PAIR_IDX=$(( GLOBAL_ID / N_EPISODES ))
EPISODE=$(( GLOBAL_ID % N_EPISODES ))

M1="${MODELS[$PAIR_IDX]}"
M2="${MODELS[$PAIR_IDX]}"
if [[ -z "${M1}" ]]; then
  echo "ERROR: empty model at pair_idx ${PAIR_IDX} (index out of range?)" >&2
  exit 1
fi

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Pair ${PAIR_IDX}: ${M1} × ${M2}  |  Episode: ${EPISODE}"
echo ""

python analysis/logit_lens_false_belief.py \
    --model_1 "${M1}" \
    --model_2 "${M2}" \
    --episode "${EPISODE}" \
    --seed 0 \
    --temp 0.7 \
    --hf-cache "${HF_HOME}" \
    --results_dir sotopia_results_false_belief_v2 \
    --output_dir logit_lens_results_false_belief_v2 \
    --envs_basename envs_false_belief_v2.json \
    --combos_basename env_agent_combos_false_belief_v2_fixed_two_agents.json \
    --run_label false_belief_v2

echo ""
echo "=== Job finished at $(date) ==="
