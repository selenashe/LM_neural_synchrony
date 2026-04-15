#!/usr/bin/env bash
#SBATCH --job-name=ll_fb
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_fb_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-71

# 4 model pairs × 18 episodes = 72 tasks
# task_id = pair_idx * 18 + episode

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit 1

MODEL_1_LIST=(
    Mistral-7B-Instruct-v0.3
    Mistral-7B-Instruct-v0.3
    Mistral-7B-Instruct-v0.2
    Mistral-7B-Instruct-v0.2
)
MODEL_2_LIST=(
    Mistral-7B-Instruct-v0.3
    Mistral-7B-Instruct-v0.2
    Mistral-7B-Instruct-v0.3
    Mistral-7B-Instruct-v0.2
)

PAIR_IDX=$(( SLURM_ARRAY_TASK_ID / 18 ))
EPISODE=$(( SLURM_ARRAY_TASK_ID % 18 ))

M1="${MODEL_1_LIST[$PAIR_IDX]}"
M2="${MODEL_2_LIST[$PAIR_IDX]}"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Pair ${PAIR_IDX}: ${M1} × ${M2}  |  Episode: ${EPISODE}"
echo ""

python logit_lens_false_belief.py \
    --model_1 "${M1}" \
    --model_2 "${M2}" \
    --episode "${EPISODE}" \
    --seed 0 \
    --temp 0.7 \
    --hf-cache "${HF_HOME}" \
    --results_dir sotopia_results_false_belief

echo ""
echo "=== Job finished at $(date) ==="
