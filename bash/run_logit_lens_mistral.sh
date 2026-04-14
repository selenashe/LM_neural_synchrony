#!/usr/bin/env bash
#SBATCH --job-name=logit_lens
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-3

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

# Mistral-7B shards are large; default ~/.cache/huggingface on AFS often hits quota.
# HF_HOME should be the *huggingface* root (contains hub/). Override before sbatch if needed.
export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

echo "=== Job started at $(date) ==="
echo "Node: $(hostname), Task ID: ${SLURM_ARRAY_TASK_ID}"
echo ""

# 4 Mistral model pairs — each job analyses both agents together
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

M1="${MODEL_1_LIST[$SLURM_ARRAY_TASK_ID]}"
M2="${MODEL_2_LIST[$SLURM_ARRAY_TASK_ID]}"

echo "Model pair: ${M1} × ${M2}"

python logit_lens_mistral.py \
    --model_1 "${M1}" \
    --model_2 "${M2}" \
    --episode 6 \
    --seed 0 \
    --temp 0.7 \
    --hf-cache "${HF_HOME}"

echo ""
echo "=== Job finished at $(date) ==="
