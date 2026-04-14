#!/usr/bin/env bash
# Single pair: Mistral-7B-Instruct-v0.2 × Mistral-7B-Instruct-v0.2 (same as array task 3 in run_logit_lens_mistral.sh)
#SBATCH --job-name=logit_lens_v02v02
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_v02v02_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/logit_lens_v02v02_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:h100:1
#SBATCH--mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

export HF_HOME="${HF_HOME:-/juice6/scr6/jshe/.cache/huggingface}"
mkdir -p "${HF_HOME}"

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "Pair: Mistral-7B-Instruct-v0.2 × Mistral-7B-Instruct-v0.2"
echo ""

python logit_lens_mistral.py \
    --model_1 Mistral-7B-Instruct-v0.2 \
    --model_2 Mistral-7B-Instruct-v0.2 \
    --episode 6 \
    --seed 0 \
    --temp 0.7 \
    --hf-cache "${HF_HOME}"

echo ""
echo "=== Job finished at $(date) ==="
