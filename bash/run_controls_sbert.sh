#!/usr/bin/env bash
#SBATCH --job-name=ctrl_sbert
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/controls_sbert_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/controls_sbert_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

export SOTOPIA_REFERENCE_RUN="Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo ""

python analyze_controls_sbert.py --all_pairs

echo ""
echo "=== Job finished at $(date) ==="
