#!/usr/bin/env bash
#SBATCH --job-name=svd_all_L
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/svd_all_layers_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/svd_all_layers_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

export SOTOPIA_REFERENCE_RUN="Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo ""

python analysis/analyze_svd.py --all_pairs --k 50 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --output_prefix svd_alllayers

echo ""
echo "=== Job finished at $(date) ==="
