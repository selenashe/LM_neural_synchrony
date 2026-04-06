#!/usr/bin/env bash
#SBATCH --job-name=fix_labels
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/fix_labels_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/fix_labels_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

echo "=== Job started at $(date) ==="

python fix_parse_errors.py

echo ""
echo "=== Job finished at $(date) ==="
