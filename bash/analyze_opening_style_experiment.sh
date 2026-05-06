#!/usr/bin/env bash
#SBATCH --job-name=os_analyze
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/os_analyze_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/os_analyze_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

# Analyze opening-style experiment results.
# Usage: sbatch bash/analyze_opening_style_experiment.sh
#    or: bash bash/analyze_opening_style_experiment.sh

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

echo "=== Analyze opening style experiment: started at $(date) ==="
python analysis/analyze_opening_style_experiment.py \
    --results_csv opening_style_experiment_results/opening_style_experiment_results.csv \
    --output_dir opening_style_experiment_results
echo ""

echo "=== All done at $(date) ==="
