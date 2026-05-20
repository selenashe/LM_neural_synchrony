#!/usr/bin/env bash
#SBATCH --job-name=olmo31_data_audit
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_data_audit_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/olmo31_data_audit_%j.err
#SBATCH --partition=sphinx
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00

# Tier 1 structural analysis of OLMo 3.1 32B post-training datasets.
# No GPU needed — CPU-only data analysis with pandas/pyarrow.
# Submit: sbatch bash/run_analyze_olmo31_datasets.sh

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

cd /juice6/u/jshe/nlp/LM_neural_synchrony

python -u data_audit/analyze_olmo31_datasets.py \
    --output data_audit/tier1_results.txt
