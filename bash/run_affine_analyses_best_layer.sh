#!/usr/bin/env bash
#SBATCH --job-name=affine_best
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_best_%j.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_best_%j.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

cd /juice6/u/jshe/nlp/LM_neural_synchrony || exit

export SOTOPIA_REFERENCE_RUN="Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0"

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo ""

echo "===== analyze_alignment.py (best layer) ====="
python analyze_alignment.py --all_pairs
echo ""

echo "===== analyze_stratified.py — Pathway A+C (best layer) ====="
python analyze_stratified.py --all_pairs --pathway both
echo ""

echo "===== analyze_controls.py (best layer) ====="
python analyze_controls.py --all_pairs --n_perms 100
echo ""

echo "===== analyze_pathway_d.py (best layer) ====="
python analyze_pathway_d.py --all_pairs
echo ""

echo "=== Job finished at $(date) ==="
