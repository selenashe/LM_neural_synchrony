#!/usr/bin/env bash
# Run all Deal-or-No-Deal analysis scripts.
# Outputs go to PROJECT_ROOT/summary_plots_deal_or_no_deal/
#
# Usage:
#   bash bash/run_dond_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/nlp/scr/jshe/miniconda3/envs/neural_sync/bin/python"
ANALYSIS="$PROJECT_ROOT/analysis"

echo "=== Deal-or-No-Deal Analysis Pipeline ==="
echo "Project root: $PROJECT_ROOT"
echo ""

echo "[1/5] FAO analysis + outcome metrics..."
$PYTHON "$ANALYSIS/fao_analysis.py"
echo ""

echo "[2/5] Per-model Pareto gap plots..."
$PYTHON "$ANALYSIS/fao_per_model_plot.py"
echo ""

echo "[3/5] Quality audit..."
$PYTHON "$ANALYSIS/audit_dond_quality.py"
echo ""

echo "[4/5] Error case extraction..."
$PYTHON "$ANALYSIS/extract_dond_error_cases.py"
echo ""

echo "[5/5] Speech act tagging..."
$PYTHON "$ANALYSIS/dond_speech_acts.py"
echo ""

echo "=== All done. Outputs in $PROJECT_ROOT/summary_plots_deal_or_no_deal/ ==="
