#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run v4.5 2×2×2 factorial with Gemini via Vertex AI — locally, all 8 conditions.
#
# Usage:
#   bash bash/run_gemini_v4_5_local.sh                    # full 100-item, label=full
#   bash bash/run_gemini_v4_5_local.sh --label pilot      # pilot 10-item
#   bash bash/run_gemini_v4_5_local.sh --max_episodes 5   # smoke test
# ---------------------------------------------------------------------------
set -euo pipefail

# Conda activation (skip if already in correct env)
if [ -f /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh ]; then
    source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
    conda activate neural_sync
fi

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

mkdir -p "${REPO_ROOT}/logs"

# ── Defaults; can be overridden via env vars ────────────────────────────────
VERTEX_MODEL="${VERTEX_MODEL:-vertex_ai/gemini-2.5-pro-preview-05-06}"
MODEL_KEY="${MODEL_KEY:-gemini}"
ENVS="${ENVS:-envs_false_belief_v4_1.json}"
LABEL="${LABEL:-full}"

CONDITIONS=(
  'v4_5_single_sys_quote'
  'v4_5_single_sys_report'
  'v4_5_single_uo_quote'
  'v4_5_single_uo_report'
  'v4_5_multi_sys_quote'
  'v4_5_multi_sys_report'
  'v4_5_multi_uo_quote'
  'v4_5_multi_uo_report'
)

echo "=== Gemini v4.5 local run started at $(date) ==="
echo "Vertex model: ${VERTEX_MODEL}"
echo "Model key:    ${MODEL_KEY}"
echo "Envs:         ${ENVS}"
echo "Label:        ${LABEL}"
echo ""

for condition in "${CONDITIONS[@]}"; do
    echo "--- Starting condition: ${condition} at $(date) ---"
    python simulation/sample_gemini_v4_5_factorial.py \
        --condition "${condition}" \
        --envs "${ENVS}" \
        --label "${LABEL}" \
        --model_key "${MODEL_KEY}" \
        --vertex_model "${VERTEX_MODEL}" \
        "$@"
    echo "--- Finished condition: ${condition} at $(date) ---"
    echo ""
done

echo "=== All conditions done at $(date) ==="
