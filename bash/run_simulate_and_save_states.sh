#!/usr/bin/env bash
#SBATCH --job-name=sim_states
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/simulate_states_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --constraint=80G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --array=0-63

# One array task = one (model_1, model_2) pair. 64 pairs = 28 cross + 28 reverse + 8 self.
# from scripts/list_false_belief_pairs.py (case-insensitive lexicographic model names).
# Submit: sbatch bash/run_simulate_and_save_states.sh
# Regenerate pair arrays: python scripts/list_false_belief_pairs.py --bash-export > bash/pairs_false_belief_45.bash

export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_100
export SOTOPIA_RUN_LABEL=false_belief_fixed_two_agents

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

mkdir -p /juice6/u/jshe/nlp/LM_neural_synchrony/logs

REPO_ROOT="${REPO_ROOT:-/juice6/u/jshe/nlp/LM_neural_synchrony}"
cd "${REPO_ROOT}" || exit 1

# Slurm copies the batch script under /var/lib/slurm/... — dirname(BASH_SOURCE) is not the repo.
PAIRS_FILE="${REPO_ROOT}/bash/pairs_false_belief_45.bash"
if [[ ! -f "${PAIRS_FILE}" ]]; then
  echo "ERROR: missing ${PAIRS_FILE}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${PAIRS_FILE}"

i=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — submit with sbatch (array 0-63)}
model_1=${MODEL_1[$i]}
model_2=${MODEL_2[$i]}
if [[ -z "${model_1}" || -z "${model_2}" ]]; then
  echo "ERROR: empty model pair at array index ${i} (MODEL_1/MODEL_2 not set?)" >&2
  exit 1
fi

echo "=== Job array task ${i} / 0-63 started at $(date) ==="
echo "Node: $(hostname)"
echo "SOTOPIA_ENV_AGENT_COMBOS_BASENAME=${SOTOPIA_ENV_AGENT_COMBOS_BASENAME}"
echo "SOTOPIA_ENVS_BASENAME=${SOTOPIA_ENVS_BASENAME}"
echo "SOTOPIA_RUN_LABEL=${SOTOPIA_RUN_LABEL}"
echo "Pair: ${model_1}  x  ${model_2}"
echo ""

python simulation/sample_normal_agent.py --model_1 "${model_1}" --model_2 "${model_2}"

echo ""
echo "=== Task ${i} finished at $(date) ==="
