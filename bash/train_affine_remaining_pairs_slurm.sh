#!/usr/bin/env bash
#SBATCH --job-name=affine_v02_first
#SBATCH --account=nlp
#SBATCH --output=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_remaining_%A_%a.out
#SBATCH --error=/juice6/u/jshe/nlp/LM_neural_synchrony/logs/affine_remaining_%A_%a.err
#SBATCH --partition=sphinx
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#
# Runs the four affine_transformation.py jobs that populate Misv0.2Misv0.3 and Misv0.2Misv0.2
# (A_forward + B_forward for each), in parallel across array tasks.
#
# Prerequisite: full 32×32 grids already exist for Misv0.3Misv0.3 and Misv0.3Misv0.2.
# Check before submitting:
#   bash bash/verify_affine_layer_grid.sh
#
# Submit (from repo root recommended):
#   cd /juice6/u/jshe/nlp/LM_neural_synchrony && sbatch bash/train_affine_remaining_pairs_slurm.sh
#
# Note: Slurm executes a *copy* of this script under /var/spool/slurm/..., so $(dirname "$0") is NOT
# the repo. REPO is taken from SLURM_SUBMIT_DIR (cwd when you ran sbatch) or the fallback path below.
#
# Walltime: 48h per task (same order as svd_v2_cv.sh). If a task times out, resubmit only
# that index, e.g. sbatch --array=2 bash/train_affine_remaining_pairs_slurm.sh
set -euo pipefail

source /nlp/scr/jshe/miniconda3/etc/profile.d/conda.sh
conda activate neural_sync

REPO_FALLBACK="/juice6/u/jshe/nlp/LM_neural_synchrony"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/affine_transformation.py" ]]; then
  REPO="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -f "${REPO_FALLBACK}/affine_transformation.py" ]]; then
  REPO="${REPO_FALLBACK}"
else
  echo "Cannot find repo: no affine_transformation.py in SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} or ${REPO_FALLBACK}" >&2
  exit 1
fi

mkdir -p "${REPO}/logs"
cd "${REPO}" || exit 1
echo "REPO=${REPO}  PWD=$(pwd)"

export SOTOPIA_REFERENCE_RUN="${SOTOPIA_REFERENCE_RUN:-Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0}"

echo "=== Task ${SLURM_ARRAY_TASK_ID} started at $(date) on $(hostname) ==="

# Cheap sanity check (Misv0.3× pairs must be complete).
# Keep this inline so the batch job does not depend on a separate helper file.
python3 - <<'PY'
import os
import sys

repo = os.getcwd()
affine = os.path.join(repo, "affine_transformation")
tags = ("Misv0.3Misv0.3", "Misv0.3Misv0.2")
settings = ("A_forward", "B_forward")
n = 32
suffix = "[0, 1, 2, 3, 4]"

def path(setting, tag, la, lb):
    fn = f"combined_metrics_{tag}{suffix}_layerA{la}_layerB{lb}_allreps.json"
    return os.path.join(affine, setting, f"layerA{la}", "data", fn)

bad = False
for tag in tags:
    for setting in settings:
        missing = []
        for la in range(n):
            for lb in range(n):
                if not os.path.isfile(path(setting, tag, la, lb)):
                    missing.append((la, lb))
        total = n * n
        ok = total - len(missing)
        print(f"{tag} / {setting}: {ok}/{total} present")
        if missing:
            print(f"First missing entries: {missing[:5]}", file=sys.stderr)
            bad = True

if bad:
    sys.exit("Baseline affine grids are incomplete; aborting remaining-pairs job.")
PY

M23="Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0"
M22="Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    echo "Running A_forward for v0.2 (A) × v0.3 (B)"
    python affine_transformation.py --model "${M23}" --setting A_forward
    ;;
  1)
    echo "Running B_forward for v0.2 (A) × v0.3 (B)"
    python affine_transformation.py --model "${M23}" --setting B_forward
    ;;
  2)
    echo "Running A_forward for v0.2 × v0.2"
    python affine_transformation.py --model "${M22}" --setting A_forward
    ;;
  3)
    echo "Running B_forward for v0.2 × v0.2"
    python affine_transformation.py --model "${M22}" --setting B_forward
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "=== Task ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
