"""
Shared settings for this checkout. See REPO_LOCAL_CHANGES.txt for rationale.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SOTOPIA_DATA_DIR = os.path.join(REPO_ROOT, "sotopia_utils", "sotopia_data")
# Override with env var to use e.g. env_agent_combos_fixed_two_agents.json (90 episodes).
ENV_AGENT_COMBOS_BASENAME = os.environ.get(
    "SOTOPIA_ENV_AGENT_COMBOS_BASENAME",
    "env_agent_combos_fixed_two_agents.json",
)
ENV_AGENT_COMBOS_PATH = os.path.join(SOTOPIA_DATA_DIR, ENV_AGENT_COMBOS_BASENAME)
# Alias for scripts that refer to the combo list path as COMBOS_PATH.
COMBOS_PATH = ENV_AGENT_COMBOS_PATH

ENVS_BASENAME = os.environ.get("SOTOPIA_ENVS_BASENAME", "envs.json")
ENVS_PATH = os.path.join(SOTOPIA_DATA_DIR, ENVS_BASENAME)

RESULTS_DIR_NAME = os.environ.get("SOTOPIA_RESULTS_DIR", "sotopia_results")
RESULTS_DIR = os.path.join(REPO_ROOT, RESULTS_DIR_NAME)

with open(ENV_AGENT_COMBOS_PATH, "r", encoding="utf-8") as _f:
    EPISODES_NUM = len(json.load(_f))

# Seeds for Stage A simulation and matching downstream eval loops. Original: [0, 1, 2, 3, 4].
STAGE_A_SEEDS = [0]

# Folder suffix under sotopia_results/ for Mia+Ava fixed-agent (90-episode) runs.
MIA_AVA_RESULTS_POSTFIX = "_mia_ava_fixed_two_agents"
