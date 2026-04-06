"""
Re-annotate the PARSE_ERROR entries in goal_alignment_labels.json using
Gemini, then rebuild goal_alignment_labels_v2.json.

Usage:
    python fix_parse_errors.py [--model vertex_ai/gemini-3.1-pro-preview]
"""

import os
import json
import argparse
from goal_alignment_labels import (
    classify_combo, RELATIONSHIP_LABELS, clean_tags,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/envs.json")
COMBOS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/env_agent_combos.json")
AGENTS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/agents.json")
V1_PATH = os.path.join(SCRIPT_DIR, "goal_alignment_labels.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vertex_ai/gemini-3.1-pro-preview")
    args = parser.parse_args()

    with open(ENVS_PATH) as f:
        envs = json.load(f)
    with open(COMBOS_PATH) as f:
        combos = json.load(f)
    with open(AGENTS_PATH) as f:
        agents = json.load(f)
    with open(V1_PATH) as f:
        labels = json.load(f)

    missing = [k for k, v in labels.items() if v.get("alignment_score") is None]
    print(f"Found {len(missing)} entries with missing alignment_score")

    fixed = 0
    still_broken = 0

    for combo_key in sorted(missing, key=int):
        i = int(combo_key)
        combo = combos[i]
        env = envs[combo["env_id"]]
        agent1 = agents[combo["agent_ids"][0]]
        agent2 = agents[combo["agent_ids"][1]]
        relationship = RELATIONSHIP_LABELS.get(env.get("relationship", 0), "unknown")

        print(f"  Re-annotating combo {i} ({env['source']}/{env['codename']})...")

        result = classify_combo(
            args.model,
            env["scenario"],
            relationship,
            agent1, agent2,
            env["agent_goals"][0],
            env["agent_goals"][1],
            max_retries=5,
        )

        if result["alignment_score"] is not None:
            labels[combo_key].update({
                "alignment_score": result["alignment_score"],
                "goal_structure": result["goal_structure"],
                "value_compatibility": result["value_compatibility"],
                "power_dynamic": result["power_dynamic"],
                "stakes": result["stakes"],
                "reasoning": result["reasoning"],
            })
            fixed += 1
        else:
            still_broken += 1
            print(f"    STILL FAILED: {result.get('reasoning', '?')}")

        if fixed % 10 == 0 and fixed > 0:
            with open(V1_PATH, "w") as f:
                json.dump(labels, f, indent=2)
            print(f"    Checkpoint ({fixed} fixed so far)")

    with open(V1_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone. Fixed {fixed}/{len(missing)}, still broken: {still_broken}")
    print(f"Updated: {V1_PATH}")

    print("\nRebuilding v2 labels...")
    os.system(f"python3 {os.path.join(SCRIPT_DIR, 'build_labels_v2.py')}")


if __name__ == "__main__":
    main()
