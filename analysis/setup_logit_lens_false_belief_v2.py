#!/usr/bin/env python3
"""
Phase 1: Create output directories and save behavioral summaries
for all model pairs x episodes. No GPU required.

V2 false-belief scenarios (C1-C6 redesigned conditions).

Run this once before submitting the Phase 2 SLURM array
(run_logit_lens_false_belief.sh).
"""

import argparse
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, SCRIPT_DIR)

from logit_lens_false_belief import (
    REPO_ROOT,
    load_scenario_meta,
    load_dialog_record,
    infer_agent_names,
    build_behavioral_summary_fb,
    save_json,
)

V2_ENVS_BASENAME = "envs_false_belief_v2.json"
V2_COMBOS_BASENAME = "env_agent_combos_false_belief_v2_fixed_two_agents.json"


def parse_pairs_file(path):
    """Parse MODEL_1 and MODEL_2 arrays from bash pairs file."""
    with open(path) as f:
        text = f.read()

    def extract_array(name):
        m = re.search(rf"{name}=\(\s*(.*?)\s*\)", text, re.DOTALL)
        if not m:
            raise ValueError(f"Could not find {name} array in {path}")
        return re.findall(r"'([^']+)'", m.group(1))

    m1 = extract_array("MODEL_1")
    m2 = extract_array("MODEL_2")
    assert len(m1) == len(m2), f"MODEL_1 ({len(m1)}) and MODEL_2 ({len(m2)}) lengths differ"
    return list(zip(m1, m2))


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: create result dirs and behavioral summaries for v2 false-belief (no GPU)."
    )
    parser.add_argument("--model_1", type=str, default=None,
                        help="Process a single pair (use with --model_2).")
    parser.add_argument("--model_2", type=str, default=None,
                        help="Process a single pair (use with --model_1).")
    parser.add_argument("--pairs_file", type=str,
                        default=os.path.join(REPO_ROOT, "bash", "pairs_false_belief_45.bash"))
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Number of episodes. Default: auto-detect from envs JSON.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--results_dir", type=str, default="sotopia_results_false_belief_v2")
    parser.add_argument("--output_dir", type=str, default="logit_lens_results_false_belief_v2",
                        help="Output directory name (relative to REPO_ROOT).")
    parser.add_argument("--envs_basename", type=str, default=V2_ENVS_BASENAME)
    parser.add_argument("--combos_basename", type=str, default=V2_COMBOS_BASENAME)
    parser.add_argument("--run_label", type=str, default="false_belief_v2",
                        help="Suffix used in pair directory names.")
    args = parser.parse_args()

    if args.n_episodes is None:
        import json
        envs_path = os.path.join(REPO_ROOT, "sotopia_utils", "sotopia_data", args.envs_basename)
        with open(envs_path, encoding="utf-8") as f:
            args.n_episodes = len(json.load(f))

    if args.model_1 and args.model_2:
        pairs = [(args.model_1, args.model_2)]
    else:
        pairs = parse_pairs_file(args.pairs_file)
    print(f"Found {len(pairs)} model pairs, {args.n_episodes} episodes each")
    print(f"Total: {len(pairs) * args.n_episodes} directories\n")

    created = 0
    skipped = 0
    errors = 0

    for pair_idx, (model_1, model_2) in enumerate(pairs):
        m1_full = f"{model_1}_None_0"
        m2_full = f"{model_2}_None_0"
        pair_name = f"{m1_full}_{m2_full}_{args.run_label}"

        for episode in range(args.n_episodes):
            try:
                meta = load_scenario_meta(
                    episode,
                    envs_basename=args.envs_basename,
                    combos_basename=args.combos_basename,
                )
                out_dir = os.path.join(
                    REPO_ROOT, args.output_dir, pair_name,
                    f"episode_{episode}_{meta['codename']}",
                )
                os.makedirs(out_dir, exist_ok=True)

                behavioral_path = os.path.join(out_dir, "behavioral_summary.json")
                if os.path.exists(behavioral_path):
                    skipped += 1
                    continue

                dialog_record = load_dialog_record(
                    pair_name, episode, args.seed, args.temp,
                    results_dir=args.results_dir,
                )
                agent1_name, agent2_name = infer_agent_names(dialog_record)
                behavioral = build_behavioral_summary_fb(
                    dialog_record, agent1_name, agent2_name, meta,
                )
                save_json(behavioral_path, behavioral)
                created += 1

            except Exception as e:
                print(f"  ERROR pair {pair_idx} ({model_1} x {model_2}) ep {episode}: {e}")
                errors += 1

    print(f"\nDone. Created: {created}, Skipped (existing): {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
