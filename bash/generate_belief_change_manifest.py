#!/usr/bin/env python3
"""
Generate a task manifest for logit-lens analysis restricted to
belief-change episodes:
  - false_help  where correct == 1  (seeker was persuaded to correct location)
  - false_deceive where correct == 0 (seeker was persuaded to wrong location)

Reads behavioral_individual.csv and the episode directory structure to produce
a TSV manifest of (model_1, model_2, episode_index) triples.

Only includes pairs that have all 600 episodes present in logit_lens_results.

Usage:
    python bash/generate_belief_change_manifest.py \
        --behavioral_csv summary_plots_false_belief_100/behavioral_individual.csv \
        --logit_lens_dir logit_lens_results_false_belief_100 \
        --output bash/logit_lens_belief_change_manifest.tsv
"""

import argparse
import csv
import os
import re
import sys


def build_scenario_episode_map(sample_pair_dir: str) -> dict:
    """Map (scenario_name, condition) -> episode_index from directory names."""
    ep_map = {}
    for name in os.listdir(sample_pair_dir):
        m = re.match(r"episode_(\d+)_(.+)", name)
        if not m:
            continue
        ep_idx = int(m.group(1))
        rest = m.group(2)
        parts = rest.rsplit("_", 2)
        if len(parts) == 3:
            scenario, cond_a, cond_b = parts
            condition = f"{cond_a}_{cond_b}"
        else:
            continue
        ep_map[(scenario, condition)] = ep_idx
    return ep_map


def parse_pair(pair_str: str):
    parts = pair_str.split(" × ")
    return parts[0].strip(), parts[1].strip()


def pair_to_dirname(m1: str, m2: str) -> str:
    return f"{m1}_None_0_{m2}_None_0_false_belief_fixed_two_agents"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral_csv", default="summary_plots_false_belief_100/behavioral_individual.csv")
    parser.add_argument("--logit_lens_dir", default="logit_lens_results_false_belief_100")
    parser.add_argument("--output", default="bash/logit_lens_belief_change_manifest.tsv")
    parser.add_argument("--min_episodes", type=int, default=600,
                        help="Only include pairs with at least this many episode dirs")
    args = parser.parse_args()

    existing_dirs = set(os.listdir(args.logit_lens_dir))

    complete_pairs = {}
    for dirname in sorted(existing_dirs):
        pair_path = os.path.join(args.logit_lens_dir, dirname)
        if not os.path.isdir(pair_path):
            continue
        n_eps = len(os.listdir(pair_path))
        if n_eps >= args.min_episodes:
            complete_pairs[dirname] = pair_path

    if not complete_pairs:
        print("ERROR: no pairs with enough episodes found", file=sys.stderr)
        sys.exit(1)

    sample_dir = next(iter(complete_pairs.values()))
    ep_map = build_scenario_episode_map(sample_dir)

    with open(args.behavioral_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    manifest = []
    for r in rows:
        cond = r["condition"]
        correct = int(r["correct"])
        if not ((cond == "false_help" and correct == 1) or
                (cond == "false_deceive" and correct == 0)):
            continue
        m1, m2 = parse_pair(r["pair"])
        dirname = pair_to_dirname(m1, m2)
        if dirname not in complete_pairs:
            continue
        ep_idx = ep_map.get((r["scenario"], cond))
        if ep_idx is None:
            print(f"WARNING: no episode mapping for {r['scenario']} {cond}", file=sys.stderr)
            continue
        manifest.append((m1, m2, ep_idx))

    with open(args.output, "w") as f:
        for m1, m2, ep in manifest:
            f.write(f"{m1}\t{m2}\t{ep}\n")

    n_pairs = len(set((m1, m2) for m1, m2, _ in manifest))
    n_fh = sum(1 for _, _, ep in manifest if ep % 6 == 4)
    n_fd = sum(1 for _, _, ep in manifest if ep % 6 == 5)
    print(f"Manifest: {len(manifest)} tasks across {n_pairs} pairs")
    print(f"  false_help (correct==1): {n_fh}")
    print(f"  false_deceive (correct==0): {n_fd}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
