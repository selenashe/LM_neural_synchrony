#!/usr/bin/env python3
"""Aggregate and analyze logit lens results across all model-pair x episode combinations."""

import json
import os
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "logit_lens_results_false_belief_100"
FINAL_LAYER = 32  # index into the 33-element list (0..32)

# ──────────────────────────────────────────────────────────────────────
# PART 1 — Behavioral summary
# ──────────────────────────────────────────────────────────────────────

def load_all_behavioral():
    rows = []
    for pair_dir in sorted(BASE.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair_name = pair_dir.name
        parts = pair_name.split("_None_0_")
        model_a = parts[0]
        model_b = parts[1].split("_false_belief")[0]
        short_pair = f"{model_a.split('-')[-1]} × {model_b.split('-')[-1]}"

        for ep_dir in sorted(pair_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            bfile = ep_dir / "behavioral_summary.json"
            if not bfile.exists():
                continue
            with open(bfile) as f:
                d = json.load(f)
            meta = d["scenario_meta"]
            rows.append({
                "pair": pair_name,
                "short_pair": short_pair,
                "episode": ep_dir.name,
                "goal": meta["goal_condition"],
                "belief": meta["belief_condition"],
                "item": meta["item"],
                "final_choice": d["final_choice"],
                "correct": d["correct"],
                "n_turns": d["n_turns"],
            })
    return rows


def print_behavioral_table(rows):
    print("=" * 90)
    print("PART 1  —  BEHAVIORAL SUMMARY")
    print("=" * 90)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["short_pair"], r["goal"], r["belief"])].append(r)

    header = f"{'Model pair':<22} {'Goal':<10} {'Belief':<16} {'N':>3} {'%Correct':>9} {'AvgTurns':>9}  Final-choice breakdown"
    print(header)
    print("-" * len(header) + "-" * 30)

    for (pair, goal, belief) in sorted(groups):
        grp = groups[(pair, goal, belief)]
        n = len(grp)
        frac_correct = sum(r["correct"] for r in grp) / n
        avg_turns = sum(r["n_turns"] for r in grp) / n
        choice_counts = defaultdict(int)
        for r in grp:
            fc = r["final_choice"] if r["final_choice"] is not None else "NONE"
            choice_counts[fc] += 1
        choices_str = ", ".join(f"{k}: {v}" for k, v in sorted(choice_counts.items(), key=lambda x: str(x[0])))
        print(f"{pair:<22} {goal:<10} {belief:<16} {n:>3} {frac_correct:>8.1%} {avg_turns:>9.1f}  {choices_str}")

    # Also print overall stats per model pair
    print()
    print("Overall accuracy by model pair:")
    pair_groups = defaultdict(list)
    for r in rows:
        pair_groups[r["short_pair"]].append(r)
    for pair in sorted(pair_groups):
        grp = pair_groups[pair]
        n = len(grp)
        frac = sum(r["correct"] for r in grp) / n
        avg_t = sum(r["n_turns"] for r in grp) / n
        print(f"  {pair:<22}  N={n:>3}  accuracy={frac:.1%}  avg_turns={avg_t:.1f}")

    # Accuracy by goal condition
    print()
    print("Overall accuracy by goal condition:")
    goal_groups = defaultdict(list)
    for r in rows:
        goal_groups[r["goal"]].append(r)
    for goal in sorted(goal_groups):
        grp = goal_groups[goal]
        n = len(grp)
        frac = sum(r["correct"] for r in grp) / n
        print(f"  {goal:<10}  N={n:>3}  accuracy={frac:.1%}")


# ──────────────────────────────────────────────────────────────────────
# PART 2 — Contrastive probes
# ──────────────────────────────────────────────────────────────────────

PROBES = ["self_belief_location", "self_honesty", "other_strategy"]

def load_all_contrastive():
    """Return list of dicts with contrastive scores at the final layer."""
    records = []
    for pair_dir in sorted(BASE.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair_name = pair_dir.name
        parts = pair_name.split("_None_0_")
        model_a = parts[0]
        model_b = parts[1].split("_false_belief")[0]
        short_pair = f"{model_a.split('-')[-1]} × {model_b.split('-')[-1]}"

        for ep_dir in sorted(pair_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            cfile = ep_dir / "contrastive_summary.json"
            if not cfile.exists():
                continue
            with open(cfile) as f:
                d = json.load(f)
            meta = d["scenario_meta"]
            goal = meta["goal_condition"]
            belief = meta["belief_condition"]

            for agent_key in ["agent_1", "agent_2"]:
                agent_data = d[agent_key]
                agent_idx = 1 if agent_key == "agent_1" else 2
                for probe in PROBES:
                    if probe not in agent_data:
                        continue
                    for turn_str, layer_scores in agent_data[probe].items():
                        # layer_scores is list of 33 elements, each [score_comp0, score_comp1]
                        scores = layer_scores[FINAL_LAYER]
                        diff = scores[0] - scores[1]
                        records.append({
                            "short_pair": short_pair,
                            "goal": goal,
                            "belief": belief,
                            "agent_idx": agent_idx,
                            "probe": probe,
                            "turn": int(turn_str),
                            "comp0": scores[0],
                            "comp1": scores[1],
                            "diff": diff,
                        })
    return records


def print_contrastive_summary(records):
    print()
    print("=" * 90)
    print("PART 2  —  CONTRASTIVE PROBES AT FINAL LAYER (layer 32)")
    print("=" * 90)
    print()
    print("Completion indices: for self_belief_location, comp0 = loc_a (true loc), comp1 = loc_b")
    print("  for self_honesty, comp0 = honest, comp1 = deceptive")
    print("  for other_strategy, comp0 = help, comp1 = mislead")
    print("  diff = comp0 - comp1  (positive = comp0 is more likely)")
    print()

    for probe in PROBES:
        print(f"─── {probe} ───")
        probe_recs = [r for r in records if r["probe"] == probe]

        # Group by (pair, goal, agent_idx) and average diff
        groups = defaultdict(list)
        for r in probe_recs:
            groups[(r["short_pair"], r["goal"], r["agent_idx"])].append(r)

        header = f"  {'Model pair':<22} {'Goal':<10} {'Agent':>5} {'N':>4} {'AvgDiff':>9} {'Avg(c0)':>9} {'Avg(c1)':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in sorted(groups):
            grp = groups[key]
            n = len(grp)
            avg_diff = sum(r["diff"] for r in grp) / n
            avg_c0 = sum(r["comp0"] for r in grp) / n
            avg_c1 = sum(r["comp1"] for r in grp) / n
            pair, goal, aidx = key
            print(f"  {pair:<22} {goal:<10} {aidx:>5} {n:>4} {avg_diff:>+9.3f} {avg_c0:>9.3f} {avg_c1:>9.3f}")
        print()

        # Also break down by belief condition
        print(f"  Breakdown by belief condition:")
        groups2 = defaultdict(list)
        for r in probe_recs:
            groups2[(r["goal"], r["belief"], r["agent_idx"])].append(r)

        header2 = f"    {'Goal':<10} {'Belief':<16} {'Agent':>5} {'N':>4} {'AvgDiff':>9}"
        print(header2)
        print("    " + "-" * (len(header2) - 4))
        for key in sorted(groups2):
            grp = groups2[key]
            n = len(grp)
            avg_diff = sum(r["diff"] for r in grp) / n
            goal, belief, aidx = key
            print(f"    {goal:<10} {belief:<16} {aidx:>5} {n:>4} {avg_diff:>+9.3f}")
        print()


# ──────────────────────────────────────────────────────────────────────
# PART 3 — Stem predictions for select episodes
# ──────────────────────────────────────────────────────────────────────

STEMS_OF_INTEREST = [
    "The key is in the",
    "I am being",
    "I should tell them the key is in the",
]
LAYERS_OF_INTEREST = [0, 8, 16, 24, 31]
V02_PAIR = "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0_false_belief_fixed_two_agents"
EPISODES_OF_INTEREST = [0, 1, 4, 5]


def print_stem_predictions():
    print()
    print("=" * 100)
    print("PART 3  —  STEM PREDICTIONS (v0.2 × v0.2, episodes 0/1/4/5)")
    print("=" * 100)

    pair_dir = BASE / V02_PAIR
    ep_dirs = sorted(pair_dir.iterdir())
    ep_name_map = {}
    for d in ep_dirs:
        parts = d.name.split("_")
        ep_num = int(parts[1])
        ep_name_map[ep_num] = d

    for ep_num in EPISODES_OF_INTEREST:
        if ep_num not in ep_name_map:
            print(f"\n  Episode {ep_num}: NOT FOUND")
            continue
        ep_dir = ep_name_map[ep_num]
        sfile = ep_dir / "stem_summary.json"
        if not sfile.exists():
            print(f"\n  Episode {ep_num}: stem_summary.json missing")
            continue
        with open(sfile) as f:
            sd = json.load(f)

        scenario = sd["scenario"]
        meta = sd["scenario_meta"]
        print(f"\n{'─' * 100}")
        print(f"Episode {ep_num}: {scenario}")
        print(f"  true_location={meta['true_location']}  belief={meta['belief_condition']}  goal={meta['goal_condition']}")
        print(f"  loc_a={meta['loc_a']}  loc_b={meta['loc_b']}")

        for stem in STEMS_OF_INTEREST:
            print(f"\n  Stem: \"{stem}\"")
            layer_header = "".join(f"{'L' + str(l):>12}" for l in LAYERS_OF_INTEREST)
            print(f"    {'Turn':>4} {'Agent':>7} {layer_header}")
            print(f"    " + "-" * (4 + 7 + 12 * len(LAYERS_OF_INTEREST) + 2))

            for turn_data in sd["turns"]:
                turn_idx = turn_data["turn"]
                agent_idx = turn_data["agent_index"]
                if stem not in turn_data["stems"]:
                    continue
                preds = turn_data["stems"][stem]["continuation_prediction_after_full_stem"]
                tokens = []
                for li in LAYERS_OF_INTEREST:
                    if li < len(preds):
                        tok = preds[li]["predicted_token_clean"]
                        tokens.append(tok[:10])
                    else:
                        tokens.append("?")
                row = "".join(f"{t:>12}" for t in tokens)
                print(f"    {turn_idx:>4} {agent_idx:>7} {row}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading behavioral data...")
    behavioral_rows = load_all_behavioral()
    print(f"  Loaded {len(behavioral_rows)} episodes.")
    print_behavioral_table(behavioral_rows)

    print("\nLoading contrastive data...")
    contrastive_records = load_all_contrastive()
    print(f"  Loaded {len(contrastive_records)} contrastive records.")
    print_contrastive_summary(contrastive_records)

    print_stem_predictions()
