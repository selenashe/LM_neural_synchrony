"""
Build goal_alignment_labels_v2.json with two clean axes:

  1. task_structure  — deterministic, derived from Sotopia source/codename
     Categories:
       "distributive_bargaining"  — dividing a fixed pie (deal-or-no-deal, craigslist)
       "social_persuasion"        — changing behavior/beliefs (persuasion_for_good, charity, borrow, etc.)
       "relational_maintenance"   — managing ongoing relationships (social_chemistry, social_iqa relational)
       "information_exchange"     — discovering or sharing knowledge (mutual_friends, reveal_secret/answer)
       "norm_negotiation"         — navigating social norms / rules (normbank, talk_loudly, fire_trash, etc.)

  2. outcome_correspondence  — from existing alignment_score, discretised into thirds
       "corresponding"   (score > 0.3)
       "orthogonal"      (-0.3 <= score <= 0.3)
       "conflicting"     (score < -0.3)

Also preserves the raw alignment_score and all original v1 fields for reference.

Usage:
    python build_labels_v2.py [--v1 goal_alignment_labels.json] [--output goal_alignment_labels_v2.json]
"""

import os
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/envs.json")
COMBOS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/env_agent_combos.json")

CODENAME_TO_TASK = {
    # --- Distributive bargaining ---
    "divide_items":     "distributive_bargaining",
    "divide_things":    "distributive_bargaining",
    "divide_fruits":    "distributive_bargaining",
    "share_blanket":    "distributive_bargaining",
    "share_stuff":      "distributive_bargaining",
    "sleep_arrangement":"distributive_bargaining",
    "movie_to_watch":   "distributive_bargaining",
    "dinner_decision":  "distributive_bargaining",

    # --- Social persuasion ---
    "charity_donation":     "social_persuasion",
    "donate_charity":       "social_persuasion",
    "donate_funds":         "social_persuasion",
    "donate_money":         "social_persuasion",
    "donate_to_cause":      "social_persuasion",
    "small_donation":       "social_persuasion",
    "matching_donation":    "social_persuasion",
    "ask_for_donantion":    "social_persuasion",
    "financial_report":     "social_persuasion",
    "borrow_money":         "social_persuasion",
    "financial_support":    "social_persuasion",
    "join_trip":            "social_persuasion",
    "split_classes":        "social_persuasion",
    "drink_less":           "social_persuasion",
    "take_turns":           "social_persuasion",
    "sell_item":            "social_persuasion",
    "play_rights":          "social_persuasion",
    "revenge_plot":         "social_persuasion",
    "life_dilemma":         "social_persuasion",
    "prison_dilemma":       "social_persuasion",

    # --- Relational maintenance ---
    "apology_and_acceptance": "relational_maintenance",
    "rekindle_relationship":  "relational_maintenance",
    "emotional_support":      "relational_maintenance",
    "new_friends":            "relational_maintenance",
    "lying_affair":           "relational_maintenance",
    "political_views":        "relational_maintenance",
    "distance_friend":        "relational_maintenance",
    "go_on_a_date":           "relational_maintenance",
    "compliment_conflict":    "relational_maintenance",
    "interrupted_speech":     "relational_maintenance",
    "confess_mistake":        "relational_maintenance",
    "helping_hand":           "relational_maintenance",
    "break_bad_luck":         "relational_maintenance",
    "movie_scene":            "relational_maintenance",
    "drive_sportscar":        "relational_maintenance",
    "flirt_with_someone":     "relational_maintenance",
    "game_winning":           "relational_maintenance",
    "music_preference":       "relational_maintenance",
    "food_refusal":           "relational_maintenance",
    "free_stuff":             "relational_maintenance",
    "play_hooky":             "relational_maintenance",

    # --- Information exchange ---
    "reveal_secret":        "information_exchange",
    "reveal_answer":        "information_exchange",
    "ask_gift_preference":  "information_exchange",
    "secret_feeling":       "information_exchange",
    "correct_misinformation": "information_exchange",

    # --- Norm negotiation ---
    "talk_loudly":          "norm_negotiation",
    "fire_trash":           "norm_negotiation",
    "tree_trimming":        "norm_negotiation",
    "unwelcome_guest":      "norm_negotiation",
    "facetime_etiquettes":  "norm_negotiation",
    "yell":                 "norm_negotiation",
    "disagree_on_movie":    "norm_negotiation",
}

SOURCE_FALLBACK = {
    "craigslist_bargains": "distributive_bargaining",
    "deal-or-no-deal":     "distributive_bargaining",
    "mutual_friends":      "information_exchange",
    "persuation_for_good": "social_persuasion",
}


def classify_task_structure(codename, source):
    if codename in CODENAME_TO_TASK:
        return CODENAME_TO_TASK[codename]
    if source in SOURCE_FALLBACK:
        return SOURCE_FALLBACK[source]
    return "UNKNOWN"


def classify_outcome_correspondence(score):
    if score is None:
        return None
    if score > 0.3:
        return "corresponding"
    elif score < -0.3:
        return "conflicting"
    else:
        return "orthogonal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default=os.path.join(SCRIPT_DIR, "goal_alignment_labels.json"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "goal_alignment_labels_v2.json"))
    args = parser.parse_args()

    with open(ENVS_PATH) as f:
        envs = json.load(f)
    with open(COMBOS_PATH) as f:
        combos = json.load(f)
    with open(args.v1) as f:
        v1_labels = json.load(f)

    env_info = {}
    for ek, ev in envs.items():
        env_info[ek] = {
            "source": ev.get("source", "?"),
            "codename": ev.get("codename", "?"),
        }

    v2_labels = {}
    unknown_count = 0

    for idx in range(len(combos)):
        combo = combos[idx]
        env_id = combo["env_id"]
        ei = env_info.get(env_id, {"source": "?", "codename": "?"})
        source = ei["source"]
        codename = ei["codename"]

        task_structure = classify_task_structure(codename, source)
        if task_structure == "UNKNOWN":
            unknown_count += 1
            print(f"  WARNING: No mapping for codename={codename}, source={source} (combo {idx})")

        v1 = v1_labels.get(str(idx), {})
        alignment_score = v1.get("alignment_score")
        outcome_correspondence = classify_outcome_correspondence(alignment_score)

        v2_labels[str(idx)] = {
            "env_id": env_id,
            "source": source,
            "codename": codename,
            "task_structure": task_structure,
            "alignment_score": alignment_score,
            "outcome_correspondence": outcome_correspondence,
            # Preserve v1 fields for reference
            "v1_goal_structure": v1.get("goal_structure"),
            "v1_value_compatibility": v1.get("value_compatibility"),
            "v1_power_dynamic": v1.get("power_dynamic"),
            "v1_stakes": v1.get("stakes"),
            "v1_reasoning": v1.get("reasoning"),
        }

    with open(args.output, "w") as f:
        json.dump(v2_labels, f, indent=2)

    # --- Summary ---
    from collections import Counter
    ts_counts = Counter()
    oc_counts = Counter()
    ts_x_oc = {}

    for v in v2_labels.values():
        ts = v["task_structure"]
        oc = v["outcome_correspondence"]
        ts_counts[ts] += 1
        oc_counts[str(oc)] += 1
        key = (ts, str(oc))
        ts_x_oc[key] = ts_x_oc.get(key, 0) + 1

    print(f"\nTotal entries: {len(v2_labels)}")
    print(f"Unknown task_structure: {unknown_count}")

    print(f"\ntask_structure distribution:")
    for cat, n in ts_counts.most_common():
        print(f"  {cat:30s}: {n:4d} ({n/len(v2_labels):.1%})")

    print(f"\noutcome_correspondence distribution:")
    for cat, n in oc_counts.most_common():
        print(f"  {cat:30s}: {n:4d} ({n/len(v2_labels):.1%})")

    print(f"\ntask_structure x outcome_correspondence:")
    for ts in sorted(ts_counts.keys()):
        parts = []
        for oc in ["corresponding", "orthogonal", "conflicting", "None"]:
            n = ts_x_oc.get((ts, oc), 0)
            if n > 0:
                parts.append(f"{oc}={n}")
        print(f"  {ts:30s}: {', '.join(parts)}")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
