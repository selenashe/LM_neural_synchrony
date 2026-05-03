#!/usr/bin/env python3
"""Generate envs_false_belief_v3.json: 3 evidence structures × 8 conditions × 100 scenarios."""

import json
import re

with open("envs_false_belief_v2.json") as f:
    v2 = json.load(f)

# ---------------------------------------------------------------------------
# 1. Extract per-scenario-type config from v2
# ---------------------------------------------------------------------------
scenario_cfgs = {}

for k, v in v2.items():
    st = v["_meta"]["scenario_type"]
    cond = v["_meta"]["condition"]

    if st not in scenario_cfgs:
        scenario_cfgs[st] = {
            "scenario_text": v["scenario"],
            "true_location": v["_meta"]["true_location"],
        }

    cfg = scenario_cfgs[st]

    if cond == "C1":
        # Extract "place_true" phrase from Mia's mutual-cooperate goal
        m = re.search(r"Sam place (.+?)\. Ava also", v["agent_goals"][0])
        cfg["place_true"] = m.group(1)
        # Decision phrase from Ava
        m2 = re.search(r"Talk to Mia and decide (.+?)\. Once", v["agent_goals"][1])
        cfg["decision_phrase"] = m2.group(1) if m2 else "where to look"

    if cond == "C2":
        # Item name
        m = re.search(r"no information about where the (.+?) is", v["agent_goals"][1])
        cfg["item_name"] = m.group(1) if m else "item"

    if cond == "C4":
        # False location phrase from Ava's conflict goal
        m = re.search(r"Sam place (.+?) yesterday", v["agent_goals"][1])
        cfg["place_false"] = m.group(1) if m else None
        # "moved it to X" from Mia
        m2 = re.search(r"moved it to (.+?) overnight", v["agent_goals"][0])
        cfg["moved_to_true"] = m2.group(1) if m2 else None
        cfg["false_location"] = v["_meta"]["b_evidence"]

# ---------------------------------------------------------------------------
# 2. Condition definitions (3 evidence × 8 manipulations = 24)
# ---------------------------------------------------------------------------
# Evidence structures: mutual, asymmetric, conflict
# Manipulations:
#   A1=expert, A2=unreliable, A3=incentivized, A4=malicious,
#   B1=forgetting, B2=intervening, B3=poor_visibility,
#   C=cooperate

CONDITIONS = [
    # (condition_code, evidence, manipulation, goal_condition)
    # --- Mutual ---
    ("mutual_A1_expert",       "mutual",    "expert",          "adversarial"),
    ("mutual_A2_unreliable",   "mutual",    "unreliable",      "adversarial"),
    ("mutual_A3_incentivized", "mutual",    "incentivized",    "adversarial"),
    ("mutual_A4_malicious",    "mutual",    "malicious",       "adversarial"),
    ("mutual_B1_forgetting",   "mutual",    "forgetting",      "adversarial"),
    ("mutual_B2_intervening",  "mutual",    "intervening",     "adversarial"),
    ("mutual_B3_poor_vis",     "mutual",    "poor_visibility", "adversarial"),
    ("mutual_C_cooperate",     "mutual",    "none",            "cooperate"),
    # --- Asymmetric ---
    ("asym_A1_expert",         "asymmetric", "expert",          "adversarial"),
    ("asym_A2_unreliable",     "asymmetric", "unreliable",      "adversarial"),
    ("asym_A3_incentivized",   "asymmetric", "incentivized",    "adversarial"),
    ("asym_A4_malicious",      "asymmetric", "malicious",       "adversarial"),
    ("asym_B1_forgetting",     "asymmetric", "forgetting",      "adversarial"),
    ("asym_B2_intervening",    "asymmetric", "intervening",     "adversarial"),
    ("asym_B3_poor_vis",       "asymmetric", "poor_visibility", "adversarial"),
    ("asym_C_cooperate",       "asymmetric", "none",            "cooperate"),
    # --- Conflict ---
    ("conflict_A1_expert",       "conflict", "expert",          "adversarial"),
    ("conflict_A2_unreliable",   "conflict", "unreliable",      "adversarial"),
    ("conflict_A3_incentivized", "conflict", "incentivized",    "adversarial"),
    ("conflict_A4_malicious",    "conflict", "malicious",       "adversarial"),
    ("conflict_B1_forgetting",   "conflict", "forgetting",      "adversarial"),
    ("conflict_B2_intervening",  "conflict", "intervening",     "adversarial"),
    ("conflict_B3_poor_vis",     "conflict", "poor_visibility", "adversarial"),
    ("conflict_C_cooperate",     "conflict", "none",            "cooperate"),
]

# ---------------------------------------------------------------------------
# 3. Text generation helpers
# ---------------------------------------------------------------------------

def mia_observation_mutual(cfg, manipulation):
    """Mia's observation sentence for mutual evidence."""
    place = cfg["place_true"]
    if manipulation == "forgetting":
        return f"You watched Sam place {place} this morning. Ava watched Sam place {place} three weeks ago."
    else:
        return f"You watched Sam place {place}. Ava also watched Sam place {place}."


def ava_observation_mutual(cfg, manipulation):
    """Ava's observation sentence for mutual evidence."""
    place = cfg["place_true"]
    if manipulation == "forgetting":
        return f"You watched Sam place {place} three weeks ago. Mia also watched Sam place {place}."
    elif manipulation == "poor_visibility":
        return (
            f"You watched Sam place {place}, though the lighting in the room was poor "
            f"and you were at the far end of the room when it happened. "
            f"Mia also watched Sam place {place}."
        )
    elif manipulation == "intervening":
        return (
            f"You watched Sam place {place}. Mia also watched Sam place {place}. "
            f"Since then, several other people have had access to the room."
        )
    else:
        return f"You watched Sam place {place}. Mia also watched Sam place {place}."


def mia_observation_asym(cfg, manipulation):
    """Mia's observation sentence for asymmetric evidence."""
    place = cfg["place_true"]
    item = cfg["item_name"]
    if manipulation == "forgetting":
        return f"You watched Sam place {place} three weeks ago. Ava was not in the room and has no information about the {item}'s location."
    else:
        return f"You watched Sam place {place}. Ava was not in the room and has no information about the {item}'s location."


def ava_observation_asym(cfg, manipulation):
    """Ava's observation sentence for asymmetric evidence."""
    item = cfg["item_name"]
    base = f"You were not in the room when Sam placed the {item}. You have no information about where the {item} is."
    if manipulation == "forgetting":
        return base + f" Mia was in the room when Sam placed the {item}, though that was three weeks ago."
    elif manipulation == "poor_visibility":
        return (
            base
            + f" Mia was in the room when Sam placed the {item}, though the lighting in the room was poor "
            f"and Mia was at the far end of the room when it happened."
        )
    elif manipulation == "intervening":
        return base + f" Since Mia observed Sam place the {item}, several other people have had access to the room."
    else:
        return base


def mia_observation_conflict(cfg, manipulation):
    """Mia's observation sentence for conflicting evidence."""
    place_true = cfg["place_true"]
    place_false = cfg["place_false"]
    moved_to = cfg["moved_to_true"]
    if manipulation == "forgetting":
        return (
            f"You watched Sam place {place_true} this morning. "
            f"Ava watched Sam place {place_false} three weeks ago, "
            f"but Sam moved it to {moved_to} since then, which Ava does not know."
        )
    else:
        return (
            f"You watched Sam place {place_true} this morning. "
            f"Ava watched Sam place {place_false} yesterday, "
            f"but Sam moved it to {moved_to} overnight, which Ava does not know."
        )


def ava_observation_conflict(cfg, manipulation):
    """Ava's observation sentence for conflicting evidence."""
    place_false = cfg["place_false"]
    if manipulation == "forgetting":
        return f"You watched Sam place {place_false} three weeks ago. Mia was in the room more recently than you were."
    elif manipulation == "poor_visibility":
        return (
            f"You watched Sam place {place_false} yesterday, "
            f"though the lighting in the room was poor and you were at the far end of the room when it happened. "
            f"Mia was in the room more recently than you were."
        )
    elif manipulation == "intervening":
        return (
            f"You watched Sam place {place_false} yesterday. "
            f"Mia was in the room more recently than you were. "
            f"Since you last observed the room, several other people have had access to it."
        )
    else:
        return f"You watched Sam place {place_false} yesterday. Mia was in the room more recently than you were."


def ava_trust_clause(manipulation):
    """Ava's trust/reliability clause (the manipulation)."""
    if manipulation == "expert":
        return " You have worked with Mia for years and she has never once misremembered where something was placed."
    elif manipulation == "unreliable":
        return " You have worked with Mia for years and she has mixed up where things were placed on several occasions."
    elif manipulation == "incentivized":
        return " You know that Mia has a side bet with Sam — she wins money if you choose the wrong location."
    elif manipulation == "malicious":
        return " You have seen Mia deliberately tell people the wrong location of things in the past."
    else:
        return ""


def build_goals(cfg, evidence, manipulation, goal_condition):
    """Build (mia_goal, ava_goal) strings."""
    dec = cfg["decision_phrase"]

    # --- Mia's goal ---
    if evidence == "mutual":
        mia_obs = mia_observation_mutual(cfg, manipulation)
    elif evidence == "asymmetric":
        mia_obs = mia_observation_asym(cfg, manipulation)
    else:
        mia_obs = mia_observation_conflict(cfg, manipulation)

    if goal_condition == "cooperate":
        if evidence == "mutual":
            mia_goal_line = "Your goal: confirm to Ava where the {item} is. Be direct and concise."
        elif evidence == "asymmetric":
            mia_goal_line = "Your goal: help Ava find the {item}. Be direct and concise."
        else:
            mia_goal_line = "Your goal: help Ava find the {item}. Be direct and concise."
        mia_goal_line = mia_goal_line.format(item=cfg["item_name"])
        # For conflict cooperate, Mia also knows Ava knows recency
        if evidence == "conflict":
            mia_obs += " Ava knows you were in the room more recently than she was."
    else:
        mia_goal_line = "Your goal: mislead Ava into choosing the wrong location. Be direct and concise."

    mia_full = f"You are Mia. {mia_obs} {mia_goal_line}"

    # --- Ava's goal ---
    if evidence == "mutual":
        ava_obs = ava_observation_mutual(cfg, manipulation)
    elif evidence == "asymmetric":
        ava_obs = ava_observation_asym(cfg, manipulation)
    else:
        ava_obs = ava_observation_conflict(cfg, manipulation)

    # Trust clause (only for A-type manipulations, not B-type or cooperate)
    if manipulation in ("expert", "unreliable", "incentivized", "malicious"):
        trust = ava_trust_clause(manipulation)
    else:
        trust = ""

    ava_full = (
        f"You are Ava. {ava_obs}{trust} "
        f"Talk to Mia and decide {dec}. "
        f"Once you have decided, state your final choice and say LEAVE."
    )

    return mia_full, ava_full


# ---------------------------------------------------------------------------
# 4. Generate all entries
# ---------------------------------------------------------------------------
output = {}
entry_num = 0

for st in sorted(scenario_cfgs.keys()):
    cfg = scenario_cfgs[st]
    for cond_code, evidence, manipulation, goal_cond in CONDITIONS:
        entry_num += 1
        pk = f"01JGBENV{entry_num:04d}"
        codename = f"{st.split('_')[0]}_{cond_code}"  # e.g. key_mutual_A1_expert

        # Build codename from scenario type prefix + condition
        # Use full scenario type name for clarity
        codename = f"{st}_{cond_code}"

        mia_goal, ava_goal = build_goals(cfg, evidence, manipulation, goal_cond)

        # Determine b_evidence (what Ava saw)
        if evidence == "mutual":
            b_evidence = cfg["true_location"]
        elif evidence == "asymmetric":
            b_evidence = "none"
        else:
            b_evidence = cfg["false_location"]

        # Determine a_gold
        a_gold = "say_true" if goal_cond == "cooperate" else "say_false"

        # Determine b_gold (correct answer is always true_location)
        b_gold = cfg["true_location"]

        entry = {
            "pk": pk,
            "codename": codename,
            "source": "false_belief",
            "scenario": cfg["scenario_text"],
            "agent_goals": [mia_goal, ava_goal],
            "relationship": 3,
            "age_constraint": "[(18, 70), (18, 70)]",
            "occupation_constraint": "nan",
            "agent_constraint": None,
            "_meta": {
                "scenario_id": codename,
                "scenario_type": st,
                "condition": cond_code,
                "true_location": cfg["true_location"],
                "belief_condition": {
                    "mutual": "mutual_knowledge",
                    "asymmetric": "asymmetric_knowledge",
                    "conflict": "conflicting_evidence",
                }[evidence],
                "goal_condition": goal_cond,
                "trust_manipulation": manipulation,
                "b_evidence": b_evidence,
                "mismatch": evidence == "conflict",
                "a_gold": a_gold,
                "b_gold": b_gold,
            },
        }

        output[pk] = entry

with open("envs_false_belief_v3.json", "w") as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(output)} entries ({len(scenario_cfgs)} scenarios × {len(CONDITIONS)} conditions)")
