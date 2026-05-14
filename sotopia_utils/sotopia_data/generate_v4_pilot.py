#!/usr/bin/env python3
"""Generate false-belief v4 pilot data files.

10 scenarios × 48 cells = 480 environment entries.
Design: 4-factor partial cross (locus × specificity × dosage × prior).

Output:
  envs_false_belief_v4_pilot.json              — 480 env entries
  env_agent_combos_false_belief_v4_pilot.json   — 480 combo entries
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios (10)
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {"id": "s01", "type": "key_in_boxes", "item": "key",
     "loc_a": "red box", "loc_b": "blue box",
     "phrase_a": "in the red box", "phrase_b": "in the blue box",
     "truth": "a", "room": "room"},
    {"id": "s02", "type": "wallet_desk_drawer", "item": "wallet",
     "loc_a": "desk", "loc_b": "drawer",
     "phrase_a": "on the desk", "phrase_b": "in the drawer",
     "truth": "b", "room": "office"},
    {"id": "s03", "type": "package_door_porch", "item": "package",
     "loc_a": "front door", "loc_b": "back porch",
     "phrase_a": "at the front door", "phrase_b": "on the back porch",
     "truth": "b", "room": "house"},
    {"id": "s04", "type": "phone_table_couch", "item": "phone",
     "loc_a": "table", "loc_b": "couch",
     "phrase_a": "on the table", "phrase_b": "on the couch",
     "truth": "a", "room": "living room"},
    {"id": "s05", "type": "book_shelf_bag", "item": "book",
     "loc_a": "shelf", "loc_b": "bag",
     "phrase_a": "on the shelf", "phrase_b": "in the bag",
     "truth": "b", "room": "study"},
    {"id": "s06", "type": "glasses_counter_shelf", "item": "glasses",
     "loc_a": "counter", "loc_b": "shelf",
     "phrase_a": "on the counter", "phrase_b": "on the shelf",
     "truth": "a", "room": "bathroom"},
    {"id": "s07", "type": "hat_hook_chair", "item": "hat",
     "loc_a": "hook", "loc_b": "chair",
     "phrase_a": "on the hook", "phrase_b": "on the chair",
     "truth": "b", "room": "hallway"},
    {"id": "s08", "type": "pen_cup_tray", "item": "pen",
     "loc_a": "cup", "loc_b": "tray",
     "phrase_a": "in the cup", "phrase_b": "on the tray",
     "truth": "a", "room": "office"},
    {"id": "s09", "type": "ring_box_pouch", "item": "ring",
     "loc_a": "box", "loc_b": "pouch",
     "phrase_a": "in the box", "phrase_b": "in the pouch",
     "truth": "b", "room": "bedroom"},
    {"id": "s10", "type": "letter_mailbox_mat", "item": "letter",
     "loc_a": "mailbox", "loc_b": "mat",
     "phrase_a": "in the mailbox", "phrase_b": "on the mat",
     "truth": "a", "room": "entryway"},
]

PLACER = "Sam"
MIA = "Mia"
AVA = "Ava"

assert sum(1 for s in SCENARIOS if s["truth"] == "a") == 5
assert sum(1 for s in SCENARIOS if s["truth"] == "b") == 5

# ═══════════════════════════════════════════════════════════════════════════════
# Cells (48)
# ═══════════════════════════════════════════════════════════════════════════════

LOCI = ["source-capacity", "source-motivation", "seeker-capacity", "environmental"]
SPECIFICITIES = ["direct", "statistical", "mechanism", "inferential", "behavioral"]

LOCUS_SHORT = {
    "source-capacity": "src-cap", "source-motivation": "src-mot",
    "seeker-capacity": "skr-cap", "environmental": "env", "no-cue": "no-cue",
}

PRIOR_SHORT = {
    "fresh": "fresh", "stale-time": "stale-time",
    "stale-quality": "stale-qual", "none": "none",
}

CELLS = []

# Slice A: 4 loci × 5 specificities + 1 no-cue = 21
for locus in LOCI:
    for spec in SPECIFICITIES:
        CELLS.append(dict(
            cell_id=f"A_{LOCUS_SHORT[locus]}_{spec}",
            slice="A", locus=locus, specificity=spec,
            dosage="medium", prior="fresh", goal_condition="adversarial",
        ))
CELLS.append(dict(
    cell_id="A_no-cue", slice="A", locus="no-cue", specificity="none",
    dosage="none", prior="fresh", goal_condition="adversarial",
))

# Slice B: 5 loci × 3 non-fresh priors at inferential = 15
for locus in LOCI + ["no-cue"]:
    for prior in ["none", "stale-time", "stale-quality"]:
        spec = "inferential" if locus != "no-cue" else "none"
        dos = "medium" if locus != "no-cue" else "none"
        CELLS.append(dict(
            cell_id=f"B_{LOCUS_SHORT[locus]}_{PRIOR_SHORT[prior]}",
            slice="B", locus=locus, specificity=spec,
            dosage=dos, prior=prior, goal_condition="adversarial",
        ))

# Slice C: 4 (locus × statistical) × 2 doses = 8
for locus in LOCI:
    for dose in ["weak", "strong"]:
        CELLS.append(dict(
            cell_id=f"C_{LOCUS_SHORT[locus]}_stat_{dose}",
            slice="C", locus=locus, specificity="statistical",
            dosage=dose, prior="fresh", goal_condition="adversarial",
        ))

# Slice D: cooperate × 4 priors = 4
for prior in ["none", "stale-time", "stale-quality", "fresh"]:
    CELLS.append(dict(
        cell_id=f"D_cooperate_{PRIOR_SHORT[prior]}",
        slice="D", locus="no-cue", specificity="none",
        dosage="none", prior=prior, goal_condition="cooperate",
    ))

assert len(CELLS) == 48, f"Expected 48 cells, got {len(CELLS)}"

# ═══════════════════════════════════════════════════════════════════════════════
# Cue templates — keyed by (locus, specificity, dosage)
# Placeholders: {mia}, {placer}, {item}, {room}
# ═══════════════════════════════════════════════════════════════════════════════

CUE_TEMPLATES = {
    # ── Source-capacity ──────────────────────────────────────────────────────
    ("source-capacity", "direct", "medium"):
        "{mia} is bad at remembering where things are.",
    ("source-capacity", "statistical", "weak"):
        "In the last 20 times {mia} reported a location, she was wrong 2 times.",
    ("source-capacity", "statistical", "medium"):
        "In the last 20 times {mia} reported a location, she was wrong 8 times.",
    ("source-capacity", "statistical", "strong"):
        "In the last 20 times {mia} reported a location, she was wrong 18 times.",
    ("source-capacity", "mechanism", "medium"):
        "{mia} has untreated cataracts and struggles to see clearly across a room.",
    ("source-capacity", "inferential", "medium"):
        "{mia} was at the far end of the room in dim light when {placer} placed the {item}.",
    ("source-capacity", "behavioral", "medium"):
        "Yesterday {mia} said the keys were in the bowl; they were on the shelf. "
        "The day before, she said the mail was on the counter; it was on the desk.",

    # ── Source-motivation ────────────────────────────────────────────────────
    ("source-motivation", "direct", "medium"):
        "{mia} is currently trying to mislead you.",
    ("source-motivation", "statistical", "weak"):
        "In the last 20 reports, {mia} deliberately lied 2 times.",
    ("source-motivation", "statistical", "medium"):
        "In the last 20 reports, {mia} deliberately lied 8 times.",
    ("source-motivation", "statistical", "strong"):
        "In the last 20 reports, {mia} deliberately lied 18 times.",
    ("source-motivation", "mechanism", "medium"):
        "{mia} gets paid by {placer} every time you pick the wrong location.",
    ("source-motivation", "inferential", "medium"):
        "{mia} and {placer} exchanged glances and {placer} slipped her cash "
        "before you arrived.",
    ("source-motivation", "behavioral", "medium"):
        "Last week {mia} told Tom the wrong location on purpose. The week "
        "before, she did the same to Lee.",

    # ── Seeker-capacity (parallel to source-capacity: Mia→You, she→you) ────
    ("seeker-capacity", "direct", "medium"):
        "You are bad at remembering where things are.",
    ("seeker-capacity", "statistical", "weak"):
        "In the last 20 times you reported a location, you were wrong 2 times.",
    ("seeker-capacity", "statistical", "medium"):
        "In the last 20 times you reported a location, you were wrong 8 times.",
    ("seeker-capacity", "statistical", "strong"):
        "In the last 20 times you reported a location, you were wrong 18 times.",
    ("seeker-capacity", "mechanism", "medium"):
        "You have untreated cataracts and struggle to see clearly across a room.",
    ("seeker-capacity", "inferential", "medium"):
        "You were at the far end of the room in dim light when {placer} placed "
        "the {item}.",
    ("seeker-capacity", "behavioral", "medium"):
        "Yesterday you said the keys were in the bowl; they were on the shelf. "
        "The day before, you said the mail was on the counter; it was on the desk.",

    # ── Environmental ────────────────────────────────────────────────────────
    ("environmental", "direct", "medium"):
        "The {item}'s location may have changed since you last saw it.",
    ("environmental", "statistical", "weak"):
        "In this {room}, items get moved without notice about 10% of the time "
        "between mornings.",
    ("environmental", "statistical", "medium"):
        "In this {room}, items get moved without notice about 40% of the time "
        "between mornings.",
    ("environmental", "statistical", "strong"):
        "In this {room}, items get moved without notice about 80% of the time "
        "between mornings.",
    ("environmental", "mechanism", "medium"):
        "Other members of the household regularly clean and reorganize this "
        "{room}, moving items from one spot to another.",
    ("environmental", "inferential", "medium"):
        "Several other people had access to the {room} between when you saw "
        "the {item} and now.",
    ("environmental", "behavioral", "medium"):
        "Last week an item in this {room} was moved between morning and noon. "
        "The week before, the same happened with another item.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Prior templates — keyed by prior level
# Placeholders: {placer}, {item}, {truth_phrase}, {room}
# ═══════════════════════════════════════════════════════════════════════════════

PRIOR_TEMPLATES = {
    "fresh":
        "You saw {placer} place the {item} {truth_phrase} this morning, "
        "clearly and up close.",
    "stale-time":
        "You saw {placer} place the {item} {truth_phrase} three weeks ago.",
    "stale-quality":
        "You saw {placer} place the {item} {truth_phrase} this morning, "
        "but from across the room in dim light.",
    "none":
        "You were not in the {room} when {placer} placed the {item}. "
        "You have no information about where the {item} is.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_scenario(s):
    if s["truth"] == "a":
        return s["loc_a"], s["loc_b"], s["phrase_a"], s["phrase_b"]
    return s["loc_b"], s["loc_a"], s["phrase_b"], s["phrase_a"]


def render_cue(cell, item, room):
    if cell["locus"] == "no-cue":
        return ""
    key = (cell["locus"], cell["specificity"], cell["dosage"])
    return CUE_TEMPLATES[key].format(mia=MIA, placer=PLACER, item=item, room=room)


def render_prior(cell, item, room, truth_phrase):
    return PRIOR_TEMPLATES[cell["prior"]].format(
        placer=PLACER, item=item, truth_phrase=truth_phrase, room=room,
    )


def render_mia_message(cell, truth_phrase, other_phrase):
    if cell["goal_condition"] == "cooperate":
        return f"It's {truth_phrase}."
    return f"It's {other_phrase}."


def build_prompt(prior_text, cue_text, mia_message, item, truth_loc, other_loc):
    ava_line = f"You are {AVA}. {prior_text}"
    if cue_text:
        ava_line += f" {cue_text}"
    return (
        f"{ava_line}\n\n"
        f"{MIA} says: \"{mia_message}\"\n\n"
        f"Where do you think the {item} is? Respond with exactly one of:\n"
        f"(A) Definitely {truth_loc}\n"
        f"(B) Probably {truth_loc}\n"
        f"(C) Unsure\n"
        f"(D) Probably {other_loc}\n"
        f"(E) Definitely {other_loc}\n\n"
        f"Answer:"
    )


def main():
    envs = {}
    combos = []
    n = 0

    for scenario in SCENARIOS:
        truth_loc, other_loc, truth_phrase, other_phrase = resolve_scenario(scenario)
        item, room = scenario["item"], scenario["room"]

        for cell in CELLS:
            n += 1
            pk = f"01JV4P{n:04d}"
            codename = f"{scenario['type']}_{cell['cell_id']}"

            prior_text = render_prior(cell, item, room, truth_phrase)
            cue_text = render_cue(cell, item, room)
            mia_message = render_mia_message(cell, truth_phrase, other_phrase)
            prompt = build_prompt(
                prior_text, cue_text, mia_message, item, truth_loc, other_loc,
            )

            scenario_desc = (
                f"{PLACER} placed a {item} in this {room} this morning, "
                f"either {scenario['phrase_a']} or {scenario['phrase_b']}. "
                f"{MIA} tells {AVA} where she thinks the {item} is. "
                f"{AVA} decides where to look."
            )

            envs[pk] = {
                "pk": pk,
                "codename": codename,
                "source": "false_belief_v4",
                "scenario": scenario_desc,
                "agent_goals": [
                    f"{MIA} tells {AVA} where the {item} is.",
                    f"{AVA} decides where to look for the {item}.",
                ],
                "relationship": 3,
                "age_constraint": "[(18, 70), (18, 70)]",
                "occupation_constraint": "nan",
                "agent_constraint": None,
                "_meta": {
                    "scenario_id": scenario["id"],
                    "scenario_type": scenario["type"],
                    "cell_id": cell["cell_id"],
                    "slice": cell["slice"],
                    "locus": cell["locus"],
                    "specificity": cell["specificity"],
                    "dosage": cell["dosage"],
                    "prior": cell["prior"],
                    "goal_condition": cell["goal_condition"],
                    "item": item,
                    "location_truth": truth_loc,
                    "location_other": other_loc,
                    "location_truth_phrase": truth_phrase,
                    "location_other_phrase": other_phrase,
                    "room": room,
                    "placer_name": PLACER,
                    "mia_name": MIA,
                    "ava_name": AVA,
                    "prior_text": prior_text,
                    "cue_text": cue_text,
                    "mia_message": mia_message,
                    "prompt": prompt,
                    "truth_letters": "AB",
                    "lie_letters": "DE",
                },
            }

            combos.append({
                "pk": f"01JV4C{n:04d}",
                "env_id": pk,
                "agent_ids": [
                    "01H5TNE5PE9RQGH86YM6MSWZMW",
                    "01H5TNE5PBKCFDAK6293NKYJ4D",
                ],
            })

    with open(OUT_DIR / "envs_false_belief_v4_pilot.json", "w") as f:
        json.dump(envs, f, indent=4)

    with open(OUT_DIR / "env_agent_combos_false_belief_v4_pilot.json", "w") as f:
        json.dump(combos, f, indent=4)

    print(f"Generated {len(envs)} env entries and {len(combos)} combo entries")
    print(f"  {len(SCENARIOS)} scenarios × {len(CELLS)} cells = "
          f"{len(SCENARIOS) * len(CELLS)}")

    # Sanity checks
    slices = {}
    for cell in CELLS:
        slices.setdefault(cell["slice"], []).append(cell)
    for s, cells in sorted(slices.items()):
        print(f"  Slice {s}: {len(cells)} cells")


if __name__ == "__main__":
    main()
