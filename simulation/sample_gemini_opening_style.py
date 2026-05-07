#!/usr/bin/env python3
"""
Multi-turn opening style experiment: fix Mia's Turn 0 to each of 13 patterns,
then allow normal Gemini-generated turn-taking for subsequent turns.

13 patterns × 100 scenarios × 3 belief states = 3,900 multi-turn episodes.

Output format matches sotopia_results_*/dialogs/ structure:
  - Per-episode CSV: Intro, Dialog, Intended Dialog
  - prompt_records/ CSV: one row per turn with the full prompt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from analysis.run_opening_style_experiment import (
    PATTERN_TEMPLATES,
    fill_template,
    BELIEF_STATES,
    _ACTOR_SYSTEM,
)

DEFAULT_VERTEX_MODEL = "vertex_ai/gemini-3.1-pro-preview"
DEFAULT_VERTEX_PROJECT = "hs-soil-gemini"
DEFAULT_VERTEX_LOCATION = "global"

_vertex_generate = None
_postprocess_spoken_line = None


def _lazy_load_gemini():
    global _vertex_generate, _postprocess_spoken_line
    if _vertex_generate is not None:
        return
    from simulation.gemini_sot_env import (
        _vertex_generate as _vg,
        _postprocess_spoken_line as _pp,
    )
    _vertex_generate = _vg
    _postprocess_spoken_line = _pp

REPO_ROOT = Path(__file__).resolve().parent.parent

A1_INSTRUCTION = (
    "Respond only with dialogue. Do not narrate actions or describe thoughts. "
    "You should continue the conversation until the other person says LEAVE, "
    "do not say LEAVE first. Keep your responses concise.\n"
)

A2_INSTRUCTION = (
    "Respond only with dialogue. Do not narrate actions or describe thoughts. "
    "Once you have decided which location to check, you should state your final "
    "choice and leave the conversation by saying LEAVE. Keep your responses concise.\n"
)

AGENT1_NAME = "Mia Sanders"
AGENT2_NAME = "Ava Thompson"


def build_intro(agent_name, scenario_text, my_goal, other_name, instruction):
    return (
        f"You are {agent_name}. Respond only with short, direct dialogue.\n"
        f"Scenario: {scenario_text}\n"
        f"Participants: {AGENT1_NAME} and {AGENT2_NAME}\n"
        f"{agent_name}'s goal: {my_goal}\n"
        f"{other_name}'s goal: Unknown\n"
        f"{instruction}"
    )


def build_complete_intro(scenario_text, mia_goal, ava_goal):
    return (
        f"Here is the context of this interaction:\n"
        f"Scenario: {scenario_text}\n"
        f"Participants: {AGENT1_NAME} and {AGENT2_NAME}\n"
        f"{AGENT1_NAME}'s goal: {mia_goal}\n"
        f"{AGENT2_NAME}'s goal: {ava_goal}"
    )


def judge_terminate(dialog, cur_turn, max_turns):
    if cur_turn >= max_turns:
        return True
    last = dialog[-1].lower()
    head, tail = last[:10], last[-10:]
    if any(x in head or x in tail for x in ["left", "leave"]):
        return True
    return False


def slugify(pattern_name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", pattern_name).strip("_")


def load_scenarios_with_mia_goals(csv_path):
    pd = __import__("pandas")
    df = pd.read_csv(csv_path)
    scenarios = []
    for ep_id in sorted(df["episode"].unique()):
        ep_rows = df[df["episode"] == ep_id]
        row_sh = ep_rows[ep_rows["condition"] == "shared_help"]
        row_ih = ep_rows[ep_rows["condition"] == "ignorance_help"]
        row_fh = ep_rows[ep_rows["condition"] == "false_help"]

        if row_sh.empty or row_ih.empty or row_fh.empty:
            continue

        r = row_sh.iloc[0]

        from analysis.run_opening_style_experiment import (
            extract_prep_location,
            extract_both_locations,
        )

        prep_loc = extract_prep_location(r)
        loc1, loc2 = extract_both_locations(r["scenario"])
        true_loc = r["true_location"]
        other_loc = loc2 if (true_loc in loc1 or loc1 in true_loc) else loc1

        scenario = {
            "episode": int(ep_id),
            "item": r["item"],
            "true_location": true_loc,
            "other_location": other_loc,
            "scenario_text": r["scenario"],
            "prep_loc": prep_loc,
            "ava_goals": {
                "shared_truth": row_sh.iloc[0]["ava_goal"],
                "ignorance": row_ih.iloc[0]["ava_goal"],
                "false_belief": row_fh.iloc[0]["ava_goal"],
            },
            "mia_goals": {
                "shared_truth": row_sh.iloc[0]["mia_goal"],
                "ignorance": row_ih.iloc[0]["mia_goal"],
                "false_belief": row_fh.iloc[0]["mia_goal"],
            },
        }
        scenarios.append(scenario)
    return scenarios


def run_episode(
    scenario,
    belief,
    mia_opening,
    model,
    project,
    location,
    temperature,
    max_turns,
    delay,
):
    _lazy_load_gemini()
    mia_goal = scenario["mia_goals"][belief]
    ava_goal = scenario["ava_goals"][belief]
    scenario_text = scenario["scenario_text"]

    agent1_intro = build_intro(
        AGENT1_NAME, scenario_text, mia_goal, AGENT2_NAME, A1_INSTRUCTION
    )
    agent2_intro = build_intro(
        AGENT2_NAME, scenario_text, ava_goal, AGENT1_NAME, A2_INSTRUCTION
    )
    complete_intro = build_complete_intro(scenario_text, mia_goal, ava_goal)

    cur_dialog = [f"Turn 0: {AGENT1_NAME} said:{mia_opening}"]
    prompts = [agent1_intro + "\n".join(cur_dialog)]
    cur_turn = 1

    while cur_turn < max_turns:
        cur_dialog.append(f"Turn {cur_turn}: {AGENT2_NAME} said:")
        prompt = agent2_intro + "\n".join(cur_dialog)
        prompts.append(prompt)

        raw = _vertex_generate(
            model,
            prompt,
            temperature=temperature,
            max_tokens=None,
            vertex_project=project,
            vertex_location=location,
        )
        result = _postprocess_spoken_line(raw)
        cur_dialog[-1] += result
        cur_turn += 1

        if judge_terminate(cur_dialog, cur_turn, max_turns):
            break

        if delay > 0:
            time.sleep(delay)

        cur_dialog.append(f"Turn {cur_turn}: {AGENT1_NAME} said:")
        prompt = agent1_intro + "\n".join(cur_dialog)
        prompts.append(prompt)

        raw = _vertex_generate(
            model,
            prompt,
            temperature=temperature,
            max_tokens=None,
            vertex_project=project,
            vertex_location=location,
        )
        result = _postprocess_spoken_line(raw)
        cur_dialog[-1] += result
        cur_turn += 1

        if judge_terminate(cur_dialog, cur_turn, max_turns):
            break

        if delay > 0:
            time.sleep(delay)

    return complete_intro, cur_dialog, prompts, cur_turn


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn opening style experiment"
    )
    parser.add_argument("--scenarios_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="opening_style_multiturn_results")
    parser.add_argument("--model", type=str, default=DEFAULT_VERTEX_MODEL)
    parser.add_argument("--project", type=str, default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--location", type=str, default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--pattern", type=str, default=None,
                        help="Run only this pattern (for testing)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Cap scenarios (for smoke tests)")
    parser.add_argument("--belief", type=str, default=None,
                        help="Run only this belief state (e.g. false_belief)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    csv_path = Path(args.scenarios_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    scenarios = load_scenarios_with_mia_goals(csv_path)
    if args.max_episodes is not None:
        scenarios = scenarios[: args.max_episodes]
    print(f"Loaded {len(scenarios)} scenarios")

    patterns = PATTERN_TEMPLATES
    if args.pattern:
        if args.pattern not in patterns:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {list(patterns.keys())}")
            sys.exit(1)
        patterns = {args.pattern: patterns[args.pattern]}

    beliefs = BELIEF_STATES
    if args.belief:
        if args.belief not in BELIEF_STATES:
            print(f"Unknown belief: {args.belief}")
            print(f"Available: {BELIEF_STATES}")
            sys.exit(1)
        beliefs = [args.belief]

    model_slug = args.model.replace("vertex_ai/", "")
    dialog_base = out_dir / "dialogs" / f"{model_slug}_opening_style"

    total = len(patterns) * len(scenarios) * len(beliefs)
    print(f"Total episodes: {total} ({len(patterns)} patterns × "
          f"{len(scenarios)} scenarios × {len(beliefs)} beliefs)")

    if args.dry_run:
        sc = scenarios[0]
        for pat_name in list(patterns.keys())[:2]:
            for belief in beliefs[:1]:
                mia_opening = fill_template(
                    pat_name, sc["item"], sc["prep_loc"], sc["true_location"]
                )
                mia_goal = sc["mia_goals"][belief]
                ava_goal = sc["ava_goals"][belief]
                a1_intro = build_intro(
                    AGENT1_NAME, sc["scenario_text"], mia_goal,
                    AGENT2_NAME, A1_INSTRUCTION
                )
                a2_intro = build_intro(
                    AGENT2_NAME, sc["scenario_text"], ava_goal,
                    AGENT1_NAME, A2_INSTRUCTION
                )
                print(f"\n{'='*60}")
                print(f"Pattern: {pat_name} | Episode: {sc['episode']} | Belief: {belief}")
                print(f"Mia opening: {mia_opening}")
                print(f"{'='*60}")
                print(f"[SYSTEM]\n{_ACTOR_SYSTEM}\n")
                print(f"[MIA INTRO]\n{a1_intro}")
                print(f"[AVA INTRO + DIALOG]\n{a2_intro}"
                      f"Turn 0: {AGENT1_NAME} said:{mia_opening}\n"
                      f"Turn 1: {AGENT2_NAME} said:")
        print(f"\n--- DRY RUN: would run {total} episodes ---")
        return

    from tqdm import tqdm

    temp = args.temperature
    seed = 0

    tasks = []
    for pat_name in patterns:
        for sc in scenarios:
            for belief in beliefs:
                tasks.append((pat_name, sc, belief))

    skipped = 0
    pbar = tqdm(tasks, desc="Episodes", unit="ep")
    for pat_name, sc, belief in pbar:
        pat_slug = slugify(pat_name)
        pat_dir = dialog_base / pat_slug
        pr_dir = pat_dir / "prompt_records"
        pat_dir.mkdir(parents=True, exist_ok=True)
        pr_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{sc['episode']}_{belief}_temp{temp}_seed{seed}.csv"
        out_csv = pat_dir / fname
        pr_csv = pr_dir / fname

        if out_csv.exists() and out_csv.stat().st_size > 0:
            skipped += 1
            continue

        mia_opening = fill_template(
            pat_name, sc["item"], sc["prep_loc"], sc["true_location"]
        )

        pbar.set_postfix_str(
            f"{pat_slug[:20]} ep={sc['episode']} {belief[:5]}"
        )

        try:
            intro, dialog, prompts, n_turns = run_episode(
                sc, belief, mia_opening,
                args.model, args.project, args.location,
                args.temperature, args.max_turns, args.delay,
            )
        except Exception as e:
            print(f"  Error ep={sc['episode']} belief={belief} "
                  f"pattern={pat_name}: {e!r}")
            continue

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Intro", "Dialog", "Intended Dialog"])
            w.writerow([intro, dialog, dialog])

        with open(pr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["save_states"])
            for p in prompts:
                w.writerow([p])

    done = len(tasks) - skipped
    print(f"Done. {done} new + {skipped} existing = {len(tasks)} total "
          f"episodes in {dialog_base}")


if __name__ == "__main__":
    main()
