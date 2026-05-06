#!/usr/bin/env python3
"""
Controlled experiment: fix Mia's opening to each of 13 unique patterns,
always stating the correct location, and measure Ava's 1-turn response.

13 patterns × 100 scenarios × 3 belief states = 3,900 Gemini API calls.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from litellm import completion
except ImportError:
    completion = None

REPO_ROOT = Path(__file__).resolve().parent.parent

_ACTOR_SYSTEM = """You are simulating one speaking turn in a structured multi-agent social scenario (Sotopia).

The user message contains:
- The agent's private context (background, relationship, scenario) with opponent secrets/goals masked.
- The dialogue so far; the last line ends with that agent having just been prompted to speak (e.g. "Alice said:" with nothing after the colon).

Rules:
- Output ONLY the words that agent speaks for this turn (no "Turn N:" prefix, no name prefix like "Alice said:").
- Stay in character; follow the private goal; do not reveal information the prompt marks as unknown to you.
- Be concise. If your goal is achieved and the scenario allows leaving, you may say LEAVE as instructed in the prompt.
- Reply on a single line when possible."""

A2_INSTRUCTION = (
    "Respond only with dialogue. Do not narrate actions or describe thoughts. "
    "Once you have decided which location to check, you should state your final "
    "choice and leave the conversation by saying LEAVE. Keep your responses concise.\n"
)

DEFAULT_MODEL = "vertex_ai/gemini-3.1-pro-preview"
DEFAULT_PROJECT = "hs-soil-gemini"
DEFAULT_LOCATION = "global"

PATTERN_TEMPLATES = {
    "The X is [prep] Y": "The {item} is {prep_loc}.",
    "The X you're looking for is Y": "The {item} you're looking for is {prep_loc}.",
    "If you're looking for X": "If you're looking for the {item}, it's {prep_loc}.",
    "Are you looking for X?": "Are you looking for the {item}? It's {prep_loc}.",
    "Looking for X?": "Looking for the {item}? It's {prep_loc}.",
    "I saw X on/in Y": "I saw the {item} {prep_loc}.",
    "Vocative (Ava/Hey Ava, ...)": "Ava, the {item} is {prep_loc}.",
    "Your X is on/in Y": "Your {item} is {prep_loc}.",
    "I think X is in Y": "I think the {item} is {prep_loc}.",
    "If you need X": "If you need the {item}, it's {prep_loc}.",
    "Bare anaphoric (It's in Y)": "It's {prep_loc}.",
    "Imperative (Check...)": "Check the {location}.",
    "You're looking for X, right?": "You're looking for the {item}, right? It's {prep_loc}.",
}

BELIEF_STATES = ["shared_truth", "ignorance", "false_belief"]
BELIEF_TO_CONDITION = {
    "shared_truth": "shared_help",
    "ignorance": "ignorance_help",
    "false_belief": "false_help",
}


def load_scenarios(csv_path: Path) -> list[dict]:
    df = __import__("pandas").read_csv(csv_path)
    scenarios = []
    for ep_id in sorted(df["episode"].unique()):
        ep_rows = df[df["episode"] == ep_id]
        row_sh = ep_rows[ep_rows["condition"] == "shared_help"]
        row_ih = ep_rows[ep_rows["condition"] == "ignorance_help"]
        row_fh = ep_rows[ep_rows["condition"] == "false_help"]

        if row_sh.empty or row_ih.empty or row_fh.empty:
            continue

        r = row_sh.iloc[0]
        scenario = {
            "episode": int(ep_id),
            "item": r["item"],
            "true_location": r["true_location"],
            "scenario_text": r["scenario"],
            "ava_goals": {
                "shared_truth": row_sh.iloc[0]["ava_goal"],
                "ignorance": row_ih.iloc[0]["ava_goal"],
                "false_belief": row_fh.iloc[0]["ava_goal"],
            },
        }

        prep_loc = extract_prep_location(r)
        scenario["prep_loc"] = prep_loc

        loc1, loc2 = extract_both_locations(r["scenario"])
        true_loc = r["true_location"]
        if true_loc in loc1 or loc1 in true_loc:
            scenario["other_location"] = loc2
        else:
            scenario["other_location"] = loc1

        scenarios.append(scenario)

    return scenarios


def extract_prep_location(row) -> str:
    dialogue = str(row["dialogue"])
    for line in dialogue.split("\n"):
        line = line.strip()
        if line.startswith("[Mia Sanders]:"):
            opening = line.replace("[Mia Sanders]:", "").strip()
            break
    else:
        return f"on the {row['true_location']}"

    patterns = [
        r"(?:is|are|it's|It's|They're|they're) ((?:on|in|at|by|over|right|hanging) .+?)\.?\s*$",
        r"saw .+? ((?:on|in|at|by|over|right) .+?)\.?\s*$",
        r"left .+? ((?:on|in|at|by|over|right) .+?)\.?\s*$",
        r"((?:on|in|at|by|over) the .+?)\.?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, opening)
        if m:
            return m.group(1).rstrip(".")
    return f"on the {row['true_location']}"


def extract_both_locations(scenario_text: str) -> tuple[str, str]:
    m = re.search(r"with (?:a |an |the )(.+?) and (?:a |an |the )(.+?)\.", scenario_text)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def build_ava_prompt(scenario_text: str, ava_goal: str, mia_opening: str) -> str:
    intro = (
        f"You are Ava Thompson. Respond only with short, direct dialogue.\n"
        f"Scenario: {scenario_text}\n"
        f"Participants: Mia Sanders and Ava Thompson\n"
        f"Ava Thompson's goal: {ava_goal}\n"
        f"Mia Sanders's goal: Unknown\n"
        f"{A2_INSTRUCTION}"
    )
    dialog = (
        f"Turn 0: Mia Sanders said:{mia_opening}\n"
        f"Turn 1: Ava Thompson said:"
    )
    return intro + dialog


def fill_template(pattern_name: str, item: str, prep_loc: str, location: str) -> str:
    template = PATTERN_TEMPLATES[pattern_name]
    return template.format(item=item, prep_loc=prep_loc, location=location)


def extract_choice(response: str, loc1: str, loc2: str) -> Optional[str]:
    text = response.lower()
    l1 = loc1.lower().replace("_", " ")
    l2 = loc2.lower().replace("_", " ")

    if "leave" in text:
        leave_idx = text.rfind("leave")
        best_loc = None
        best_dist = float("inf")
        for loc_str, loc_name in [(l1, loc1), (l2, loc2)]:
            idx = text.rfind(loc_str, 0, leave_idx)
            if idx >= 0:
                dist = leave_idx - idx
                if dist < best_dist:
                    best_dist = dist
                    best_loc = loc_name
        if best_loc is not None:
            return best_loc

    commit_patterns = [
        r"i'?ll check the",
        r"i'?ll go (?:to |check )",
        r"i'?m going to check",
        r"i will check the",
        r"let me check the",
        r"i'?ll trust you",
        r"okay,? i'?ll",
        r"alright,? i'?ll",
        r"thanks,? i'?ll",
    ]
    has_commit = any(re.search(p, text) for p in commit_patterns)
    if not has_commit:
        return None

    last_l1 = text.rfind(l1)
    last_l2 = text.rfind(l2)
    if last_l1 >= 0 and last_l2 >= 0:
        return loc1 if last_l1 > last_l2 else loc2
    if last_l1 >= 0:
        return loc1
    if last_l2 >= 0:
        return loc2
    return None


def generate_response(
    user_prompt: str,
    model: str,
    project: str,
    location: str,
    temperature: float = 0.7,
    max_tokens: int = 300,
    max_attempts: int = 3,
) -> str:
    if completion is None:
        raise RuntimeError("litellm not installed — cannot call API")
    for attempt in range(max_attempts):
        try:
            resp = completion(
                model=model,
                messages=[
                    {"role": "system", "content": _ACTOR_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                vertex_project=project,
                vertex_location=location,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
                num_retries=2,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^```\w*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return text
        except Exception as e:
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}/{max_attempts}): {e!r}; sleeping {wait}s")
            time.sleep(wait)
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled opening style experiment.")
    parser.add_argument("--scenarios_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--pattern", type=str, default=None,
                        help="Run only this pattern (for testing)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls")
    args = parser.parse_args()

    csv_path = Path(args.scenarios_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(csv_path)
    print(f"Loaded {len(scenarios)} scenarios")

    patterns = PATTERN_TEMPLATES
    if args.pattern:
        if args.pattern not in patterns:
            print(f"Unknown pattern: {args.pattern}")
            print(f"Available: {list(patterns.keys())}")
            sys.exit(1)
        patterns = {args.pattern: patterns[args.pattern]}

    out_csv = out_dir / "opening_style_experiment_results.csv"
    file_exists = out_csv.exists()
    existing_keys = set()
    if file_exists:
        import pandas as pd
        existing = pd.read_csv(out_csv)
        for _, r in existing.iterrows():
            existing_keys.add((r["pattern"], int(r["episode"]), r["belief"]))
        print(f"Resuming: {len(existing_keys)} existing results")

    total = len(patterns) * len(scenarios) * len(BELIEF_STATES)
    print(f"Total calls: {total} ({len(patterns)} patterns × "
          f"{len(scenarios)} scenarios × {len(BELIEF_STATES)} beliefs)")

    if args.dry_run:
        for pat_name in list(patterns.keys())[:2]:
            sc = scenarios[0]
            for belief in BELIEF_STATES[:1]:
                mia_opening = fill_template(
                    pat_name, sc["item"], sc["prep_loc"], sc["true_location"])
                prompt = build_ava_prompt(
                    sc["scenario_text"], sc["ava_goals"][belief], mia_opening)
                print(f"\n{'='*60}")
                print(f"Pattern: {pat_name} | Episode: {sc['episode']} | Belief: {belief}")
                print(f"Mia opening: {mia_opening}")
                print(f"{'='*60}")
                print(f"[SYSTEM]\n{_ACTOR_SYSTEM}\n")
                print(f"[USER]\n{prompt}")
        print(f"\n--- DRY RUN: would make {total} API calls ---")
        return

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "pattern", "episode", "item", "true_location", "other_location",
                "belief", "mia_opening", "ava_response", "final_choice", "correct",
            ])

        done = 0
        for pat_name in patterns:
            for sc in scenarios:
                for belief in BELIEF_STATES:
                    key = (pat_name, sc["episode"], belief)
                    if key in existing_keys:
                        done += 1
                        continue

                    mia_opening = fill_template(
                        pat_name, sc["item"], sc["prep_loc"],
                        sc["true_location"])
                    prompt = build_ava_prompt(
                        sc["scenario_text"], sc["ava_goals"][belief],
                        mia_opening)

                    response = generate_response(
                        prompt, args.model, args.project, args.location,
                        args.temperature)

                    choice = extract_choice(
                        response, sc["true_location"], sc["other_location"])
                    correct = int(choice == sc["true_location"]) if choice else ""

                    writer.writerow([
                        pat_name, sc["episode"], sc["item"],
                        sc["true_location"], sc["other_location"],
                        belief, mia_opening, response, choice or "", correct,
                    ])
                    f.flush()

                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{total} done")

                    if args.delay > 0:
                        time.sleep(args.delay)

    print(f"Done. Saved {out_csv}")


if __name__ == "__main__":
    main()
