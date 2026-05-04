#!/usr/bin/env python3
"""
Extract "surprising" behavioral outcomes for manual inspection.
V3 false-belief scenarios (24 conditions: 3 beliefs × 8 trust manipulations).

Target case types:
  1. mutual_coop_fail      — both know truth, guide cooperates, yet seeker picks wrong
  2. mutual_adv_fail       — both know truth, guide adversarial, seeker picks wrong
  3. conflict_coop_success — seeker has conflicting evidence, guide cooperates, seeker finds truth
  4. conflict_adv_success  — seeker has conflicting evidence AND adversarial guide, seeker finds truth
  5. conflict_coop_fail    — seeker has conflicting evidence, guide cooperates, seeker still wrong
  6. mutual_adv_success    — both know truth, guide adversarial, seeker picks correctly

Produces:
  - transcripts/{case_type}/{model1}_x_{model2}.txt  (with prompt context)
  - error_cases_summary.csv
  - error_cases_report.txt
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "logit_lens_results_false_belief_v3"
DIALOGS_ROOT = REPO_ROOT / "sotopia_results_false_belief_v3" / "dialogs"
OUT_DIR = REPO_ROOT / "summary_plots_false_belief_v3" / "error_cases"

BELIEF_SHORT = {
    "mutual_knowledge": "mutual",
    "asymmetric_knowledge": "asym",
    "conflicting_evidence": "conflict",
}

CASE_TYPES = {
    "mutual_coop_fail": {"belief": "mutual_knowledge", "goal": "cooperate", "correct": False},
    "mutual_adv_fail": {"belief": "mutual_knowledge", "goal": "adversarial", "correct": False},
    "conflict_coop_success": {"belief": "conflicting_evidence", "goal": "cooperate", "correct": True},
    "conflict_adv_success": {"belief": "conflicting_evidence", "goal": "adversarial", "correct": True},
    "conflict_coop_fail": {"belief": "conflicting_evidence", "goal": "cooperate", "correct": False},
    "mutual_adv_success": {"belief": "mutual_knowledge", "goal": "adversarial", "correct": True},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pair(pair_dir_name: str) -> Tuple[str, str]:
    parts = pair_dir_name.split("_None_0_")
    if len(parts) < 2:
        return pair_dir_name, ""
    m1 = parts[0]
    m2 = parts[1].split("_false_belief_v3")[0]
    return m1, m2


def pair_label(m1: str, m2: str) -> str:
    return f"{m1} x {m2}"


def parse_episode_dir(ep_name: str) -> Tuple[int, str]:
    parts = ep_name.split("_", 2)
    idx = int(parts[1])
    codename = parts[2] if len(parts) > 2 else ""
    return idx, codename


def match_case_type(meta: Dict, correct: bool) -> Optional[str]:
    for ct, spec in CASE_TYPES.items():
        if (meta["belief_condition"] == spec["belief"]
                and meta["goal_condition"] == spec["goal"]
                and correct == spec["correct"]):
            return ct
    return None


# ---------------------------------------------------------------------------
# Lightweight quality flags
# ---------------------------------------------------------------------------

def check_flags(d: Dict) -> Dict[str, bool]:
    flags = {}

    flags["no_leave"] = not any("leave" in t["text"].lower() for t in d["turns"])

    flags["prompt_leakage"] = any(
        "Mia Sanders said" in t["text"] or "Ava Thompson said" in t["text"]
        for t in d["turns"]
    )

    texts = Counter(t["text"].strip().lower() for t in d["turns"])
    flags["repetitive_loop"] = any(c >= 3 for c in texts.values())

    flags["max_turns_hit"] = d.get("n_turns", len(d["turns"])) >= 16

    flags["guide_says_leave"] = any(
        t["speaker"] == "Mia Sanders" and "LEAVE" in t["text"]
        for t in d["turns"]
    )

    meta = d["scenario_meta"]
    flags["guide_goal_failure"] = False
    neg_re = re.compile(r"(?:not|n['']t|no|isn['']t|don['']t|never)\s")
    if d["turns"] and d["turns"][0]["speaker"] == "Mia Sanders":
        text = d["turns"][0]["text"].lower()
        la, lb = meta["loc_a"].lower(), meta["loc_b"].lower()
        true_display = la if la.replace(" ", "_") == meta["true_location"] else lb
        false_display = lb if true_display == la else la
        has_true = true_display in text
        has_false = false_display in text

        def _is_negated(loc, txt):
            idx = txt.find(loc)
            if idx < 0:
                return False
            prefix = txt[max(0, idx - 30):idx]
            return bool(neg_re.search(prefix))

        if meta["goal_condition"] == "cooperate" and has_false and not has_true:
            if not _is_negated(false_display, text):
                flags["guide_goal_failure"] = True
        if meta["goal_condition"] == "adversarial" and has_true and not has_false:
            if not _is_negated(true_display, text):
                flags["guide_goal_failure"] = True

    return flags


# ---------------------------------------------------------------------------
# Load prompt context from CSV
# ---------------------------------------------------------------------------

def load_prompt_context(pair_dir_name: str, ep_idx: int,
                        dialogs_root: Path = None) -> Optional[str]:
    if dialogs_root is None:
        dialogs_root = DIALOGS_ROOT
    csv_path = dialogs_root / pair_dir_name / f"{ep_idx}_temp0.7_seed0.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get("Intro", "")
    except Exception:
        return None


def format_prompt_context(intro: str) -> str:
    if not intro:
        return "  (prompt context not available)"
    lines = []
    for part in intro.split("\n"):
        part = part.strip()
        if part:
            lines.append(f"  {part}")
    return "\n".join(lines) if lines else f"  {intro.strip()}"


def load_thinking_traces(pair_dir_name: str, ep_idx: int,
                         dialogs_root: Path = None) -> Optional[Dict[int, str]]:
    if dialogs_root is None:
        dialogs_root = DIALOGS_ROOT
    csv_path = dialogs_root / pair_dir_name / "thinking_traces" / f"{ep_idx}_temp0.7_seed0.csv"
    if not csv_path.exists():
        return None
    traces = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    traces[int(row[0])] = row[1].strip()
    except Exception:
        return None
    return traces if traces else None


# ---------------------------------------------------------------------------
# Format transcript
# ---------------------------------------------------------------------------

def format_case_transcript(d: Dict, ep_dir_name: str, ep_idx: int,
                           pair_dir_name: str, flags: Dict[str, bool],
                           dialogs_root: Path = None,
                           thinking_traces: Optional[Dict[int, str]] = None) -> str:
    meta = d["scenario_meta"]
    _, codename = parse_episode_dir(ep_dir_name)
    b_belief = meta.get("b_belief", meta.get("b_evidence", ""))

    lines = [f"--- Episode {ep_idx} ({codename}) ---"]
    lines.append(
        f"Item: {meta['item']} | True location: {meta['true_location'].replace('_', ' ')} | "
        f"Ava's belief: {str(b_belief).replace('_', ' ')} | "
        f"Final choice: {d.get('final_choice', 'None')} | "
        f"Correct: {d['correct']}"
    )
    lines.append(
        f"Condition: {meta.get('condition', '')} | Trust: {meta.get('trust_manipulation', '')}"
    )
    lines.append("")

    intro = load_prompt_context(pair_dir_name, ep_idx, dialogs_root)
    lines.append("PROMPT CONTEXT:")
    lines.append(format_prompt_context(intro))
    lines.append("")

    lines.append("CONVERSATION:")
    for t in d["turns"]:
        speaker = "Mia" if t["speaker"] == "Mia Sanders" else "Ava"
        if thinking_traces and t["turn"] in thinking_traces:
            trace = thinking_traces[t["turn"]]
            lines.append(f"  Turn {t['turn']} [{speaker} thinking]: {trace}")
        lines.append(f"  Turn {t['turn']} [{speaker}]: {t['text']}")

    active = [k for k, v in flags.items() if v]
    if active:
        lines.append(f"  ** FLAGS: {', '.join(active)} **")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RESULTS_ROOT, DIALOGS_ROOT, OUT_DIR

    parser = argparse.ArgumentParser(description="Extract surprising behavioral outcomes for inspection (v3).")
    parser.add_argument("--results_dir", type=str, default="logit_lens_results_false_belief_v3",
                        help="Results directory name (relative to REPO_ROOT).")
    parser.add_argument("--dialogs_dir", type=str, default=None,
                        help="Dialogs directory (relative to REPO_ROOT). "
                             "Default: derived from results_dir.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory name (relative to REPO_ROOT). "
                             "Default: summary_plots_{suffix}/error_cases")
    args = parser.parse_args()

    RESULTS_ROOT = REPO_ROOT / args.results_dir
    if args.dialogs_dir:
        DIALOGS_ROOT = REPO_ROOT / args.dialogs_dir
    else:
        sotopia_dir = args.results_dir.replace("logit_lens_results_", "sotopia_results_")
        DIALOGS_ROOT = REPO_ROOT / sotopia_dir / "dialogs"
    if args.output_dir:
        OUT_DIR = REPO_ROOT / args.output_dir
    else:
        suffix = args.results_dir.replace("logit_lens_results_", "summary_plots_")
        OUT_DIR = REPO_ROOT / suffix / "error_cases"

    if not RESULTS_ROOT.exists():
        print(f"Results directory not found: {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transcript_dir = OUT_DIR / "transcripts"
    for ct in CASE_TYPES:
        (transcript_dir / ct).mkdir(parents=True, exist_ok=True)

    cases: Dict[str, List[Dict]] = defaultdict(list)
    csv_rows: List[Dict] = []

    pair_dirs = sorted(RESULTS_ROOT.iterdir())
    for pair_dir in pair_dirs:
        if not pair_dir.is_dir():
            continue
        m1, m2 = parse_pair(pair_dir.name)

        for ep_dir in sorted(pair_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            beh_path = ep_dir / "behavioral_summary.json"
            if not beh_path.exists():
                continue
            with open(beh_path) as f:
                d = json.load(f)

            ep_idx, codename = parse_episode_dir(ep_dir.name)
            meta = d["scenario_meta"]
            ct = match_case_type(meta, d["correct"])
            if ct is None:
                continue

            flags = check_flags(d)
            traces = load_thinking_traces(pair_dir.name, ep_idx, DIALOGS_ROOT)
            transcript = format_case_transcript(
                d, ep_dir.name, ep_idx, pair_dir.name, flags, DIALOGS_ROOT,
                thinking_traces=traces
            )

            cases[ct].append({
                "pair": pair_label(m1, m2),
                "m1": m1, "m2": m2,
                "scenario": meta.get("scenario_type", codename),
                "codename": codename,
                "episode_idx": ep_idx,
                "n_turns": d.get("n_turns", len(d["turns"])),
                "final_choice": d.get("final_choice", ""),
                "transcript": transcript,
                "flags": flags,
            })

            csv_rows.append({
                "case_type": ct,
                "pair": pair_label(m1, m2),
                "model_1": m1,
                "model_2": m2,
                "scenario": meta.get("scenario_type", codename),
                "episode_idx": ep_idx,
                "n_turns": d.get("n_turns", len(d["turns"])),
                "final_choice": d.get("final_choice", ""),
                "has_leave": int(not flags["no_leave"]),
                "has_prompt_leak": int(flags["prompt_leakage"]),
                "has_repetition": int(flags["repetitive_loop"]),
                "has_max_turns": int(flags["max_turns_hit"]),
                "has_guide_leave": int(flags["guide_says_leave"]),
                "has_guide_goal_fail": int(flags["guide_goal_failure"]),
            })

    for ct, items in cases.items():
        by_pair: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            by_pair[item["pair"]].append(item)

        for pair, pair_items in sorted(by_pair.items()):
            safe_name = pair.replace(" ", "").replace("/", "_")
            out_path = transcript_dir / ct / f"{safe_name}.txt"
            with open(out_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"PAIR: {pair}\n")
                f.write(f"CASE TYPE: {ct}\n")
                f.write(f"COUNT: {len(pair_items)} episodes\n")
                f.write("=" * 80 + "\n\n")
                for item in sorted(pair_items, key=lambda x: x["episode_idx"]):
                    f.write(item["transcript"])
                    f.write("\n")

    print(f"Wrote transcripts to {transcript_dir}")

    csv_path = OUT_DIR / "error_cases_summary.csv"
    csv_cols = [
        "case_type", "pair", "model_1", "model_2", "scenario",
        "episode_idx", "n_turns", "final_choice",
        "has_leave", "has_prompt_leak", "has_repetition",
        "has_max_turns", "has_guide_leave", "has_guide_goal_fail",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path} ({len(csv_rows)} rows)")

    report_path = OUT_DIR / "error_cases_report.txt"
    with open(report_path, "w") as rpt:
        rpt.write("ERROR CASES REPORT (V3)\n")
        rpt.write("=" * 60 + "\n\n")

        rpt.write("CASE TYPE COUNTS\n")
        rpt.write("-" * 60 + "\n")
        for ct in CASE_TYPES:
            n = len(cases.get(ct, []))
            rpt.write(f"  {ct:<30} {n:>6}\n")
        rpt.write(f"  {'TOTAL':<30} {sum(len(v) for v in cases.values()):>6}\n\n")

        rpt.write("BY GUIDE MODEL (MODEL_1)\n")
        rpt.write("-" * 60 + "\n")
        for ct in CASE_TYPES:
            rpt.write(f"\n  {ct}:\n")
            m1_counts = Counter(item["m1"] for item in cases.get(ct, []))
            for model, count in m1_counts.most_common():
                rpt.write(f"    {model:<45} {count:>5}\n")

        rpt.write("\nBY SEEKER MODEL (MODEL_2)\n")
        rpt.write("-" * 60 + "\n")
        for ct in CASE_TYPES:
            rpt.write(f"\n  {ct}:\n")
            m2_counts = Counter(item["m2"] for item in cases.get(ct, []))
            for model, count in m2_counts.most_common():
                rpt.write(f"    {model:<45} {count:>5}\n")

        rpt.write("\nTOP PAIRS PER CASE TYPE\n")
        rpt.write("-" * 60 + "\n")
        for ct in CASE_TYPES:
            rpt.write(f"\n  {ct}:\n")
            pair_counts = Counter(item["pair"] for item in cases.get(ct, []))
            for pair, count in pair_counts.most_common(10):
                rpt.write(f"    {pair:<60} {count:>5}\n")

        rpt.write("\nQUALITY FLAG CROSS-TABULATION\n")
        rpt.write("-" * 60 + "\n")
        flag_names = ["no_leave", "prompt_leakage", "repetitive_loop",
                      "max_turns_hit", "guide_says_leave", "guide_goal_failure"]
        for ct in CASE_TYPES:
            items = cases.get(ct, [])
            n = len(items)
            if n == 0:
                continue
            rpt.write(f"\n  {ct} (n={n}):\n")
            for fn in flag_names:
                count = sum(1 for item in items if item["flags"].get(fn))
                pct = count / n * 100 if n else 0
                rpt.write(f"    {fn:<30} {count:>5} ({pct:>5.1f}%)\n")
            any_flag = sum(1 for item in items
                          if any(item["flags"].get(fn) for fn in flag_names))
            rpt.write(f"    {'any_flag':<30} {any_flag:>5} ({any_flag/n*100:>5.1f}%)\n")
            clean = n - any_flag
            rpt.write(f"    {'clean (no flags)':<30} {clean:>5} ({clean/n*100:>5.1f}%)\n")

    print(f"Wrote {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
