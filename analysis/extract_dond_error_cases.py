#!/usr/bin/env python3
"""
Extract "surprising" negotiation outcomes from Deal-or-No-Deal episodes
for manual inspection.

Case types:
  1. pareto_dominated   — Agreed deal is Pareto-dominated by another feasible allocation
  2. one_sided_deal     — One agent scores >= 8, other <= 2
  3. irrational_acceptance — An agent accepts 0 points
  4. value_destroying   — Agreed joint score < 10

Produces:
  - transcripts/{case_type}/{model}.txt
  - error_cases_summary.csv
  - error_cases_report.txt
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fao_analysis import (
    load_scenarios, RESULT_DIRS, _load_csv_dialog, _get_speaker, _get_utterance,
    parse_episode, score_allocation, compute_scenario_metadata, extract_model_name,
    PROJECT_ROOT, Scenario, _pluralize,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'summary_plots_deal_or_no_deal' / 'error_cases'

CASE_TYPES = {
    "pareto_dominated": "Agreed deal is Pareto-dominated",
    "one_sided_deal": "One agent >= 8 pts, other <= 2 pts",
    "irrational_acceptance": "Agent accepts 0 points",
    "value_destroying": "Agreed joint score < 10",
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_case_types(
    sa: float, sb: float, alloc_a: Dict[str, int],
    scenario_meta: dict,
) -> List[str]:
    matched = []

    scored = scenario_meta['scored']
    if any(sa2 >= sa and sb2 >= sb and (sa2 > sa or sb2 > sb)
           for _, sa2, sb2 in scored):
        matched.append("pareto_dominated")

    if max(sa, sb) >= 8 and min(sa, sb) <= 2:
        matched.append("one_sided_deal")

    if sa == 0 or sb == 0:
        matched.append("irrational_acceptance")

    if sa + sb < 10:
        matched.append("value_destroying")

    return matched


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def check_flags(turns_text: List[str]) -> Dict[str, bool]:
    flags = {}

    flags["prompt_leakage"] = any(
        "Mia Sanders said" in t or "Ava Thompson said" in t
        for t in turns_text
    )

    texts = Counter(t.strip().lower() for t in turns_text)
    flags["repetitive_loop"] = any(c >= 3 for c in texts.values())

    flags["max_turns_hit"] = len(turns_text) >= 16

    return flags


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def format_scenario_line(sc: Scenario) -> str:
    parts = []
    for t in sc.item_types:
        count = sc.item_counts[t]
        name = _pluralize(t) if count != 1 else t
        parts.append(f"{count} {name} (A: {sc.values_a[t]}pt each, B: {sc.values_b[t]}pt each)")
    return "Items: " + ", ".join(parts)


def format_pareto_frontier(scenario_meta: dict) -> str:
    points = sorted(
        set((int(sa), int(sb)) for _, sa, sb in scenario_meta['pareto']),
        key=lambda p: -p[0],
    )
    return "Pareto frontier: " + " ".join(f"({sa},{sb})" for sa, sb in points)


def format_transcript(
    sc: Scenario, sc_idx: int, model: str,
    case_type: str, alloc_a: Dict[str, int],
    atype: str, sa: float, sb: float,
    dialog_turns: List[str], scenario_meta: dict,
) -> str:
    lines = [f"--- Scenario {sc_idx} ---"]
    lines.append(format_scenario_line(sc))

    alloc_json = json.dumps(alloc_a)
    lines.append(
        f"Agreement: {atype} | Alloc A: {alloc_json} | "
        f"Score A: {int(sa)} | Score B: {int(sb)} | Joint: {int(sa + sb)}"
    )
    lines.append(format_pareto_frontier(scenario_meta))
    lines.append("")
    lines.append("CONVERSATION:")

    for i, turn_str in enumerate(dialog_turns):
        speaker = _get_speaker(turn_str)
        text = _get_utterance(turn_str)
        short = speaker.split()[0] if speaker else f"Agent{i % 2}"
        lines.append(f"  Turn {i} [{short}]: {text}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract surprising Deal-or-No-Deal negotiation outcomes for inspection."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_dir = output_dir / "transcripts"
    for ct in CASE_TYPES:
        (transcript_dir / ct).mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios()
    scenario_meta = [compute_scenario_metadata(sc) for sc in scenarios]

    cases: Dict[str, List[dict]] = defaultdict(list)
    csv_rows: List[dict] = []

    for result_dir in RESULT_DIRS:
        if not result_dir.exists():
            print(f"Skipping missing dir: {result_dir}", file=sys.stderr)
            continue

        for pair_dir in sorted(result_dir.iterdir()):
            if not pair_dir.is_dir():
                continue

            model = extract_model_name(pair_dir.name)

            for csv_file in sorted(pair_dir.glob("*.csv")):
                m = re.match(r"(\d+)_temp", csv_file.name)
                if not m:
                    continue
                sc_idx = int(m.group(1))
                if sc_idx >= len(scenarios):
                    continue

                try:
                    dialog_turns = _load_csv_dialog(csv_file)
                except Exception:
                    continue

                sc = scenarios[sc_idx]
                meta = scenario_meta[sc_idx]

                alloc_a, atype, details = parse_episode(dialog_turns, sc)
                if atype not in ('leave', 'implicit'):
                    continue

                sa, sb = score_allocation(alloc_a, sc)
                matched = detect_case_types(sa, sb, alloc_a, meta)
                if not matched:
                    continue

                utterances = [_get_utterance(t) for t in dialog_turns]
                flags = check_flags(utterances)
                n_turns = len(dialog_turns)

                for ct in matched:
                    transcript = format_transcript(
                        sc, sc_idx, model, ct, alloc_a,
                        atype, sa, sb, dialog_turns, meta,
                    )

                    cases[ct].append({
                        "model": model,
                        "scenario_idx": sc_idx,
                        "transcript": transcript,
                        "flags": flags,
                    })

                    csv_rows.append({
                        "case_type": ct,
                        "model": model,
                        "scenario_idx": sc_idx,
                        "n_turns": n_turns,
                        "agreement_type": atype,
                        "score_a": int(sa),
                        "score_b": int(sb),
                        "joint_score": int(sa + sb),
                        "alloc_a": json.dumps(alloc_a),
                    })

    # -- Transcripts --
    for ct, items in cases.items():
        by_model: Dict[str, List[dict]] = defaultdict(list)
        for item in items:
            by_model[item["model"]].append(item)

        for model, model_items in sorted(by_model.items()):
            safe = model.replace("/", "_").replace(" ", "_")
            out_path = transcript_dir / ct / f"{safe}.txt"
            with open(out_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"MODEL: {model}\n")
                f.write(f"CASE TYPE: {ct}\n")
                f.write(f"COUNT: {len(model_items)} episodes\n")
                f.write("=" * 80 + "\n\n")
                for item in sorted(model_items, key=lambda x: x["scenario_idx"]):
                    f.write(item["transcript"])
                    f.write("\n")

    print(f"Wrote transcripts to {transcript_dir}")

    # -- CSV --
    csv_path = output_dir / "error_cases_summary.csv"
    csv_cols = [
        "case_type", "model", "scenario_idx", "n_turns",
        "agreement_type", "score_a", "score_b", "joint_score", "alloc_a",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path} ({len(csv_rows)} rows)")

    # -- Report --
    report_path = output_dir / "error_cases_report.txt"
    with open(report_path, "w") as rpt:
        rpt.write("DEAL-OR-NO-DEAL ERROR CASES REPORT\n")
        rpt.write("=" * 60 + "\n\n")

        rpt.write("CASE TYPE COUNTS\n")
        rpt.write("-" * 60 + "\n")
        for ct, desc in CASE_TYPES.items():
            n = len(cases.get(ct, []))
            rpt.write(f"  {ct:<30} {n:>6}  ({desc})\n")
        rpt.write(f"  {'TOTAL':<30} {sum(len(v) for v in cases.values()):>6}\n\n")

        rpt.write("BY MODEL PER CASE TYPE\n")
        rpt.write("-" * 60 + "\n")
        for ct in CASE_TYPES:
            items = cases.get(ct, [])
            if not items:
                continue
            rpt.write(f"\n  {ct}:\n")
            model_counts = Counter(item["model"] for item in items)
            for model, count in model_counts.most_common():
                rpt.write(f"    {model:<45} {count:>5}\n")

        rpt.write("\n\nQUALITY FLAG CROSS-TABULATION\n")
        rpt.write("-" * 60 + "\n")
        flag_names = ["prompt_leakage", "repetitive_loop", "max_turns_hit"]
        for ct in CASE_TYPES:
            items = cases.get(ct, [])
            n = len(items)
            if n == 0:
                continue
            rpt.write(f"\n  {ct} (n={n}):\n")
            for fn in flag_names:
                count = sum(1 for item in items if item["flags"].get(fn))
                pct = count / n * 100
                rpt.write(f"    {fn:<30} {count:>5} ({pct:>5.1f}%)\n")
            any_flag = sum(
                1 for item in items
                if any(item["flags"].get(fn) for fn in flag_names)
            )
            rpt.write(f"    {'any_flag':<30} {any_flag:>5} ({any_flag / n * 100:>5.1f}%)\n")
            clean = n - any_flag
            rpt.write(f"    {'clean (no flags)':<30} {clean:>5} ({clean / n * 100:>5.1f}%)\n")

    print(f"Wrote {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
