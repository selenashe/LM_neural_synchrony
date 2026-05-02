#!/usr/bin/env python3
"""
Quality audit for Deal-or-No-Deal negotiation dialogs.

Applies 15 automated quality detectors (3 tiers) to all dialog CSVs
from sotopia_results_deal_or_no_deal*/dialogs/ and produces:
  - quality_flags_all.csv          (per-episode flags)
  - quality_summary_by_model.csv   (per-model issue rates)
  - clean_vs_original_metrics.png  (side-by-side: agreement rate & joint score)
  - quality_audit_report.txt       (human-readable summary)
  - flagged_examples/{flag}/       (up to 10 transcript examples per flag)
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fao_analysis import (
    load_scenarios, RESULT_DIRS, _load_csv_dialog, _get_speaker, _get_utterance,
    parse_episode, score_allocation, compute_scenario_metadata, extract_model_name,
    parse_items_from_text, PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Detectors — each is (episode_dict) -> bool
# ---------------------------------------------------------------------------

# Tier 1: Measurement validity

def detect_parse_failure(d: Dict) -> bool:
    return d["agreement_type"] == "parse_failure"


def detect_implicit_agreement(d: Dict) -> bool:
    return d["agreement_type"] == "implicit"


def detect_inconsistent_allocation(d: Dict) -> bool:
    if d["alloc_a"] is None:
        return False
    sc = d["scenario"]
    for t in sc.item_types:
        if d["alloc_a"].get(t, 0) > sc.item_counts[t] or d["alloc_a"].get(t, 0) < 0:
            return True
    return False


def detect_zero_value_hoarding(d: Dict) -> bool:
    if d["alloc_a"] is None:
        return False
    sc = d["scenario"]
    for t in sc.item_types:
        a_takes = d["alloc_a"].get(t, 0)
        b_takes = sc.item_counts[t] - a_takes
        if a_takes > 0 and sc.values_a[t] == 0:
            return True
        if b_takes > 0 and sc.values_b[t] == 0:
            return True
    return False


# Tier 2: Dialog quality

def detect_prompt_leakage(d: Dict) -> bool:
    for t in d["turns"]:
        if "Mia Sanders said" in t["text"] or "Ava Thompson said" in t["text"]:
            return True
    return False


def detect_repetitive_loop(d: Dict) -> bool:
    texts = Counter(t["text"].strip().lower() for t in d["turns"])
    return any(c >= 3 for c in texts.values())


def detect_max_turns_hit(d: Dict) -> bool:
    return d["n_turns"] >= 16


def detect_single_word_turns(d: Dict) -> bool:
    by_speaker = defaultdict(int)
    for t in d["turns"]:
        if len(t["text"].split()) == 1:
            by_speaker[t["speaker"]] += 1
    return any(c >= 3 for c in by_speaker.values())


def detect_parenthetical_aside(d: Dict) -> bool:
    keywords = ["my goal", "strategy", "note:", "because i", "since i",
                "fulfilling", "assuming", "internal"]
    paren_re = re.compile(r"\(([^)]+)\)")
    for t in d["turns"]:
        for m in paren_re.finditer(t["text"].lower()):
            if any(kw in m.group(1) for kw in keywords):
                return True
    return False


def detect_no_proposal(d: Dict) -> bool:
    sc = d["scenario"]
    for t in d["turns"]:
        parsed = parse_items_from_text(t["text"], sc)
        if parsed is not None:
            return False
    return True


def detect_ultimatum(d: Dict) -> bool:
    pat = re.compile(r"take\s+it\s+or\s+leave\s+it", re.IGNORECASE)
    for t in d["turns"]:
        if t["turn"] <= 1 and pat.search(t["text"]):
            return True
    return False


def detect_agent_parroting(d: Dict) -> bool:
    turns = d["turns"]
    for i in range(1, len(turns)):
        if turns[i]["speaker"] != turns[i - 1]["speaker"]:
            a = turns[i]["text"].strip().lower()
            b = turns[i - 1]["text"].strip().lower()
            if a == b and len(a) > 10:
                return True
    return False


# Tier 3: Task comprehension

def detect_meta_commentary(d: Dict) -> bool:
    phrases = ["my goal", "my objective", "according to my instructions",
               "i was told to", "i am playing", "as per the scenario"]
    for t in d["turns"]:
        text = t["text"].lower()
        if any(p in text for p in phrases):
            return True
    return False


def detect_value_revelation(d: Dict) -> bool:
    pat1 = re.compile(r"worth\s+\d+\s+point", re.IGNORECASE)
    pat2 = re.compile(r"\d+\s+points?\s+each", re.IGNORECASE)
    for t in d["turns"]:
        if pat1.search(t["text"]) or pat2.search(t["text"]):
            return True
    return False


def detect_exceeds_available(d: Dict) -> bool:
    sc = d["scenario"]
    for t in d["turns"]:
        parsed = parse_items_from_text(t["text"], sc)
        if parsed is None:
            continue
        for item_type, count in parsed.items():
            if count > sc.item_counts.get(item_type, 0):
                return True
    return False


TIER1_DETECTORS = [
    ("parse_failure", detect_parse_failure),
    ("implicit_agreement", detect_implicit_agreement),
    ("inconsistent_allocation", detect_inconsistent_allocation),
    ("zero_value_hoarding", detect_zero_value_hoarding),
]

TIER2_DETECTORS = [
    ("prompt_leakage", detect_prompt_leakage),
    ("repetitive_loop", detect_repetitive_loop),
    ("max_turns_hit", detect_max_turns_hit),
    ("single_word_turns", detect_single_word_turns),
    ("parenthetical_aside", detect_parenthetical_aside),
    ("no_proposal", detect_no_proposal),
    ("ultimatum", detect_ultimatum),
    ("agent_parroting", detect_agent_parroting),
]

TIER3_DETECTORS = [
    ("meta_commentary", detect_meta_commentary),
    ("value_revelation", detect_value_revelation),
    ("exceeds_available", detect_exceeds_available),
]

ALL_DETECTORS = TIER1_DETECTORS + TIER2_DETECTORS + TIER3_DETECTORS
ALL_FLAG_NAMES = [name for name, _ in ALL_DETECTORS]
TIER1_NAMES = {name for name, _ in TIER1_DETECTORS}
TIER2_NAMES = {name for name, _ in TIER2_DETECTORS}
TIER3_NAMES = {name for name, _ in TIER3_DETECTORS}


def run_detectors(d: Dict) -> Dict[str, Any]:
    flags = {}
    for name, fn in ALL_DETECTORS:
        try:
            flags[name] = fn(d)
        except Exception:
            flags[name] = False
    flags["tier1_count"] = sum(1 for n in TIER1_NAMES if flags.get(n))
    flags["tier2_count"] = sum(1 for n in TIER2_NAMES if flags.get(n))
    flags["tier3_count"] = sum(1 for n in TIER3_NAMES if flags.get(n))
    flags["any_quality_issue"] = any(flags.get(n) for n in ALL_FLAG_NAMES)
    flags["measurement_suspect"] = flags["tier1_count"] > 0
    return flags


# ---------------------------------------------------------------------------
# Transcript formatting for flagged examples
# ---------------------------------------------------------------------------

def format_transcript(d: Dict, flags: Dict[str, bool]) -> str:
    sc = d["scenario"]
    meta = d["scenario_meta"]

    item_parts = []
    for t in sc.item_types:
        item_parts.append(
            f"{sc.item_counts[t]} {t} (A={sc.values_a[t]}pt, B={sc.values_b[t]}pt)"
        )
    items_str = ", ".join(item_parts)

    score_a = d["score_a"]
    score_b = d["score_b"]
    joint = score_a + score_b

    lines = [f"--- Model: {d['model']}, Scenario {d['scenario_idx']} ---"]
    lines.append(f"Items: {items_str}")
    lines.append(
        f"Agreement: {d['agreement_type']} | Score A: {score_a} | "
        f"Score B: {score_b} | Joint: {joint}"
    )
    lines.append(f"Turns: {d['n_turns']}")
    lines.append("")
    for t in d["turns"]:
        speaker = "Mia" if t["speaker"] == "Mia Sanders" else (
            "Ava" if t["speaker"] == "Ava Thompson" else t["speaker"])
        lines.append(f"  Turn {t['turn']} [{speaker}]: {t['text']}")
    active = [n for n in ALL_FLAG_NAMES if flags.get(n)]
    if active:
        lines.append(f"  ** FLAGS: {', '.join(active)} **")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clean_vs_original(model_rows: Dict[str, List[Dict]], out_dir: Path) -> None:
    models = sorted(model_rows.keys())
    n = len(models)

    orig_agree = []
    clean_agree = []
    orig_joint = []
    clean_joint = []

    for model in models:
        recs = model_rows[model]
        n_total = len(recs)
        agreed_orig = sum(1 for r in recs if r["agreement_type"] in ("leave", "implicit"))
        orig_agree.append(agreed_orig / n_total if n_total else 0)
        orig_joint.append(
            sum(r["score_a"] + r["score_b"] for r in recs) / n_total if n_total else 0
        )

        clean = [r for r in recs if not r["measurement_suspect"]]
        n_clean = len(clean)
        agreed_clean = sum(1 for r in clean if r["agreement_type"] in ("leave", "implicit"))
        clean_agree.append(agreed_clean / n_clean if n_clean else 0)
        clean_joint.append(
            sum(r["score_a"] + r["score_b"] for r in clean) / n_clean if n_clean else 0
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, n * 1.5), 5))
    x = np.arange(n)
    width = 0.35

    ax1.bar(x - width / 2, orig_agree, width, color="#6baed6", edgecolor="black",
            linewidth=0.6, label="Original")
    ax1.bar(x + width / 2, clean_agree, width, color="#fd8d3c", edgecolor="black",
            linewidth=0.6, label="Clean (excl Tier 1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7, rotation=45, ha="right")
    ax1.set_ylabel("Agreement Rate")
    ax1.set_title("Agreement Rate: Original vs Clean")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x - width / 2, orig_joint, width, color="#6baed6", edgecolor="black",
            linewidth=0.6, label="Original")
    ax2.bar(x + width / 2, clean_joint, width, color="#fd8d3c", edgecolor="black",
            linewidth=0.6, label="Clean (excl Tier 1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Mean Joint Score")
    ax2.set_title("Mean Joint Score: Original vs Clean")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Deal-or-No-Deal Quality: Original vs Clean Metrics", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "clean_vs_original_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quality audit for Deal-or-No-Deal negotiation dialogs.")
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "summary_plots_deal_or_no_deal" / "quality_audit"),
        help="Output directory for audit results.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "flagged_examples"
    for name in ALL_FLAG_NAMES:
        (examples_dir / name).mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios()
    scenario_meta = [compute_scenario_metadata(sc) for sc in scenarios]

    rows: List[Dict[str, Any]] = []
    episode_dicts: List[Dict] = []
    example_counts: Dict[str, int] = defaultdict(int)
    example_texts: Dict[str, List[str]] = defaultdict(list)

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
                    dialog_turns_raw = _load_csv_dialog(csv_file)
                except Exception as e:
                    print(f"Failed to load {csv_file}: {e}", file=sys.stderr)
                    continue

                sc = scenarios[sc_idx]
                meta = scenario_meta[sc_idx]

                turns = []
                for i, turn_str in enumerate(dialog_turns_raw):
                    speaker = _get_speaker(turn_str)
                    text = _get_utterance(turn_str)
                    turns.append({
                        "turn": i,
                        "speaker": speaker or "Unknown",
                        "text": text,
                    })

                alloc_a, agreement_type, _details = parse_episode(dialog_turns_raw, sc)

                if alloc_a is not None:
                    sa, sb = score_allocation(alloc_a, sc)
                else:
                    sa, sb = 0.0, 0.0

                ep = {
                    "turns": turns,
                    "scenario": sc,
                    "scenario_meta": meta,
                    "alloc_a": alloc_a,
                    "agreement_type": agreement_type,
                    "score_a": sa,
                    "score_b": sb,
                    "n_turns": len(turns),
                    "model": model,
                    "scenario_idx": sc_idx,
                }

                flags = run_detectors(ep)

                row = {
                    "model": model,
                    "scenario_idx": sc_idx,
                    "n_turns": len(turns),
                    "agreement_type": agreement_type,
                    "score_a": sa,
                    "score_b": sb,
                    "joint_score": sa + sb,
                    "measurement_suspect": int(flags["measurement_suspect"]),
                }
                row.update({n: int(flags[n]) for n in ALL_FLAG_NAMES})
                row["tier1_count"] = flags["tier1_count"]
                row["tier2_count"] = flags["tier2_count"]
                row["tier3_count"] = flags["tier3_count"]
                row["any_quality_issue"] = int(flags["any_quality_issue"])
                rows.append(row)
                episode_dicts.append(ep)

                for name in ALL_FLAG_NAMES:
                    if flags.get(name) and example_counts[name] < 10:
                        txt = format_transcript(ep, flags)
                        example_texts[name].append(txt)
                        example_counts[name] += 1

    total = len(rows)
    if total == 0:
        print("No episodes found. Check RESULT_DIRS.", file=sys.stderr)
        sys.exit(1)

    print(f"Processed {total} episodes")

    # -- quality_flags_all.csv --
    csv_cols = (
        ["model", "scenario_idx", "n_turns", "agreement_type",
         "score_a", "score_b", "joint_score"]
        + ALL_FLAG_NAMES
        + ["tier1_count", "tier2_count", "tier3_count",
           "any_quality_issue", "measurement_suspect"]
    )
    csv_path = out_dir / "quality_flags_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # -- flagged examples --
    for name in ALL_FLAG_NAMES:
        if example_texts[name]:
            p = examples_dir / name / "examples.txt"
            with open(p, "w") as f:
                f.write("\n".join(example_texts[name]))
            print(f"  {name}: {len(example_texts[name])} examples")

    # -- quality_summary_by_model.csv --
    model_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        model_groups[r["model"]].append(r)

    model_csv_path = out_dir / "quality_summary_by_model.csv"
    model_cols = ["model", "n_episodes", "agreement_rate"]
    model_cols += [f"{n}_rate" for n in ALL_FLAG_NAMES]
    with open(model_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_cols)
        writer.writeheader()
        for model in sorted(model_groups.keys()):
            recs = model_groups[model]
            n = len(recs)
            n_agreed = sum(
                1 for r in recs
                if r["agreement_type"] in ("leave", "implicit")
            )
            row_out = {
                "model": model,
                "n_episodes": n,
                "agreement_rate": f"{n_agreed / n:.4f}" if n else "0",
            }
            for name in ALL_FLAG_NAMES:
                rate = sum(r[name] for r in recs) / n if n else 0
                row_out[f"{name}_rate"] = f"{rate:.4f}"
            writer.writerow(row_out)
    print(f"Wrote {model_csv_path}")

    # -- Collect episode dicts per model (for clean vs original plot) --
    model_ep_rows: Dict[str, List[Dict]] = defaultdict(list)
    for r, ep in zip(rows, episode_dicts):
        model_ep_rows[r["model"]].append({
            "agreement_type": r["agreement_type"],
            "score_a": r["score_a"],
            "score_b": r["score_b"],
            "measurement_suspect": r["measurement_suspect"],
        })

    plot_clean_vs_original(model_ep_rows, out_dir)
    print(f"Wrote {out_dir / 'clean_vs_original_metrics.png'}")

    # -- Prevalence --
    prevalence = {}
    for name in ALL_FLAG_NAMES:
        c = sum(r[name] for r in rows)
        prevalence[name] = (c, c / total * 100 if total else 0)

    # -- Co-occurrence matrix --
    cooccur = np.zeros((len(ALL_FLAG_NAMES), len(ALL_FLAG_NAMES)), dtype=int)
    for r in rows:
        active = [i for i, n in enumerate(ALL_FLAG_NAMES) if r[n]]
        for i in active:
            for j in active:
                cooccur[i, j] += 1

    # -- quality_audit_report.txt --
    report_path = out_dir / "quality_audit_report.txt"
    models = sorted(model_groups.keys())
    with open(report_path, "w") as rpt:
        rpt.write("DEAL-OR-NO-DEAL QUALITY AUDIT REPORT\n")
        rpt.write("=" * 60 + "\n")
        rpt.write(f"Dataset: {total} episodes across {len(models)} models\n\n")

        rpt.write("OVERALL PREVALENCE\n")
        rpt.write("-" * 60 + "\n")
        rpt.write(f"{'Flag':<30} {'Count':>8} {'%':>8}\n")
        for name in ALL_FLAG_NAMES:
            c, pct = prevalence[name]
            tier = ("T1" if name in TIER1_NAMES
                    else ("T2" if name in TIER2_NAMES else "T3"))
            rpt.write(f"  [{tier}] {name:<26} {c:>8} {pct:>7.1f}%\n")
        any_count = sum(r["any_quality_issue"] for r in rows)
        suspect_count = sum(r["measurement_suspect"] for r in rows)
        rpt.write(f"\n  any_quality_issue            {any_count:>8} "
                  f"{any_count / total * 100:>7.1f}%\n")
        rpt.write(f"  measurement_suspect          {suspect_count:>8} "
                  f"{suspect_count / total * 100:>7.1f}%\n\n")

        rpt.write("METRICS: ORIGINAL vs CLEAN (excl. Tier 1) BY MODEL\n")
        rpt.write("-" * 60 + "\n")
        rpt.write(f"{'Model':<40} {'Orig AR':>8} {'Clean AR':>9} "
                  f"{'Orig JS':>8} {'Clean JS':>9} {'N':>5} {'N_cl':>5}\n")
        for model in models:
            recs = model_groups[model]
            n_all = len(recs)
            n_agreed_all = sum(
                1 for r in recs
                if r["agreement_type"] in ("leave", "implicit")
            )
            js_all = sum(r["score_a"] + r["score_b"] for r in recs)
            ar_all = n_agreed_all / n_all if n_all else 0
            js_mean_all = js_all / n_all if n_all else 0

            clean = [r for r in recs if not r["measurement_suspect"]]
            n_cl = len(clean)
            n_agreed_cl = sum(
                1 for r in clean
                if r["agreement_type"] in ("leave", "implicit")
            )
            js_cl = sum(r["score_a"] + r["score_b"] for r in clean)
            ar_cl = n_agreed_cl / n_cl if n_cl else 0
            js_mean_cl = js_cl / n_cl if n_cl else 0

            rpt.write(f"  {model:<38} {ar_all:>8.3f} {ar_cl:>9.3f} "
                      f"{js_mean_all:>8.2f} {js_mean_cl:>9.2f} "
                      f"{n_all:>5} {n_cl:>5}\n")
        rpt.write("\n")

        rpt.write("ISSUE PREVALENCE BY MODEL\n")
        rpt.write("-" * 60 + "\n")
        top_flags = ["parse_failure", "implicit_agreement", "zero_value_hoarding",
                     "prompt_leakage", "repetitive_loop", "max_turns_hit",
                     "no_proposal", "meta_commentary", "value_revelation"]
        header = f"{'Model':<35}"
        for fl in top_flags:
            header += f" {fl[:14]:>14}"
        rpt.write(header + "\n")
        for model in models:
            recs = model_groups[model]
            n = len(recs)
            line = f"  {model:<33}"
            for fl in top_flags:
                rate = sum(r[fl] for r in recs) / n * 100 if n else 0
                line += f" {rate:>13.1f}%"
            rpt.write(line + "\n")
        rpt.write("\n")

        rpt.write("CO-OCCURRENCE MATRIX (top 20 pairs)\n")
        rpt.write("-" * 60 + "\n")
        pairs_sorted = []
        for i in range(len(ALL_FLAG_NAMES)):
            for j in range(i + 1, len(ALL_FLAG_NAMES)):
                if cooccur[i, j] > 0:
                    pairs_sorted.append(
                        (cooccur[i, j], ALL_FLAG_NAMES[i], ALL_FLAG_NAMES[j]))
        pairs_sorted.sort(reverse=True)
        for count, a, b in pairs_sorted[:20]:
            rpt.write(f"  {a:<28} + {b:<28} {count:>6} "
                      f"({count / total * 100:.1f}%)\n")
        rpt.write("\n")

    print(f"Wrote {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
