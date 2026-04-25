#!/usr/bin/env python3
"""
Conversation quality audit for false-belief dialog experiments.

Applies 18 automated quality detectors (3 tiers) to all behavioral_summary.json
files and produces:
  - quality_flags_all.csv          (per-episode flags)
  - quality_summary_by_pair.csv    (per pair × condition rates)
  - quality_summary_by_model.csv   (per model × role rates)
  - clean_vs_original_accuracy.png (side-by-side accuracy comparison)
  - quality_audit_report.txt       (human-readable summary)
  - flagged_examples/{flag}/       (up to 10 transcript examples per flag)
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "logit_lens_results_false_belief_100"
OUT_DIR = REPO_ROOT / "summary_plots_false_belief_100" / "quality_audit"

CONDITIONS = [
    "shared_help", "shared_deceive",
    "ignorance_help", "ignorance_deceive",
    "false_help", "false_deceive",
]

CONDITION_LABELS = {
    "shared_help": "Shared\nHelp",
    "shared_deceive": "Shared\nDeceive",
    "ignorance_help": "Ignorance\nHelp",
    "ignorance_deceive": "Ignorance\nDeceive",
    "false_help": "False Belief\nHelp",
    "false_deceive": "False Belief\nDeceive",
}

BELIEF_SHORT = {
    "shared_truth": "shared",
    "ignorance": "ignorance",
    "false_belief": "false",
}


# ---------------------------------------------------------------------------
# Pair / condition parsing helpers
# ---------------------------------------------------------------------------

def parse_pair(pair_dir_name: str) -> Tuple[str, str]:
    parts = pair_dir_name.split("_None_0_")
    if len(parts) < 2:
        return pair_dir_name, ""
    m1 = parts[0]
    m2 = parts[1].split("_false_belief")[0]
    return m1, m2


def pair_label(m1: str, m2: str) -> str:
    return f"{m1} x {m2}"


def condition_from_meta(meta: Dict[str, Any]) -> str:
    b = BELIEF_SHORT.get(meta.get("belief_condition", ""), meta.get("belief_condition", ""))
    g = meta.get("goal_condition", "")
    return f"{b}_{g}"


def parse_episode_dir(ep_name: str) -> Tuple[int, str]:
    parts = ep_name.split("_", 2)
    idx = int(parts[1])
    codename = parts[2] if len(parts) > 2 else ""
    return idx, codename


# ---------------------------------------------------------------------------
# Quality detectors — each returns bool
# ---------------------------------------------------------------------------

def detect_no_leave(d: Dict) -> bool:
    return not any("leave" in t["text"].lower() for t in d["turns"])


def detect_null_choice(d: Dict) -> bool:
    return d.get("final_choice") is None


def detect_guide_leave_choice(d: Dict) -> bool:
    for t in reversed(d["turns"]):
        if "leave" in t["text"].lower():
            return t["speaker"] == "Mia Sanders"
    return False


def detect_ambiguous_leave(d: Dict) -> bool:
    meta = d["scenario_meta"]
    la, lb = meta["loc_a"].lower(), meta["loc_b"].lower()
    for t in reversed(d["turns"]):
        text = t["text"].lower()
        if "leave" in text:
            return la in text and lb in text
    return False


def detect_negation_choice(d: Dict) -> bool:
    meta = d["scenario_meta"]
    la, lb = meta["loc_a"].lower(), meta["loc_b"].lower()
    neg_patterns = [
        r"not\s+(?:on|in|at|the)\s+",
        r"isn'?t\s+(?:on|in|at|the)\s+",
        r"don'?t\s+.*",
        r"no,?\s+(?:the\s+)?",
    ]
    for t in reversed(d["turns"]):
        text = t["text"].lower()
        if "leave" not in text:
            continue
        has_a = la in text
        has_b = lb in text
        if has_a and not has_b:
            for pat in neg_patterns:
                if re.search(pat + re.escape(la), text):
                    return True
        if has_b and not has_a:
            for pat in neg_patterns:
                if re.search(pat + re.escape(lb), text):
                    return True
        break
    return False


def detect_substring_match(d: Dict) -> bool:
    meta = d["scenario_meta"]
    for loc in [meta["loc_a"].lower(), meta["loc_b"].lower()]:
        word_pat = re.compile(r"(?<!\w)" + re.escape(loc) + r"(?!\w)")
        for t in d["turns"]:
            text = t["text"].lower()
            if loc in text and not word_pat.search(text):
                return True
    return False


def detect_prompt_leakage(d: Dict) -> bool:
    for t in d["turns"]:
        text = t["text"]
        if "Mia Sanders said" in text or "Ava Thompson said" in text:
            return True
    return False


def detect_cascade_leakage(d: Dict) -> bool:
    for t in d["turns"]:
        text = t["text"]
        count = text.count("Mia Sanders said") + text.count("Ava Thompson said")
        if count >= 3:
            return True
    return False


def detect_repetitive_loop(d: Dict) -> bool:
    texts = Counter(t["text"].strip().lower() for t in d["turns"])
    return any(c >= 3 for c in texts.values())


def detect_max_turns_hit(d: Dict) -> bool:
    return d.get("n_turns", len(d["turns"])) >= 16


def detect_single_word_seeker(d: Dict) -> bool:
    count = sum(
        1 for t in d["turns"]
        if t["speaker"] == "Ava Thompson" and len(t["text"].split()) == 1
    )
    return count >= 3


def detect_guide_says_leave(d: Dict) -> bool:
    for t in d["turns"]:
        if t["speaker"] == "Mia Sanders" and "LEAVE" in t["text"]:
            return True
    return False


def detect_multi_leave(d: Dict) -> bool:
    count = sum(1 for t in d["turns"] if "LEAVE" in t["text"])
    return count > 1


def detect_ava_parroting(d: Dict) -> bool:
    turns = d["turns"]
    for i in range(1, len(turns)):
        if turns[i]["speaker"] != turns[i - 1]["speaker"]:
            a = turns[i]["text"].strip().lower()
            b = turns[i - 1]["text"].strip().lower()
            if a == b and len(a) > 10:
                return True
    return False


def detect_parenthetical_aside(d: Dict) -> bool:
    keywords = ["my goal", "strategy", "note:", "because i", "since i",
                 "fulfilling", "assuming", "internal"]
    paren_re = re.compile(r"\(([^)]+)\)")
    for t in d["turns"]:
        for m in paren_re.finditer(t["text"].lower()):
            inner = m.group(1)
            if any(kw in inner for kw in keywords):
                return True
    return False


def detect_guide_goal_failure(d: Dict) -> bool:
    meta = d["scenario_meta"]
    if not d["turns"]:
        return False
    t0 = d["turns"][0]
    if t0["speaker"] != "Mia Sanders":
        return False
    text = t0["text"].lower()
    la, lb = meta["loc_a"].lower(), meta["loc_b"].lower()
    true_loc = meta["true_location"].replace("_", " ").lower()
    false_loc = lb if la == true_loc else la
    has_true = true_loc in text or la.lower() in text and la.lower().replace(" ", "_") == meta["true_location"]
    has_false = false_loc in text

    # Re-check with proper display names
    true_display = la if la.replace(" ", "_") == meta["true_location"] else lb
    false_display = lb if true_display == la else la
    has_true = true_display.lower() in text
    has_false = false_display.lower() in text

    # Check for negation: "not on the X" / "isn't on X" means the mention
    # is actually steering AWAY from X, so it's not a goal failure.
    neg_re = re.compile(
        r"(?:not|n['’]t|no|isn['’]t|don['’]t|never)\s"
    )

    def _is_negated(loc: str, txt: str) -> bool:
        idx = txt.find(loc)
        if idx < 0:
            return False
        prefix = txt[max(0, idx - 30):idx]
        return bool(neg_re.search(prefix))

    goal = meta["goal_condition"]
    if goal == "help" and has_false and not has_true:
        if _is_negated(false_display.lower(), text):
            return False
        return True
    if goal == "deceive" and has_true and not has_false:
        if _is_negated(true_display.lower(), text):
            return False
        return True
    return False


def detect_seeker_contradicts_knowledge(d: Dict) -> bool:
    meta = d["scenario_meta"]
    return meta["belief_condition"] == "shared_truth" and not d["correct"]


def detect_meta_commentary(d: Dict) -> bool:
    phrases = ["my goal", "my objective", "according to my instructions",
               "i was told to", "i am playing", "as per the scenario"]
    for t in d["turns"]:
        text = t["text"].lower()
        if any(p in text for p in phrases):
            return True
    return False


TIER1_DETECTORS = [
    ("no_leave", detect_no_leave),
    ("null_choice", detect_null_choice),
    ("guide_leave_choice", detect_guide_leave_choice),
    ("ambiguous_leave", detect_ambiguous_leave),
    ("negation_choice", detect_negation_choice),
    ("substring_match", detect_substring_match),
]

TIER2_DETECTORS = [
    ("prompt_leakage", detect_prompt_leakage),
    ("cascade_leakage", detect_cascade_leakage),
    ("repetitive_loop", detect_repetitive_loop),
    ("max_turns_hit", detect_max_turns_hit),
    ("single_word_seeker", detect_single_word_seeker),
    ("guide_says_leave", detect_guide_says_leave),
    ("multi_leave", detect_multi_leave),
    ("ava_parroting", detect_ava_parroting),
    ("parenthetical_aside", detect_parenthetical_aside),
]

TIER3_DETECTORS = [
    ("guide_goal_failure", detect_guide_goal_failure),
    ("seeker_contradicts_knowledge", detect_seeker_contradicts_knowledge),
    ("meta_commentary", detect_meta_commentary),
]

ALL_DETECTORS = TIER1_DETECTORS + TIER2_DETECTORS + TIER3_DETECTORS
ALL_FLAG_NAMES = [name for name, _ in ALL_DETECTORS]
TIER1_NAMES = {name for name, _ in TIER1_DETECTORS}
TIER2_NAMES = {name for name, _ in TIER2_DETECTORS}
TIER3_NAMES = {name for name, _ in TIER3_DETECTORS}


# ---------------------------------------------------------------------------
# Run all detectors on one episode
# ---------------------------------------------------------------------------

def run_detectors(d: Dict) -> Dict[str, bool]:
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
# Format a transcript for flagged examples
# ---------------------------------------------------------------------------

def format_transcript(d: Dict, ep_name: str, flags: Dict[str, bool]) -> str:
    meta = d["scenario_meta"]
    lines = [f"--- {ep_name} ---"]
    lines.append(
        f"Item: {meta['item']} | True location: {meta['true_location']} | "
        f"Ava belief: {meta['b_belief']} | Final choice: {d['final_choice']} | "
        f"Correct: {d['correct']}"
    )
    lines.append(f"Condition: {meta['belief_condition']} + {meta['goal_condition']}")
    lines.append("")
    for t in d["turns"]:
        speaker = "Mia" if t["speaker"] == "Mia Sanders" else "Ava"
        lines.append(f"  Turn {t['turn']} [{speaker}]: {t['text']}")
    active = [n for n in ALL_FLAG_NAMES if flags.get(n)]
    if active:
        lines.append(f"  ** FLAGS: {', '.join(active)} **")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RESULTS_ROOT.exists():
        print(f"Results directory not found: {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples_dir = OUT_DIR / "flagged_examples"
    for name in ALL_FLAG_NAMES:
        (examples_dir / name).mkdir(parents=True, exist_ok=True)

    # Collect all rows
    rows: List[Dict[str, Any]] = []
    example_counts: Dict[str, int] = defaultdict(int)
    example_texts: Dict[str, List[str]] = defaultdict(list)

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
            cond = condition_from_meta(meta)
            flags = run_detectors(d)

            row = {
                "pair": pair_label(m1, m2),
                "model_1": m1,
                "model_2": m2,
                "scenario": meta.get("scenario_type", codename),
                "condition": cond,
                "episode_idx": ep_idx,
                "correct": int(d["correct"]),
                "final_choice": d.get("final_choice", ""),
                "n_turns": d.get("n_turns", len(d["turns"])),
            }
            row.update({n: int(flags[n]) for n in ALL_FLAG_NAMES})
            row["tier1_count"] = flags["tier1_count"]
            row["tier2_count"] = flags["tier2_count"]
            row["tier3_count"] = flags["tier3_count"]
            row["any_quality_issue"] = int(flags["any_quality_issue"])
            row["measurement_suspect"] = int(flags["measurement_suspect"])
            rows.append(row)

            # Collect flagged examples (up to 10 per flag)
            for name in ALL_FLAG_NAMES:
                if flags.get(name) and example_counts[name] < 10:
                    txt = format_transcript(d, ep_dir.name, flags)
                    example_texts[name].append(txt)
                    example_counts[name] += 1

    print(f"Processed {len(rows)} episodes across {len(pair_dirs)} pairs")

    # ── Write quality_flags_all.csv ──
    csv_cols = (
        ["pair", "model_1", "model_2", "scenario", "condition",
         "episode_idx", "correct", "final_choice", "n_turns"]
        + ALL_FLAG_NAMES
        + ["tier1_count", "tier2_count", "tier3_count",
           "any_quality_issue", "measurement_suspect"]
    )
    csv_path = OUT_DIR / "quality_flags_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    # ── Write flagged examples ──
    for name in ALL_FLAG_NAMES:
        if example_texts[name]:
            out = examples_dir / name / "examples.txt"
            with open(out, "w") as f:
                f.write("\n".join(example_texts[name]))
            print(f"  {name}: {len(example_texts[name])} examples")

    # ── Aggregate summaries ──
    total = len(rows)

    # Overall prevalence
    prevalence = {}
    for name in ALL_FLAG_NAMES:
        c = sum(r[name] for r in rows)
        prevalence[name] = (c, c / total * 100 if total else 0)

    # Per-condition accuracy (original and clean)
    cond_orig: Dict[str, List[int]] = defaultdict(list)
    cond_clean: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        cond_orig[r["condition"]].append(r["correct"])
        if not r["measurement_suspect"]:
            cond_clean[r["condition"]].append(r["correct"])

    # By pair × condition
    pair_cond: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        pair_cond[(r["pair"], r["condition"])].append(r)

    # By model × role
    model_role: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        model_role[(r["model_1"], "guide")].append(r)
        model_role[(r["model_2"], "seeker")].append(r)

    # Co-occurrence matrix
    cooccur = np.zeros((len(ALL_FLAG_NAMES), len(ALL_FLAG_NAMES)), dtype=int)
    for r in rows:
        active = [i for i, n in enumerate(ALL_FLAG_NAMES) if r[n]]
        for i in active:
            for j in active:
                cooccur[i, j] += 1

    # ── Write quality_summary_by_pair.csv ──
    pair_csv_path = OUT_DIR / "quality_summary_by_pair.csv"
    pair_cols = ["pair", "model_1", "model_2", "condition", "n_episodes", "accuracy"]
    pair_cols += [f"{n}_rate" for n in ALL_FLAG_NAMES]
    with open(pair_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pair_cols)
        writer.writeheader()
        for (pair, cond), recs in sorted(pair_cond.items()):
            m1 = recs[0]["model_1"]
            m2 = recs[0]["model_2"]
            n = len(recs)
            acc = sum(r["correct"] for r in recs) / n if n else 0
            row_out = {
                "pair": pair, "model_1": m1, "model_2": m2,
                "condition": cond, "n_episodes": n,
                "accuracy": f"{acc:.4f}",
            }
            for name in ALL_FLAG_NAMES:
                rate = sum(r[name] for r in recs) / n if n else 0
                row_out[f"{name}_rate"] = f"{rate:.4f}"
            writer.writerow(row_out)
    print(f"Wrote {pair_csv_path}")

    # ── Write quality_summary_by_model.csv ──
    model_csv_path = OUT_DIR / "quality_summary_by_model.csv"
    model_cols = ["model", "role", "n_episodes", "accuracy"]
    model_cols += [f"{n}_rate" for n in ALL_FLAG_NAMES]
    with open(model_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_cols)
        writer.writeheader()
        for (model, role), recs in sorted(model_role.items()):
            n = len(recs)
            acc = sum(r["correct"] for r in recs) / n if n else 0
            row_out = {
                "model": model, "role": role,
                "n_episodes": n, "accuracy": f"{acc:.4f}",
            }
            for name in ALL_FLAG_NAMES:
                rate = sum(r[name] for r in recs) / n if n else 0
                row_out[f"{name}_rate"] = f"{rate:.4f}"
            writer.writerow(row_out)
    print(f"Wrote {model_csv_path}")

    # ── Plot: clean vs original accuracy ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CONDITIONS))
    width = 0.35
    orig_means = []
    clean_means = []
    orig_ns = []
    clean_ns = []
    for c in CONDITIONS:
        vals = cond_orig.get(c, [])
        orig_means.append(np.mean(vals) if vals else 0)
        orig_ns.append(len(vals))
        vals_c = cond_clean.get(c, [])
        clean_means.append(np.mean(vals_c) if vals_c else 0)
        clean_ns.append(len(vals_c))

    bars1 = ax.bar(x - width / 2, orig_means, width, label="Original (all episodes)",
                   color="#6baed6", edgecolor="white")
    bars2 = ax.bar(x + width / 2, clean_means, width, label="Clean (excl. Tier 1)",
                   color="#fd8d3c", edgecolor="white")

    for bar, val in zip(bars1, orig_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, clean_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title("Behavioral Accuracy: Original vs. Clean (Tier 1 excluded)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plot_path = OUT_DIR / "clean_vs_original_accuracy.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # ── Write quality_audit_report.txt ──
    report_path = OUT_DIR / "quality_audit_report.txt"
    with open(report_path, "w") as rpt:
        rpt.write("CONVERSATION QUALITY AUDIT REPORT\n")
        rpt.write("=" * 60 + "\n")
        rpt.write(f"Dataset: {total} episodes across {len(set(r['pair'] for r in rows))} model pairs\n\n")

        rpt.write("OVERALL PREVALENCE\n")
        rpt.write("-" * 60 + "\n")
        rpt.write(f"{'Flag':<30} {'Count':>8} {'%':>8}\n")
        for name in ALL_FLAG_NAMES:
            c, pct = prevalence[name]
            tier = "T1" if name in TIER1_NAMES else ("T2" if name in TIER2_NAMES else "T3")
            rpt.write(f"  [{tier}] {name:<26} {c:>8} {pct:>7.1f}%\n")
        any_count = sum(r["any_quality_issue"] for r in rows)
        suspect_count = sum(r["measurement_suspect"] for r in rows)
        rpt.write(f"\n  any_quality_issue            {any_count:>8} {any_count/total*100:>7.1f}%\n")
        rpt.write(f"  measurement_suspect          {suspect_count:>8} {suspect_count/total*100:>7.1f}%\n\n")

        rpt.write("ACCURACY: ORIGINAL vs CLEAN (excl. Tier 1)\n")
        rpt.write("-" * 60 + "\n")
        rpt.write(f"{'Condition':<25} {'Orig Acc':>10} {'N_orig':>8} {'Clean Acc':>10} {'N_clean':>8}\n")
        for i, c in enumerate(CONDITIONS):
            rpt.write(f"  {c:<23} {orig_means[i]:>10.4f} {orig_ns[i]:>8} {clean_means[i]:>10.4f} {clean_ns[i]:>8}\n")
        rpt.write("\n")

        rpt.write("ISSUE PREVALENCE BY GUIDE MODEL (MODEL_1)\n")
        rpt.write("-" * 60 + "\n")
        guides = sorted(set(r["model_1"] for r in rows))
        header = f"{'Model':<40}"
        top_flags = ["prompt_leakage", "repetitive_loop", "max_turns_hit",
                     "no_leave", "guide_goal_failure", "guide_says_leave"]
        for fl in top_flags:
            header += f" {fl[:12]:>12}"
        rpt.write(header + "\n")
        for g in guides:
            recs = [r for r in rows if r["model_1"] == g]
            n = len(recs)
            line = f"  {g:<38}"
            for fl in top_flags:
                rate = sum(r[fl] for r in recs) / n * 100 if n else 0
                line += f" {rate:>11.1f}%"
            rpt.write(line + "\n")
        rpt.write("\n")

        rpt.write("ISSUE PREVALENCE BY SEEKER MODEL (MODEL_2)\n")
        rpt.write("-" * 60 + "\n")
        seekers = sorted(set(r["model_2"] for r in rows))
        header = f"{'Model':<40}"
        seeker_flags = ["prompt_leakage", "repetitive_loop", "max_turns_hit",
                        "no_leave", "seeker_contradicts_knowledge", "single_word_seeker"]
        for fl in seeker_flags:
            header += f" {fl[:12]:>12}"
        rpt.write(header + "\n")
        for s in seekers:
            recs = [r for r in rows if r["model_2"] == s]
            n = len(recs)
            line = f"  {s:<38}"
            for fl in seeker_flags:
                rate = sum(r[fl] for r in recs) / n * 100 if n else 0
                line += f" {rate:>11.1f}%"
            rpt.write(line + "\n")
        rpt.write("\n")

        rpt.write("CO-OCCURRENCE MATRIX (top pairs)\n")
        rpt.write("-" * 60 + "\n")
        pairs_sorted = []
        for i in range(len(ALL_FLAG_NAMES)):
            for j in range(i + 1, len(ALL_FLAG_NAMES)):
                if cooccur[i, j] > 0:
                    pairs_sorted.append((cooccur[i, j], ALL_FLAG_NAMES[i], ALL_FLAG_NAMES[j]))
        pairs_sorted.sort(reverse=True)
        for count, a, b in pairs_sorted[:20]:
            rpt.write(f"  {a:<28} + {b:<28} {count:>6} ({count/total*100:.1f}%)\n")
        rpt.write("\n")

        rpt.write("TOP QUALITY-CONCERN PAIRS (by any_quality_issue rate)\n")
        rpt.write("-" * 60 + "\n")
        pair_rates = []
        for pair in sorted(set(r["pair"] for r in rows)):
            recs = [r for r in rows if r["pair"] == pair]
            n = len(recs)
            rate = sum(r["any_quality_issue"] for r in recs) / n if n else 0
            pair_rates.append((rate, pair, n))
        pair_rates.sort(reverse=True)
        for rate, pair, n in pair_rates[:15]:
            rpt.write(f"  {pair:<60} {rate*100:>6.1f}% ({n} eps)\n")

    print(f"Wrote {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
