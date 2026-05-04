#!/usr/bin/env python3
"""
Conversation quality audit for false-belief v3 dialog experiments.

V3 false-belief scenarios (24 conditions: 3 beliefs × 8 trust manipulations).

Applies 18 automated quality detectors (3 tiers) to all behavioral_summary.json
files and produces:
  - quality_flags_all.csv          (per-episode flags)
  - quality_summary_by_pair.csv    (per pair × condition rates)
  - quality_summary_by_model.csv   (per model × role rates)
  - clean_vs_original_accuracy.png (side-by-side accuracy comparison)
  - quality_audit_report.txt       (human-readable summary)
  - flagged_examples/{flag}/       (up to 10 transcript examples per flag)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "logit_lens_results_false_belief_v3"
OUT_DIR = REPO_ROOT / "summary_plots_false_belief_v3" / "quality_audit"

CONDITIONS = [
    "mutual_A1_expert", "mutual_A2_unreliable", "mutual_A3_incentivized", "mutual_A4_malicious",
    "mutual_B1_forgetting", "mutual_B2_intervening", "mutual_B3_poor_vis", "mutual_C_cooperate",
    "asym_A1_expert", "asym_A2_unreliable", "asym_A3_incentivized", "asym_A4_malicious",
    "asym_B1_forgetting", "asym_B2_intervening", "asym_B3_poor_vis", "asym_C_cooperate",
    "conflict_A1_expert", "conflict_A2_unreliable", "conflict_A3_incentivized", "conflict_A4_malicious",
    "conflict_B1_forgetting", "conflict_B2_intervening", "conflict_B3_poor_vis", "conflict_C_cooperate",
]

CONDITION_LABELS = {
    "mutual_A1_expert": "Mutual A1-Expert",
    "mutual_A2_unreliable": "Mutual A2-Unreliable",
    "mutual_A3_incentivized": "Mutual A3-Incentiv.",
    "mutual_A4_malicious": "Mutual A4-Malicious",
    "mutual_B1_forgetting": "Mutual B1-Forgetting",
    "mutual_B2_intervening": "Mutual B2-Interven.",
    "mutual_B3_poor_vis": "Mutual B3-PoorVis",
    "mutual_C_cooperate": "Mutual C-Cooperate",
    "asym_A1_expert": "Asym A1-Expert",
    "asym_A2_unreliable": "Asym A2-Unreliable",
    "asym_A3_incentivized": "Asym A3-Incentiv.",
    "asym_A4_malicious": "Asym A4-Malicious",
    "asym_B1_forgetting": "Asym B1-Forgetting",
    "asym_B2_intervening": "Asym B2-Interven.",
    "asym_B3_poor_vis": "Asym B3-PoorVis",
    "asym_C_cooperate": "Asym C-Cooperate",
    "conflict_A1_expert": "Conflict A1-Expert",
    "conflict_A2_unreliable": "Conflict A2-Unreliable",
    "conflict_A3_incentivized": "Conflict A3-Incentiv.",
    "conflict_A4_malicious": "Conflict A4-Malicious",
    "conflict_B1_forgetting": "Conflict B1-Forgetting",
    "conflict_B2_intervening": "Conflict B2-Interven.",
    "conflict_B3_poor_vis": "Conflict B3-PoorVis",
    "conflict_C_cooperate": "Conflict C-Cooperate",
}

BELIEF_SHORT = {
    "mutual_knowledge": "mutual",
    "asymmetric_knowledge": "asym",
    "conflicting_evidence": "conflict",
}

GOAL_SHORT = {
    "cooperate": "coop",
    "adversarial": "adv",
}


# ---------------------------------------------------------------------------
# Pair / condition parsing helpers
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


def condition_from_meta(meta: Dict[str, Any]) -> str:
    cond = meta.get("condition", "")
    if cond:
        return cond
    b = BELIEF_SHORT.get(meta.get("belief_condition", ""), meta.get("belief_condition", ""))
    g = GOAL_SHORT.get(meta.get("goal_condition", ""), meta.get("goal_condition", ""))
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


def detect_ambiguous_leave(d: Dict) -> bool:
    meta = d["scenario_meta"]
    la, lb = meta["loc_a"].lower(), meta["loc_b"].lower()
    for t in reversed(d["turns"]):
        text = t["text"].lower()
        if "leave" in text:
            return la in text and lb in text
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

    true_display = la if la.replace(" ", "_") == meta["true_location"] else lb
    false_display = lb if true_display == la else la
    has_true = true_display.lower() in text
    has_false = false_display.lower() in text

    neg_re = re.compile(r"(?:not|n['']t|no|isn['']t|don['']t|never)\s")

    def _is_negated(loc: str, txt: str) -> bool:
        idx = txt.find(loc)
        if idx < 0:
            return False
        prefix = txt[max(0, idx - 30):idx]
        return bool(neg_re.search(prefix))

    goal = meta["goal_condition"]
    if goal == "cooperate" and has_false and not has_true:
        if _is_negated(false_display.lower(), text):
            return False
        return True
    if goal == "adversarial" and has_true and not has_false:
        if _is_negated(true_display.lower(), text):
            return False
        return True
    return False


def detect_meta_commentary(d: Dict) -> bool:
    phrases = ["my goal", "my objective", "according to my instructions",
               "i was told to", "i am playing", "as per the scenario"]
    for t in d["turns"]:
        text = t["text"].lower()
        if any(p in text for p in phrases):
            return True
    return False


TIER1_DETECTORS = [
    ("guide_goal_failure", detect_guide_goal_failure),
    ("guide_says_leave", detect_guide_says_leave),
]

TIER2_DETECTORS = [
    ("no_leave", detect_no_leave),
    ("null_choice", detect_null_choice),
    ("ambiguous_leave", detect_ambiguous_leave),
    ("substring_match", detect_substring_match),
    ("prompt_leakage", detect_prompt_leakage),
    ("cascade_leakage", detect_cascade_leakage),
    ("repetitive_loop", detect_repetitive_loop),
    ("max_turns_hit", detect_max_turns_hit),
    ("single_word_seeker", detect_single_word_seeker),
    ("multi_leave", detect_multi_leave),
    ("ava_parroting", detect_ava_parroting),
    ("parenthetical_aside", detect_parenthetical_aside),
    ("meta_commentary", detect_meta_commentary),
]

TIER3_DETECTORS = []

ALL_DETECTORS = TIER1_DETECTORS + TIER2_DETECTORS + TIER3_DETECTORS
ALL_FLAG_NAMES = [name for name, _ in ALL_DETECTORS]
TIER1_NAMES = {name for name, _ in TIER1_DETECTORS}
TIER2_NAMES = {name for name, _ in TIER2_DETECTORS}
TIER3_NAMES = {name for name, _ in TIER3_DETECTORS}


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


def format_transcript(d: Dict, ep_name: str, flags: Dict[str, bool]) -> str:
    meta = d["scenario_meta"]
    b_belief = meta.get("b_belief", meta.get("b_evidence", ""))
    lines = [f"--- {ep_name} ---"]
    lines.append(
        f"Item: {meta['item']} | True location: {meta['true_location']} | "
        f"Ava belief: {b_belief} | Final choice: {d['final_choice']} | "
        f"Correct: {d['correct']}"
    )
    lines.append(f"Condition: {meta.get('condition', '')} | "
                 f"Belief: {meta['belief_condition']} | Goal: {meta['goal_condition']} | "
                 f"Trust: {meta.get('trust_manipulation', '')}")
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
# Model attribution
# ---------------------------------------------------------------------------

GUIDE_ERRORS = {"guide_goal_failure", "guide_says_leave"}
SEEKER_ERRORS = {"no_leave", "null_choice", "single_word_seeker", "ava_parroting"}
JOINT_ERRORS = {"repetitive_loop", "max_turns_hit", "ambiguous_leave", "substring_match"}


def attribute_to_speaker(flag_name: str, d: Dict) -> Optional[str]:
    if flag_name in GUIDE_ERRORS:
        return "guide"
    if flag_name in SEEKER_ERRORS:
        return "seeker"
    if flag_name in JOINT_ERRORS:
        return "both"

    turns = d.get("turns", [])
    if flag_name == "prompt_leakage":
        for t in turns:
            if "Mia Sanders said" in t["text"] or "Ava Thompson said" in t["text"]:
                return "guide" if t["speaker"] == "Mia Sanders" else "seeker"
    elif flag_name == "cascade_leakage":
        for t in turns:
            count = t["text"].count("Mia Sanders said") + t["text"].count("Ava Thompson said")
            if count >= 3:
                return "guide" if t["speaker"] == "Mia Sanders" else "seeker"
    elif flag_name == "parenthetical_aside":
        keywords = ["my goal", "strategy", "note:", "because i", "since i",
                     "fulfilling", "assuming", "internal"]
        paren_re = re.compile(r"\(([^)]+)\)")
        for t in turns:
            for m in paren_re.finditer(t["text"].lower()):
                if any(kw in m.group(1) for kw in keywords):
                    return "guide" if t["speaker"] == "Mia Sanders" else "seeker"
    elif flag_name == "meta_commentary":
        phrases = ["my goal", "my objective", "according to my instructions",
                   "i was told to", "i am playing", "as per the scenario"]
        for t in turns:
            text = t["text"].lower()
            if any(p in text for p in phrases):
                return "guide" if t["speaker"] == "Mia Sanders" else "seeker"
    elif flag_name == "multi_leave":
        guide_leaves = sum(1 for t in turns if t["speaker"] == "Mia Sanders" and "LEAVE" in t["text"])
        seeker_leaves = sum(1 for t in turns if t["speaker"] == "Ava Thompson" and "LEAVE" in t["text"])
        if guide_leaves > 1 and seeker_leaves > 1:
            return "both"
        if guide_leaves > 1:
            return "guide"
        if seeker_leaves > 1:
            return "seeker"
        return "both"

    return "both"


# ---------------------------------------------------------------------------
# Quality audit plots
# ---------------------------------------------------------------------------

def plot_quality_prevalence_overall(rows: List[Dict], prevalence: Dict) -> None:
    names = list(reversed(ALL_FLAG_NAMES))
    pcts = [prevalence[n][1] for n in names]
    colors = ["#e74c3c" if n in TIER1_NAMES else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, pcts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("% of episodes")
    ax.set_title("Quality Flag Prevalence (all conditions)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Tier 1"),
                       Patch(facecolor="#3498db", label="Tier 2")]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    for i, pct in enumerate(pcts):
        if pct > 0.3:
            ax.text(pct + 0.2, i, f"{pct:.1f}%", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_prevalence_overall.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'quality_prevalence_overall.png'}")


def plot_quality_prevalence_by_condition(rows: List[Dict]) -> None:
    ncols = 8
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 0.35 * len(ALL_FLAG_NAMES) + 1.5),
                             squeeze=False)
    names = list(reversed(ALL_FLAG_NAMES))

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx // ncols][idx % ncols]
        cond_rows = [r for r in rows if r["condition"] == cond]
        n = len(cond_rows)
        pcts = [sum(r[name] for r in cond_rows) / n * 100 if n else 0 for name in names]
        colors = ["#e74c3c" if name in TIER1_NAMES else "#3498db" for name in names]

        y = np.arange(len(names))
        ax.barh(y, pcts, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=5)
        ax.set_xlabel("% of episodes", fontsize=6)
        ax.set_title(f"{CONDITION_LABELS[cond]}  (N={n})", fontsize=7)

    fig.suptitle("Quality Flag Prevalence by Condition", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_prevalence_by_condition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'quality_prevalence_by_condition.png'}")


def plot_quality_by_model(rows: List[Dict], behavioral_data: List[Dict]) -> None:
    models = sorted(set(r["model_1"] for r in rows) | set(r["model_2"] for r in rows))
    model_counts: Dict[str, Dict[str, float]] = {m: {n: 0 for n in ALL_FLAG_NAMES} for m in models}
    model_totals: Dict[str, int] = {m: 0 for m in models}

    for r, beh_d in zip(rows, behavioral_data):
        m1, m2 = r["model_1"], r["model_2"]
        model_totals[m1] = model_totals.get(m1, 0) + 1
        model_totals[m2] = model_totals.get(m2, 0) + 1

        for name in ALL_FLAG_NAMES:
            if not r[name]:
                continue
            attr = attribute_to_speaker(name, beh_d)
            if attr == "guide":
                model_counts[m1][name] += 1
            elif attr == "seeker":
                model_counts[m2][name] += 1
            else:
                model_counts[m1][name] += 1
                model_counts[m2][name] += 1

    n_flags = len(ALL_FLAG_NAMES)
    n_models = len(models)
    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), 6))

    x = np.arange(n_models)
    width = 0.8 / n_flags
    cmap = plt.cm.tab20(np.linspace(0, 1, n_flags))

    for i, name in enumerate(ALL_FLAG_NAMES):
        rates = [model_counts[m][name] / model_totals[m] * 100
                 if model_totals[m] > 0 else 0
                 for m in models]
        ax.bar(x + i * width - 0.4 + width / 2, rates, width,
               label=name, color=cmap[i], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7, rotation=90, ha="center")
    ax.set_ylabel("% of episodes (attributed)")
    ax.set_title("Quality Flag Attribution by Model")
    ax.legend(fontsize=6, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_attribution_by_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'quality_attribution_by_model.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global RESULTS_ROOT, OUT_DIR

    parser = argparse.ArgumentParser(description="Conversation quality audit for false-belief v3 experiments.")
    parser.add_argument("--results_dir", type=str, default="logit_lens_results_false_belief_v3",
                        help="Results directory name (relative to REPO_ROOT).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory name (relative to REPO_ROOT). "
                             "Default: summary_plots_{results_dir_suffix}/quality_audit")
    args = parser.parse_args()

    RESULTS_ROOT = REPO_ROOT / args.results_dir
    if args.output_dir:
        OUT_DIR = REPO_ROOT / args.output_dir
    else:
        suffix = args.results_dir.replace("logit_lens_results_", "summary_plots_")
        OUT_DIR = REPO_ROOT / suffix / "quality_audit"

    if not RESULTS_ROOT.exists():
        print(f"Results directory not found: {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples_dir = OUT_DIR / "flagged_examples"
    for name in ALL_FLAG_NAMES:
        (examples_dir / name).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    behavioral_data: List[Dict] = []
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
            behavioral_data.append(d)

            for name in ALL_FLAG_NAMES:
                if flags.get(name) and example_counts[name] < 10:
                    txt = format_transcript(d, ep_dir.name, flags)
                    example_texts[name].append(txt)
                    example_counts[name] += 1

    print(f"Processed {len(rows)} episodes across {len(pair_dirs)} pairs")

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

    for name in ALL_FLAG_NAMES:
        if example_texts[name]:
            out = examples_dir / name / "examples.txt"
            with open(out, "w") as f:
                f.write("\n".join(example_texts[name]))
            print(f"  {name}: {len(example_texts[name])} examples")

    total = len(rows)
    if total == 0:
        print("No episodes found. Exiting.")
        return

    prevalence = {}
    for name in ALL_FLAG_NAMES:
        c = sum(r[name] for r in rows)
        prevalence[name] = (c, c / total * 100 if total else 0)

    pair_cond: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        pair_cond[(r["pair"], r["condition"])].append(r)

    model_role: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        model_role[(r["model_1"], "guide")].append(r)
        model_role[(r["model_2"], "seeker")].append(r)

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
    rows_with_choice = [r for r in rows if r.get("final_choice")]
    rows_clean = [r for r in rows_with_choice if not r["measurement_suspect"]]

    def _pair_cond_agg(subset):
        df = pd.DataFrame(subset)
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=int), 0
        pc = df.groupby(["pair", "condition"])["correct"].mean().reset_index()
        means = pc.groupby("condition")["correct"].mean().reindex(CONDITIONS).fillna(0)
        sems = pc.groupby("condition")["correct"].sem().reindex(CONDITIONS).fillna(0)
        ns = df.groupby("condition").size().reindex(CONDITIONS, fill_value=0)
        n_pairs = pc["pair"].nunique()
        return means, sems, ns, n_pairs

    orig_means, orig_sems, orig_ns, n_pairs_orig = _pair_cond_agg(rows_with_choice)
    clean_means, clean_sems, clean_ns, n_pairs_clean = _pair_cond_agg(rows_clean)
    all_df = pd.DataFrame(rows)
    total_ns = all_df.groupby("condition").size().reindex(CONDITIONS, fill_value=0) if not all_df.empty else pd.Series(0, index=CONDITIONS)

    belief_colors = {"mutual": "#66c2a5", "asym": "#fc8d62", "conflict": "#8da0cb"}
    bar_colors = [belief_colors.get(c.split("_")[0], "#999999") for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(22, 5))
    x = np.arange(len(CONDITIONS))
    width = 0.35
    ax.bar(x - width / 2, orig_means.values, width, yerr=orig_sems.values,
           capsize=3, color=[c + "AA" for c in bar_colors], edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 1.0},
           label=f"Original, excl. null-choice ({n_pairs_orig} pairs)")
    ax.bar(x + width / 2, clean_means.values, width, yerr=clean_sems.values,
           capsize=3, color=bar_colors, edgecolor="black", linewidth=0.4,
           error_kw={"linewidth": 1.0},
           label=f"Clean, excl. null-choice + Tier 1 ({n_pairs_clean} pairs)")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title("Behavioral Accuracy: Original vs. Clean (Tier 1 excluded)\n"
                 "(averaged per pair, then across pairs; ±SEM)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    for i in (8, 16):
        ax.axvline(i - 0.5, ls=":", color="gray", lw=0.5, alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    plot_path = OUT_DIR / "clean_vs_original_accuracy.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    plot_quality_prevalence_overall(rows, prevalence)
    plot_quality_prevalence_by_condition(rows)
    plot_quality_by_model(rows, behavioral_data)

    # ── Write quality_audit_report.txt ──
    report_path = OUT_DIR / "quality_audit_report.txt"
    with open(report_path, "w") as rpt:
        rpt.write("CONVERSATION QUALITY AUDIT REPORT (V3)\n")
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
