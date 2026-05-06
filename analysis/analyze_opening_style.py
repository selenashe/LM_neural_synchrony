#!/usr/bin/env python3
"""
Analyze Mia's opening instruction style and its correlation with Ava's accuracy.

Filters to 2-turn episodes only, classifies Mia's first turn into linguistic
patterns, and plots proportion + accuracy per pattern for each condition.

Produces:
  - opening_style_analysis.png  (2×3 subplot grid)
  - opening_style_analysis.csv
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = [
    "shared_help", "shared_deceive",
    "ignorance_help", "ignorance_deceive",
    "false_help", "false_deceive",
]

CONDITION_LABELS = {
    "shared_help": "Shared Help",
    "shared_deceive": "Shared Deceive",
    "ignorance_help": "Ignorance Help",
    "ignorance_deceive": "Ignorance Deceive",
    "false_help": "False Belief Help",
    "false_deceive": "False Belief Deceive",
}


def get_mia_first_turn(dialogue: str) -> str:
    if pd.isna(dialogue):
        return ""
    for line in str(dialogue).split("\n"):
        line = line.strip()
        if line.startswith("[Mia Sanders]:"):
            return line.replace("[Mia Sanders]:", "").strip()
    return ""


def count_turns(dialogue: str) -> int:
    if pd.isna(dialogue):
        return 0
    return sum(1 for line in str(dialogue).split("\n")
               if line.strip().startswith("["))


def classify_opening(text: str) -> str:
    t = text.strip()
    tl = t.lower()

    if re.match(r"^Are you looking for", t):
        return "Are you looking for X?"
    if re.match(r"^If you're looking for", t):
        return "If you're looking for X"
    if re.match(r"^Looking for", t):
        return "Looking for X?"
    if re.match(r"^You're looking for", t):
        return "You're looking for X, right?"
    if "you're looking for" in tl or "you are looking for" in tl:
        return "The X you're looking for is Y"
    if "looking" in tl:
        return "other-looking"

    if re.match(r"^(Hey )?Ava,? ", t):
        return "Vocative (Ava/Hey Ava, ...)"
    if re.match(r"^I saw ", t):
        return "I saw X on/in Y"
    if re.match(r"^I think ", t):
        return "I think X is in Y"
    if re.match(r"^If you need ", t):
        return "If you need X"
    if re.match(r"^Your ", t):
        return "Your X is on/in Y"
    if re.match(r"^Check ", t):
        return "Imperative (Check...)"
    if re.match(r"^It's ", t):
        return "Bare anaphoric (It's in Y)"
    if re.match(r"^The ", t):
        return "The X is [prep] Y"

    return "other"


PATTERN_ORDER = [
    "The X is [prep] Y",
    "The X you're looking for is Y",
    "If you're looking for X",
    "Are you looking for X?",
    "Looking for X?",
    "I saw X on/in Y",
    "Vocative (Ava/Hey Ava, ...)",
    "Your X is on/in Y",
    "I think X is in Y",
    "If you need X",
    "Bare anaphoric (It's in Y)",
    "Imperative (Check...)",
    "You're looking for X, right?",
    "other-looking",
    "other",
]

PATTERN_SHORT = {
    "The X is [prep] Y": "The X is Y",
    "The X you're looking for is Y": "X you're looking\nfor is Y",
    "If you're looking for X": "If you're\nlooking for X",
    "Are you looking for X?": "Are you\nlooking for X?",
    "Looking for X?": "Looking\nfor X?",
    "I saw X on/in Y": "I saw X\non/in Y",
    "Vocative (Ava/Hey Ava, ...)": "Vocative\n(Ava,...)",
    "Your X is on/in Y": "Your X\nis on/in Y",
    "I think X is in Y": "I think\nX is in Y",
    "If you need X": "If you\nneed X",
    "Bare anaphoric (It's in Y)": "It's in Y",
    "Imperative (Check...)": "Check...",
    "You're looking for X, right?": "You're looking\nfor X, right?",
    "other-looking": "other-looking",
    "other": "other",
}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Mia's opening style vs Ava's accuracy.")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to all_results.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (relative to REPO_ROOT or absolute)")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")

    df["n_turns"] = df["dialogue"].apply(count_turns)
    df["mia_opening"] = df["dialogue"].apply(get_mia_first_turn)
    df["pattern"] = df["mia_opening"].apply(classify_opening)

    n_all = len(df)
    df2 = df[df["n_turns"] == 2].copy()
    n_2turn = len(df2)
    print(f"2-turn episodes: {n_2turn} / {n_all}")

    # --- Build summary table ---
    csv_rows = []
    for cond in CONDITIONS:
        sub = df2[df2["condition"] == cond]
        n_cond = len(sub)
        if n_cond == 0:
            continue
        for pat, grp in sub.groupby("pattern"):
            csv_rows.append({
                "condition": cond,
                "pattern": pat,
                "count": len(grp),
                "proportion": len(grp) / n_cond,
                "accuracy": grp["correct"].mean(),
            })

    summary = pd.DataFrame(csv_rows)
    summary.to_csv(out_dir / "opening_style_analysis.csv", index=False)
    print(f"Saved {out_dir / 'opening_style_analysis.csv'}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        sub = summary[summary["condition"] == cond].copy()
        n_cond_2turn = int(sub["count"].sum()) if len(sub) > 0 else 0
        n_cond_all = len(df[df["condition"] == cond])

        if sub.empty:
            ax.set_title(f"{CONDITION_LABELS[cond]}\n(2-turn: 0 / {n_cond_all})")
            ax.set_visible(False)
            continue

        order = [p for p in PATTERN_ORDER if p in sub["pattern"].values]
        sub = sub.set_index("pattern").loc[order].reset_index()

        x = np.arange(len(sub))
        w = 0.35

        bars_prop = ax.bar(x - w / 2, sub["proportion"], w,
                           label="Proportion", color="#4C72B0", alpha=0.85)
        bars_acc = ax.bar(x + w / 2, sub["accuracy"], w,
                          label="Accuracy", color="#DD8452", alpha=0.85)

        for i, (_, row) in enumerate(sub.iterrows()):
            n = int(row["count"])
            ax.text(x[i] - w / 2, row["proportion"] + 0.02, str(n),
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.text(x[i] + w / 2, row["accuracy"] + 0.02,
                    f"{row['accuracy']:.2f}", ha="center", va="bottom",
                    fontsize=7)

        labels = [PATTERN_SHORT.get(p, p) for p in sub["pattern"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Value")
        ax.set_title(
            f"{CONDITION_LABELS[cond]}\n(2-turn: {n_cond_2turn} / {n_cond_all})",
            fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Mia Opening Style vs Ava Accuracy\n"
        f"(2-turn episodes: {n_2turn} / {n_all})",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "opening_style_analysis.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'opening_style_analysis.png'}")


if __name__ == "__main__":
    main()
