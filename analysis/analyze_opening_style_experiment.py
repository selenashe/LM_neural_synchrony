#!/usr/bin/env python3
"""
Analyze results from the controlled opening-style experiment.

Produces:
  - opening_style_experiment_accuracy.png  (grouped bar: 13 patterns × 3 beliefs)
  - opening_style_experiment_summary.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

BELIEF_ORDER = ["shared_truth", "ignorance", "false_belief"]
BELIEF_LABELS = {
    "shared_truth": "Shared Truth",
    "ignorance": "Ignorance",
    "false_belief": "False Belief",
}
BELIEF_COLORS = {
    "shared_truth": "#4C72B0",
    "ignorance": "#55A868",
    "false_belief": "#C44E52",
}

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
]

PATTERN_SHORT = {
    "The X is [prep] Y": "The X is Y",
    "The X you're looking for is Y": "X you're\nlooking for",
    "If you're looking for X": "If you're\nlooking",
    "Are you looking for X?": "Are you\nlooking?",
    "Looking for X?": "Looking\nfor X?",
    "I saw X on/in Y": "I saw X",
    "Vocative (Ava/Hey Ava, ...)": "Ava, X is Y",
    "Your X is on/in Y": "Your X",
    "I think X is in Y": "I think X",
    "If you need X": "If you\nneed X",
    "Bare anaphoric (It's in Y)": "It's in Y",
    "Imperative (Check...)": "Check...",
    "You're looking for X, right?": "You're looking\nright?",
}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze opening style experiment results.")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Results CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    n_total = len(df)
    df_valid = df[df["final_choice"].notna() & (df["final_choice"] != "")].copy()
    df_valid["correct"] = df_valid["correct"].astype(int)
    n_valid = len(df_valid)
    print(f"Valid (non-null choice): {n_valid}/{n_total}")

    # --- Summary table ---
    rows = []
    for pat in PATTERN_ORDER:
        for belief in BELIEF_ORDER:
            sub = df_valid[(df_valid["pattern"] == pat) & (df_valid["belief"] == belief)]
            sub_all = df[(df["pattern"] == pat) & (df["belief"] == belief)]
            if len(sub_all) == 0:
                continue
            rows.append({
                "pattern": pat,
                "belief": belief,
                "n_total": len(sub_all),
                "n_valid": len(sub),
                "accuracy": sub["correct"].mean() if len(sub) > 0 else float("nan"),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "opening_style_experiment_summary.csv", index=False)
    print(f"Saved {out_dir / 'opening_style_experiment_summary.csv'}")

    # --- Print table ---
    print(f"\n{'Pattern':<35s}  {'Belief':<15s}  {'N':>5s}  {'Valid':>5s}  {'Acc':>6s}")
    print("-" * 75)
    for _, r in summary.iterrows():
        acc_str = f"{r['accuracy']:.3f}" if not pd.isna(r["accuracy"]) else "N/A"
        print(f"{r['pattern']:<35s}  {r['belief']:<15s}  "
              f"{int(r['n_total']):5d}  {int(r['n_valid']):5d}  {acc_str:>6s}")

    # --- Plot: grouped bar chart ---
    patterns_present = [p for p in PATTERN_ORDER if p in df["pattern"].unique()]
    n_patterns = len(patterns_present)
    n_beliefs = len(BELIEF_ORDER)

    fig, ax = plt.subplots(figsize=(18, 7))

    x = np.arange(n_patterns)
    width = 0.25

    for i, belief in enumerate(BELIEF_ORDER):
        accs = []
        ns = []
        for pat in patterns_present:
            sub = df_valid[(df_valid["pattern"] == pat) & (df_valid["belief"] == belief)]
            acc = sub["correct"].mean() if len(sub) > 0 else 0
            accs.append(acc)
            ns.append(len(sub))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs, width,
                      label=BELIEF_LABELS[belief],
                      color=BELIEF_COLORS[belief], alpha=0.85)

        for j, (acc, n) in enumerate(zip(accs, ns)):
            ax.text(x[j] + offset, acc + 0.02, f"{n}",
                    ha="center", va="bottom", fontsize=6)

    labels = [PATTERN_SHORT.get(p, p) for p in patterns_present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy (choosing correct location)")
    ax.set_title(
        f"Ava's Accuracy by Mia's Opening Style × Belief State\n"
        f"(Mia always says correct location | valid: {n_valid}/{n_total})",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "opening_style_experiment_accuracy.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'opening_style_experiment_accuracy.png'}")


if __name__ == "__main__":
    main()
