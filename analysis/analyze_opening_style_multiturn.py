#!/usr/bin/env python3
"""
Analyze results from the multi-turn opening style experiment.

Produces:
  - all_results.csv             (one row per episode with full dialogue)
  - behavioral_summary.csv      (accuracy per pattern × belief)
  - behavioral_accuracy.png     (grouped bar: 13 patterns × 3 beliefs)
  - behavioral_accuracy_overall.png  (overall accuracy by belief)
  - turn_distribution.png       (histogram of turn counts)
  - turn_distribution_by_belief.png  (3 subplots)
  - turn_distribution_by_pattern.png (heatmap)
  - quality_flags.csv           (per-episode quality flags)
  - quality_prevalence.png      (flag prevalence bar chart)
"""

import argparse
import ast
import glob
import os
import re
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

SLUG_TO_PATTERN = {}
for p in PATTERN_ORDER:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_")
    SLUG_TO_PATTERN[slug] = p


def _location_in_text(loc, text):
    """Check if a location name appears in text, allowing partial matches.

    Handles cases where the model uses a shortened form:
      "storage box" → "box", "hook by the door" → "hook",
      "workbench" → "bench"
    """
    if not loc:
        return False
    loc_lower = loc.lower().replace("_", " ")
    if loc_lower in text:
        return True
    words = loc_lower.split()
    for w in words:
        if len(w) >= 3 and re.search(r'\b' + re.escape(w) + r'\b', text):
            return True
    if len(words) == 1 and len(loc_lower) >= 6:
        for w in [loc_lower[:len(loc_lower)//2], loc_lower[len(loc_lower)//2:]]:
            if len(w) >= 3 and re.search(r'\b' + re.escape(w) + r'\b', text):
                return True
    return False


def _location_last_pos(loc, text, end):
    """Find the last position of a location mention before *end*."""
    loc_lower = loc.lower().replace("_", " ")
    pos = text.rfind(loc_lower, 0, end)
    if pos >= 0:
        return pos
    words = loc_lower.split()
    for w in words:
        if len(w) >= 4:
            for m in re.finditer(r'\b' + re.escape(w) + r'\b', text):
                if m.start() < end:
                    pos = max(pos, m.start())
    if pos >= 0:
        return pos
    if len(words) == 1 and len(loc_lower) >= 6:
        for w in [loc_lower[:len(loc_lower)//2], loc_lower[len(loc_lower)//2:]]:
            if len(w) >= 3:
                for m in re.finditer(r'\b' + re.escape(w) + r'\b', text):
                    if m.start() < end:
                        pos = max(pos, m.start())
    return pos


def extract_final_choice(dialog_turns, true_location, other_location):
    ava_turns = [t for t in dialog_turns if "Ava Thompson said:" in t]
    if not ava_turns:
        return None, False
    last_ava = ava_turns[-1]
    text_part = last_ava.split("said:")[-1] if "said:" in last_ava else last_ava
    text = text_part.lower()
    has_leave = "leave" in text

    if not has_leave:
        return None, False

    has_1 = _location_in_text(true_location, text)
    has_2 = _location_in_text(other_location, text)

    if has_1 and not has_2:
        return true_location, True
    if has_2 and not has_1:
        return other_location, True
    if has_1 and has_2:
        leave_pos = text.rfind("leave")
        last_1 = _location_last_pos(true_location, text, leave_pos)
        last_2 = _location_last_pos(other_location, text, leave_pos)
        if last_1 > last_2:
            return true_location, True
        elif last_2 > last_1:
            return other_location, True
    return None, True


def detect_repetitive_loop(dialog_turns, threshold=3):
    if len(dialog_turns) < threshold:
        return False
    texts = [t.split("said:")[-1].strip().lower() if "said:" in t else t.lower()
             for t in dialog_turns]
    from collections import Counter
    return Counter(texts).most_common(1)[0][1] >= threshold


def load_all_episodes(results_dir, scenario_lookup):
    results_dir = Path(results_dir)
    model_dir = None
    for d in results_dir.glob("dialogs/*"):
        if d.is_dir():
            model_dir = d
            break
    if model_dir is None:
        print(f"No model directory found in {results_dir}/dialogs/", file=sys.stderr)
        sys.exit(1)

    rows = []
    for pat_dir in sorted(model_dir.iterdir()):
        if not pat_dir.is_dir() or pat_dir.name == "prompt_records":
            continue
        pattern = SLUG_TO_PATTERN.get(pat_dir.name, pat_dir.name)

        for csv_file in sorted(pat_dir.glob("*_temp*.csv")):
            fname = csv_file.stem
            m = re.match(r"(\d+)_(shared_truth|ignorance|false_belief)_temp", fname)
            if not m:
                continue
            episode = int(m.group(1))
            belief = m.group(2)

            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            if df.empty:
                continue

            dialog_raw = str(df.iloc[0]["Dialog"])
            intro = str(df.iloc[0].get("Intro", ""))
            try:
                dialog_turns = ast.literal_eval(dialog_raw)
            except (ValueError, SyntaxError):
                dialog_turns = [dialog_raw]

            n_turns = len(dialog_turns)

            mia_opening = ""
            if dialog_turns:
                first = dialog_turns[0]
                if "said:" in first:
                    mia_opening = first.split("said:")[-1].strip()

            sc = scenario_lookup.get(episode, {})
            true_loc = sc.get("true_location", "")
            other_loc = sc.get("other_location", "")
            item = sc.get("item", "")
            scenario_text = sc.get("scenario_text", "")

            final_choice, has_leave = extract_final_choice(
                dialog_turns, true_loc, other_loc
            )
            correct = (
                int(final_choice.lower().replace(" ", "_") == true_loc
                    or final_choice.lower().replace("_", " ") == true_loc.lower().replace("_", " "))
                if final_choice else ""
            )

            dialog_str = "\n".join(
                f"[{t.split(' said:')[0].replace('Turn ', '[Turn ').split(': ')[0]}] "
                f"[{t.split(': ')[1].split(' said:')[0] if ': ' in t else ''}]: "
                f"{t.split('said:')[-1].strip()}"
                if "said:" in t else t
                for t in dialog_turns
            ) if dialog_turns else ""

            no_leave = not has_leave
            null_choice = final_choice is None
            max_turns_hit = n_turns >= 15
            repetitive = detect_repetitive_loop(dialog_turns)

            rows.append({
                "pattern": pattern,
                "episode": episode,
                "belief": belief,
                "item": item,
                "true_location": true_loc,
                "other_location": other_loc,
                "scenario": scenario_text,
                "mia_opening": mia_opening,
                "n_turns": n_turns,
                "dialogue": dialog_str,
                "dialogue_raw": dialog_raw,
                "final_choice": final_choice or "",
                "correct": correct,
                "has_leave": has_leave,
                "no_leave": no_leave,
                "null_choice": null_choice,
                "max_turns_hit": max_turns_hit,
                "repetitive_loop": repetitive,
            })

    return pd.DataFrame(rows)


def load_scenario_lookup(csv_path):
    df = pd.read_csv(csv_path)
    lookup = {}
    for ep_id in df["episode"].unique():
        ep_rows = df[df["episode"] == ep_id]
        row = ep_rows.iloc[0]
        lookup[int(ep_id)] = {
            "item": row["item"],
            "true_location": row["true_location"],
            "scenario_text": row["scenario"],
        }
        locs = re.search(
            r"(?:with|has) (?:a |an |the )(.+?) and (?:a |an |the )(.+?)\.",
            str(row["scenario"]),
        )
        if locs:
            loc1, loc2 = locs.group(1), locs.group(2)
            true_loc = row["true_location"]
            if true_loc in loc1 or loc1 in true_loc:
                lookup[int(ep_id)]["other_location"] = loc2
            else:
                lookup[int(ep_id)]["other_location"] = loc1
        else:
            lookup[int(ep_id)]["other_location"] = ""
    return lookup


def plot_behavioral_accuracy(df, out_dir):
    df_valid = df[df["final_choice"] != ""].copy()
    df_valid["correct"] = df_valid["correct"].astype(int)

    patterns_present = [p for p in PATTERN_ORDER if p in df["pattern"].unique()]
    n_patterns = len(patterns_present)

    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(n_patterns)
    width = 0.25

    for i, belief in enumerate(BELIEF_ORDER):
        accs, ns, totals = [], [], []
        for pat in patterns_present:
            sub_all = df[(df["pattern"] == pat) & (df["belief"] == belief)]
            sub = df_valid[(df_valid["pattern"] == pat) & (df_valid["belief"] == belief)]
            acc = sub["correct"].mean() if len(sub) > 0 else 0
            accs.append(acc)
            ns.append(len(sub))
            totals.append(len(sub_all))

        offset = (i - 1) * width
        ax.bar(x + offset, accs, width,
               label=BELIEF_LABELS[belief],
               color=BELIEF_COLORS[belief], alpha=0.85)

        for j, (acc, n, total) in enumerate(zip(accs, ns, totals)):
            ax.text(x[j] + offset, acc + 0.02, f"{n}/{total}",
                    ha="center", va="bottom", fontsize=5.5)

    labels = [PATTERN_SHORT.get(p, p) for p in patterns_present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy (choosing correct location)")
    n_valid = len(df_valid)
    n_total = len(df)
    ax.set_title(
        f"Ava's Accuracy by Mia's Opening Style × Belief State (Multi-Turn)\n"
        f"(Mia always says correct location | valid: {n_valid}/{n_total})",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'behavioral_accuracy.png'}")


def plot_behavioral_overall(df, out_dir):
    df_valid = df[df["final_choice"] != ""].copy()
    df_valid["correct"] = df_valid["correct"].astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    accs, labels, colors, ns = [], [], [], []
    for belief in BELIEF_ORDER:
        sub = df_valid[df_valid["belief"] == belief]
        sub_all = df[df["belief"] == belief]
        acc = sub["correct"].mean() if len(sub) > 0 else 0
        accs.append(acc)
        labels.append(BELIEF_LABELS[belief])
        colors.append(BELIEF_COLORS[belief])
        ns.append(f"{len(sub)}/{len(sub_all)}")

    bars = ax.bar(labels, accs, color=colors, alpha=0.85)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                n, ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy by Belief State (Multi-Turn)", fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "behavioral_accuracy_overall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'behavioral_accuracy_overall.png'}")


def plot_turn_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["n_turns"], bins=range(1, df["n_turns"].max() + 2),
            edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel("Count")
    ax.set_title("Turn Distribution (All Episodes)", fontweight="bold")
    mean_turns = df["n_turns"].mean()
    ax.axvline(mean_turns, color="red", linestyle="--", alpha=0.7,
               label=f"Mean: {mean_turns:.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "turn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'turn_distribution.png'}")


def plot_turn_distribution_by_belief(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    max_turn = df["n_turns"].max()
    for ax, belief in zip(axes, BELIEF_ORDER):
        sub = df[df["belief"] == belief]
        ax.hist(sub["n_turns"], bins=range(1, max_turn + 2),
                edgecolor="black", alpha=0.7, color=BELIEF_COLORS[belief])
        ax.set_xlabel("Number of Turns")
        ax.set_title(BELIEF_LABELS[belief])
        mean_t = sub["n_turns"].mean()
        ax.axvline(mean_t, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_t:.1f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Turn Distribution by Belief State", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "turn_distribution_by_belief.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'turn_distribution_by_belief.png'}")


def plot_turn_distribution_by_pattern(df, out_dir):
    patterns_present = [p for p in PATTERN_ORDER if p in df["pattern"].unique()]
    max_turn = min(df["n_turns"].max(), 16)

    grid = np.zeros((len(patterns_present), len(BELIEF_ORDER)))
    for i, pat in enumerate(patterns_present):
        for j, belief in enumerate(BELIEF_ORDER):
            sub = df[(df["pattern"] == pat) & (df["belief"] == belief)]
            grid[i, j] = sub["n_turns"].mean() if len(sub) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=1, vmax=max_turn)
    ax.set_xticks(range(len(BELIEF_ORDER)))
    ax.set_xticklabels([BELIEF_LABELS[b] for b in BELIEF_ORDER], fontsize=9)
    ax.set_yticks(range(len(patterns_present)))
    ax.set_yticklabels([PATTERN_SHORT.get(p, p) for p in patterns_present], fontsize=8)
    ax.set_title("Mean Turns by Pattern × Belief", fontweight="bold")

    for i in range(len(patterns_present)):
        for j in range(len(BELIEF_ORDER)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Mean Turns", shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "turn_distribution_by_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'turn_distribution_by_pattern.png'}")


def plot_quality_prevalence(df, out_dir):
    flags = ["no_leave", "null_choice", "max_turns_hit", "repetitive_loop"]
    flag_labels = {
        "no_leave": "No LEAVE",
        "null_choice": "Null Choice",
        "max_turns_hit": "Max Turns Hit",
        "repetitive_loop": "Repetitive Loop",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, belief in zip(axes, BELIEF_ORDER):
        sub = df[df["belief"] == belief]
        rates = [sub[f].mean() * 100 for f in flags]
        y_pos = np.arange(len(flags))
        ax.barh(y_pos, rates, color=BELIEF_COLORS[belief], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([flag_labels[f] for f in flags], fontsize=9)
        ax.set_xlabel("Prevalence (%)")
        ax.set_title(BELIEF_LABELS[belief])
        for i, rate in enumerate(rates):
            ax.text(rate + 0.5, i, f"{rate:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, max(rates) * 1.3 + 5 if rates else 10)

    fig.suptitle("Quality Flag Prevalence by Belief State", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_prevalence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'quality_prevalence.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-turn opening style experiment results.")
    parser.add_argument("--results_dir", type=str,
                        default="opening_style_multiturn_results")
    parser.add_argument("--scenarios_csv", type=str, required=True,
                        help="Path to all_results.csv for scenario metadata")
    parser.add_argument("--output_dir", type=str,
                        default="summary_plots_opening_style_multiturn")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    scenarios_csv = Path(args.scenarios_csv)
    if not scenarios_csv.is_absolute():
        scenarios_csv = REPO_ROOT / scenarios_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scenario metadata from {scenarios_csv}")
    scenario_lookup = load_scenario_lookup(scenarios_csv)
    print(f"Loaded {len(scenario_lookup)} scenarios")

    print(f"Loading episodes from {results_dir}")
    df = load_all_episodes(results_dir, scenario_lookup)
    print(f"Loaded {len(df)} episodes")

    # --- Print summary ---
    print(f"\n{'Pattern':<40s}  {'Belief':<15s}  {'N':>5s}  {'Valid':>5s}  {'Acc':>6s}  {'Turns':>5s}")
    print("-" * 85)
    summary_rows = []
    for pat in PATTERN_ORDER:
        for belief in BELIEF_ORDER:
            sub = df[(df["pattern"] == pat) & (df["belief"] == belief)]
            if len(sub) == 0:
                continue
            valid = sub[sub["final_choice"] != ""]
            n_correct = valid["correct"].astype(int).sum() if len(valid) > 0 else 0
            acc = n_correct / len(valid) if len(valid) > 0 else float("nan")
            acc_str = f"{acc:.3f}" if not pd.isna(acc) else "N/A"
            mean_turns = sub["n_turns"].mean()
            print(f"{pat:<40s}  {belief:<15s}  {len(sub):5d}  "
                  f"{len(valid):5d}  {acc_str:>6s}  {mean_turns:5.1f}")
            summary_rows.append({
                "pattern": pat,
                "belief": belief,
                "n_total": len(sub),
                "n_valid": len(valid),
                "n_correct": n_correct,
                "accuracy": acc,
                "mean_turns": mean_turns,
            })

    # --- Overall by belief ---
    print(f"\n{'Overall':<40s}")
    print("-" * 85)
    for belief in BELIEF_ORDER:
        sub = df[df["belief"] == belief]
        valid = sub[sub["final_choice"] != ""]
        n_correct = valid["correct"].astype(int).sum() if len(valid) > 0 else 0
        acc = n_correct / len(valid) if len(valid) > 0 else float("nan")
        print(f"{'ALL PATTERNS':<40s}  {belief:<15s}  {len(sub):5d}  "
              f"{len(valid):5d}  {acc:.3f}  {sub['n_turns'].mean():5.1f}")

    # --- Save CSVs ---
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "behavioral_summary.csv", index=False)
    print(f"\nSaved {out_dir / 'behavioral_summary.csv'}")

    all_results = df[[
        "pattern", "episode", "belief", "item", "true_location", "other_location",
        "scenario", "mia_opening", "n_turns", "dialogue", "final_choice", "correct",
    ]].copy()
    all_results.to_csv(out_dir / "all_results.csv", index=False)
    print(f"Saved {out_dir / 'all_results.csv'}")

    quality_df = df[[
        "pattern", "episode", "belief", "no_leave", "null_choice",
        "max_turns_hit", "repetitive_loop",
    ]].copy()
    quality_df.to_csv(out_dir / "quality_flags.csv", index=False)
    print(f"Saved {out_dir / 'quality_flags.csv'}")

    # --- Plots ---
    plot_behavioral_accuracy(df, out_dir)
    plot_behavioral_overall(df, out_dir)
    plot_turn_distribution(df, out_dir)
    plot_turn_distribution_by_belief(df, out_dir)
    plot_turn_distribution_by_pattern(df, out_dir)
    plot_quality_prevalence(df, out_dir)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
