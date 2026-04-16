#!/usr/bin/env python3
"""
Aggregate and plot summary results from the false-belief logit-lens experiments.

Produces:
  1. Behavioral accuracy bar chart + CSV
  2. Contrastive log-prob diff bar charts (last layer & second-to-last) + CSV
  3. Stem true-vs-false token mention bar chart + CSV
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "logit_lens_results_false_belief"
OUT_DIR = REPO_ROOT / "summary_plots_false_belief"

PAIR_DIRS = sorted(RESULTS_ROOT.iterdir()) if RESULTS_ROOT.exists() else []

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

SCENARIO_TYPES = ["key", "wallet", "pkg"]

# For contrastive: true location is always loc_a (idx 0) for key, loc_b (idx 1) for wallet/pkg.
# The contrastive completions are [completion_A, completion_B].
# We compute: log_prob(true_loc) - log_prob(false_loc)
# so positive = model favors true location.
TRUE_LOC_INDEX = {"key": 0, "wallet": 1, "pkg": 1}

# Which contrastive classes compare locations (idx 0 = loc_a, idx 1 = loc_b)
LOCATION_CLASSES = {
    "self_belief_location", "self_think_location", "self_believe_location",
    "other_think_location", "ava_think_location", "mia_think_location",
    "tell_location", "want_check", "will_check",
}

# Non-location classes: first completion is "positive" framing
NON_LOCATION_CLASSES = {
    "self_honesty", "other_honesty", "other_strategy",
    "ava_strategy", "mia_strategy", "decision_confidence",
}

CONTRASTIVE_CLASS_LABELS = {
    "self_belief_location": "The [item] is at...",
    "self_think_location": "I think the [item] is at...",
    "self_believe_location": "I believe the [item] is at...",
    "other_think_location": "Other thinks [item] is at...",
    "ava_think_location": "Ava thinks [item] is at...",
    "mia_think_location": "Mia thinks [item] is at...",
    "tell_location": "I should tell them [item] is at...",
    "want_check": "I want them to check the...",
    "will_check": "I will check the...",
    "self_honesty": "I am being honest vs deceptive",
    "other_honesty": "Other is honest vs deceptive",
    "other_strategy": "Other is helping vs misleading",
    "ava_strategy": "Ava is helping vs misleading",
    "mia_strategy": "Mia is helping vs misleading",
    "decision_confidence": "Sure vs unsure about choice",
}

# Tokens to look for in stem last-layer predictions
TRUE_TOKENS = {"key": "red", "wallet": "drawer", "pkg": "back"}
FALSE_TOKENS = {"key": "blue", "wallet": "desk", "pkg": "front"}

SCENARIO_PREFIX_MAP = {
    "key_in_boxes": "key",
    "wallet_desk_drawer": "wallet",
    "package_door_porch": "pkg",
}


def _pair_short(pair_name: str) -> str:
    parts = pair_name.split("_None_0_")
    m1 = parts[0].split("-")[-1]
    m2 = parts[1].split("_false_belief")[0].split("-")[-1]
    return f"{m1}×{m2}"


def _condition_from_codename(codename: str) -> Tuple[str, str]:
    """Return (scenario_short, condition) e.g. ('key', 'shared_help')."""
    for prefix in ("key_", "wallet_", "pkg_"):
        if codename.startswith(prefix):
            scenario = prefix.rstrip("_")
            cond = codename[len(prefix):]
            return scenario, cond
    raise ValueError(f"Cannot parse codename: {codename}")


def load_all_episodes() -> List[Dict[str, Any]]:
    """Walk all pair × episode dirs and load available JSON summaries."""
    records = []
    for pair_dir in PAIR_DIRS:
        if not pair_dir.is_dir():
            continue
        pair_name = pair_dir.name
        for ep_dir in sorted(pair_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            beh_path = ep_dir / "behavioral_summary.json"
            con_path = ep_dir / "contrastive_summary.json"
            stem_path = ep_dir / "stem_summary.json"

            rec: Dict[str, Any] = {
                "pair": pair_name,
                "pair_short": _pair_short(pair_name),
                "ep_dir": str(ep_dir),
            }

            if beh_path.exists():
                with open(beh_path) as f:
                    rec["behavioral"] = json.load(f)
            if con_path.exists():
                with open(con_path) as f:
                    rec["contrastive"] = json.load(f)
            if stem_path.exists():
                with open(stem_path) as f:
                    rec["stem"] = json.load(f)

            meta = rec.get("behavioral", rec.get("contrastive", rec.get("stem", {})))
            sm = meta.get("scenario_meta", {})
            codename = sm.get("codename") or sm.get("scenario_id", "")
            if not codename:
                continue
            scenario, condition = _condition_from_codename(codename)
            rec["scenario"] = scenario
            rec["condition"] = condition
            rec["codename"] = codename
            records.append(rec)
    return records


# ─────────────────────────────────────────────────────────
# 1. Behavioral accuracy
# ─────────────────────────────────────────────────────────

def build_behavioral_df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        beh = r.get("behavioral")
        if beh is None:
            continue
        rows.append({
            "pair": r["pair_short"],
            "scenario": r["scenario"],
            "condition": r["condition"],
            "correct": int(beh["correct"]),
        })
    return pd.DataFrame(rows)


def plot_behavioral(df: pd.DataFrame) -> None:
    agg = df.groupby("condition")["correct"].mean().reindex(CONDITIONS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CONDITIONS))
    bars = ax.bar(x, agg.values, color=plt.cm.Set2(np.linspace(0, 1, len(CONDITIONS))),
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title("Behavioral Accuracy by Condition\n(averaged over scenarios & model pairs)")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="chance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavioral_accuracy.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'behavioral_accuracy.png'}")


# ─────────────────────────────────────────────────────────
# 2. Contrastive log-prob diffs  (per turn × agent, one subplot per class)
# ─────────────────────────────────────────────────────────

AGENT_ROLES = {1: "Mia (Guide)", 2: "Ava (Seeker)"}


def _get_contrastive_classes(contrastive: Dict) -> List[str]:
    for agent_key in ("agent_1", "agent_2"):
        agent_data = contrastive.get(agent_key, {})
        if agent_data:
            return list(agent_data.keys())
    return []


def build_contrastive_df(records: List[Dict], layer_offset: int) -> pd.DataFrame:
    """One row per (episode, class, turn, agent) with the log-prob diff."""
    rows = []
    for r in records:
        con = r.get("contrastive")
        if con is None:
            continue
        scenario = r["scenario"]
        true_idx = TRUE_LOC_INDEX[scenario]
        false_idx = 1 - true_idx
        for agent_num in (1, 2):
            agent_key = f"agent_{agent_num}"
            agent_data = con.get(agent_key, {})
            for cls_name, cls_data in agent_data.items():
                for turn_key in sorted(cls_data.keys(), key=int):
                    layers = cls_data[turn_key]
                    layer_scores = layers[layer_offset]
                    if cls_name in LOCATION_CLASSES:
                        diff = layer_scores[true_idx] - layer_scores[false_idx]
                    else:
                        diff = layer_scores[0] - layer_scores[1]
                    rows.append({
                        "pair": r["pair_short"],
                        "scenario": scenario,
                        "condition": r["condition"],
                        "class": cls_name,
                        "turn": int(turn_key),
                        "agent": agent_num,
                        "logprob_diff": float(diff),
                    })
    return pd.DataFrame(rows)


def plot_contrastive_for_turn_agent(
    df: pd.DataFrame, turn: int, agent: int,
    layer_label: str, out_dir: Path,
) -> None:
    sub = df[(df["turn"] == turn) & (df["agent"] == agent)]
    if sub.empty:
        return
    n_episodes = sub.groupby(["pair", "scenario", "condition"]).ngroups
    classes = [c for c in sub["class"].unique() if c in CONTRASTIVE_CLASS_LABELS]
    n_cls = len(classes)
    if n_cls == 0:
        return
    ncols = 3
    nrows = (n_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, cls_name in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        cls_sub = sub[sub["class"] == cls_name]
        agg = cls_sub.groupby("condition")["logprob_diff"].mean().reindex(CONDITIONS)
        x = np.arange(len(CONDITIONS))
        colors = ["#66c2a5" if v >= 0 else "#fc8d62" for v in agg.values]
        bars = ax.bar(x, agg.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=7)
        ax.axhline(0, ls="-", color="black", lw=0.6)
        label = CONTRASTIVE_CLASS_LABELS.get(cls_name, cls_name)
        if cls_name in LOCATION_CLASSES:
            ax.set_title(f"{label}\n(+) = true loc", fontsize=8)
        else:
            ax.set_title(f"{label}\n(+) = first option", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        for bar, val in zip(bars, agg.values):
            y_off = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            y_pos = val + (y_off if val >= 0 else -y_off)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=6)

    for idx in range(n_cls, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    role = AGENT_ROLES[agent]
    fig.suptitle(
        f"Contrastive Probes — {layer_label} — Turn {turn}, Agent {agent} ({role})\n"
        f"(n={n_episodes} episodes, averaged over scenarios & model pairs)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fname = f"contrastive_{layer_label.lower().replace(' ', '_').replace('-', '_')}_turn{turn}_agent{agent}.png"
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


# ─────────────────────────────────────────────────────────
# 3. Stem token mentions  (per turn × agent, one subplot per stem)
# ─────────────────────────────────────────────────────────

def _canonical_stem(stem: str) -> str:
    """Normalize agent-specific and scenario-specific stems for aggregation."""
    for name in ("Ava", "Mia"):
        stem = stem.replace(name, "[Agent]")
    longer_first = [
        ("them the key is in the", "them the [item] is [prep]"),
        ("them the wallet is in the", "them the [item] is [prep]"),
        ("them the wallet is on the", "them the [item] is [prep]"),
        ("them the package is at the", "them the [item] is [prep]"),
        ("The key is in the", "The [item] is [prep]"),
        ("The wallet is in the", "The [item] is [prep]"),
        ("The wallet is on the", "The [item] is [prep]"),
        ("The package is at the", "The [item] is [prep]"),
        ("the key is in the", "the [item] is [prep]"),
        ("the wallet is in the", "the [item] is [prep]"),
        ("the wallet is on the", "the [item] is [prep]"),
        ("the package is at the", "the [item] is [prep]"),
    ]
    for old, new in longer_first:
        stem = stem.replace(old, new)
    return stem


def build_stem_df(records: List[Dict]) -> pd.DataFrame:
    """One row per (episode, stem, turn, agent) with true/false counts."""
    rows = []
    for r in records:
        stem = r.get("stem")
        if stem is None:
            continue
        scenario = r["scenario"]
        true_tok = TRUE_TOKENS[scenario]
        false_tok = FALSE_TOKENS[scenario]
        for turn_data in stem.get("turns", []):
            turn_idx = turn_data["turn"]
            agent_num = turn_data["agent_index"]
            for stem_name, sinfo in turn_data.get("stems", {}).items():
                preds = sinfo.get("continuation_prediction_after_full_stem", [])
                if not preds:
                    continue
                last_pred = preds[-1]["predicted_token_clean"].lower().strip("·▁ ")
                tc = 1 if true_tok in last_pred else 0
                fc = 1 if (not tc and false_tok in last_pred) else 0
                rows.append({
                    "pair": r["pair_short"],
                    "scenario": scenario,
                    "condition": r["condition"],
                    "stem": stem_name,
                    "stem_canonical": _canonical_stem(stem_name),
                    "turn": turn_idx,
                    "agent": agent_num,
                    "true_count": tc,
                    "false_count": fc,
                })
    return pd.DataFrame(rows)


def plot_stem_for_turn_agent(
    df: pd.DataFrame, turn: int, agent: int, out_dir: Path,
) -> None:
    sub = df[(df["turn"] == turn) & (df["agent"] == agent)]
    if sub.empty:
        return
    n_episodes = sub.groupby(["pair", "scenario", "condition"]).ngroups
    canonical_stems = list(dict.fromkeys(sub["stem_canonical"]))
    n_stems = len(canonical_stems)
    if n_stems == 0:
        return
    ncols = 4
    nrows = (n_stems + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, cstem in enumerate(canonical_stems):
        ax = axes[idx // ncols][idx % ncols]
        csub = sub[sub["stem_canonical"] == cstem]
        grp = csub.groupby("condition")[["true_count", "false_count"]].sum().reindex(CONDITIONS).fillna(0)
        totals = grp["true_count"] + grp["false_count"]
        true_frac = (grp["true_count"] / totals).fillna(0)
        false_frac = (grp["false_count"] / totals).fillna(0)

        x = np.arange(len(CONDITIONS))
        w = 0.35
        ax.bar(x - w / 2, true_frac.values, w, label="True loc" if idx == 0 else "",
               color="#1b9e77", edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, false_frac.values, w, label="False loc" if idx == 0 else "",
               color="#d95f02", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=7)
        title = cstem if len(cstem) <= 40 else cstem[:37] + "..."
        ax.set_title(f'"{title}"', fontsize=7)

    for idx in range(n_stems, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#1b9e77", ec="black", lw=0.4),
        plt.Rectangle((0, 0), 1, 1, fc="#d95f02", ec="black", lw=0.4),
    ]
    fig.legend(handles, ["True location token", "False location token"],
               loc="upper right", fontsize=9, frameon=True)
    role = AGENT_ROLES[agent]
    fig.suptitle(
        f"Stem Last-Layer Predictions — Turn {turn}, Agent {agent} ({role})\n"
        f"(n={n_episodes} episodes, averaged over scenarios & model pairs)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fname = f"stem_turn{turn}_agent{agent}.png"
    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading all episode data...")
    records = load_all_episodes()
    print(f"  Found {len(records)} episodes across {len(PAIR_DIRS)} model pairs.\n")

    n_beh = sum(1 for r in records if "behavioral" in r)
    n_con = sum(1 for r in records if "contrastive" in r)
    n_stm = sum(1 for r in records if "stem" in r)
    print(f"  Behavioral: {n_beh}   Contrastive: {n_con}   Stem: {n_stm}\n")

    # ── 1. Behavioral ──
    print("=" * 60)
    print("1. BEHAVIORAL ACCURACY")
    print("=" * 60)
    beh_df = build_behavioral_df(records)
    if not beh_df.empty:
        beh_detail = beh_df.sort_values(["condition", "scenario", "pair"])
        print("\nIndividual results:")
        print(beh_detail.to_string(index=False))
        beh_detail.to_csv(OUT_DIR / "behavioral_individual.csv", index=False)
        print(f"\n  Saved {OUT_DIR / 'behavioral_individual.csv'}")
        plot_behavioral(beh_df)
    else:
        print("  No behavioral data found.")

    # ── 2. Contrastive ──
    print("\n" + "=" * 60)
    print("2. CONTRASTIVE LOG-PROB DIFFS (per turn × agent)")
    print("=" * 60)

    con_subdir = OUT_DIR / "contrastive"
    con_subdir.mkdir(exist_ok=True)

    for layer_offset, layer_label in [(-1, "Last Layer"), (-2, "Second-to-Last Layer")]:
        con_df = build_contrastive_df(records, layer_offset)
        if con_df.empty:
            print(f"  No contrastive data for {layer_label}.")
            continue
        safe_layer = layer_label.lower().replace(" ", "_").replace("-", "_")
        csv_name = f"contrastive_{safe_layer}.csv"
        con_df.sort_values(["turn", "agent", "class", "condition", "scenario", "pair"]) \
              .to_csv(OUT_DIR / csv_name, index=False)
        print(f"\n  {layer_label}: {len(con_df)} rows saved to {csv_name}")

        turn_agents = sorted(con_df[["turn", "agent"]].drop_duplicates().values.tolist())
        print(f"  Generating plots for {len(turn_agents)} turn×agent combos...")
        for turn, agent in turn_agents:
            plot_contrastive_for_turn_agent(con_df, turn, agent, layer_label, con_subdir)

    # ── 3. Stem ──
    print("\n" + "=" * 60)
    print("3. STEM TOKEN MENTIONS (per turn × agent)")
    print("=" * 60)

    stem_subdir = OUT_DIR / "stem"
    stem_subdir.mkdir(exist_ok=True)

    stem_df = build_stem_df(records)
    if not stem_df.empty:
        stem_df.sort_values(["turn", "agent", "stem_canonical", "condition", "scenario", "pair"]) \
               .to_csv(OUT_DIR / "stem_individual.csv", index=False)
        print(f"  {len(stem_df)} rows saved to stem_individual.csv")

        turn_agents = sorted(stem_df[["turn", "agent"]].drop_duplicates().values.tolist())
        print(f"  Generating plots for {len(turn_agents)} turn×agent combos...")
        for turn, agent in turn_agents:
            plot_stem_for_turn_agent(stem_df, turn, agent, stem_subdir)
    else:
        print("  No stem data found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
