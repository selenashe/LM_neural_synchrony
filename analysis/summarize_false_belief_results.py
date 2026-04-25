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
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOTOPIA_DATA_DIR = REPO_ROOT / "sotopia_utils" / "sotopia_data"
RESULTS_ROOT = REPO_ROOT / "logit_lens_results_false_belief_100"
OUT_DIR = REPO_ROOT / "summary_plots_false_belief_100"

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

# ── Load scenario definitions from JSON (data-driven) ──

with open(SOTOPIA_DATA_DIR / "scenario_vocab_false_belief.json", encoding="utf-8") as _f:
    _SCENARIO_VOCAB = json.load(_f)

with open(SOTOPIA_DATA_DIR / "envs_false_belief.json", encoding="utf-8") as _f:
    _ENVS = json.load(_f)

# Build codename -> (scenario_prefix, condition) lookup
_CODENAME_MAP: Dict[str, Tuple[str, str]] = {}
# Build scenario_prefix -> scenario_type lookup
SCENARIO_PREFIX_MAP: Dict[str, str] = {}
for _env in _ENVS.values():
    _meta = _env["_meta"]
    _st = _meta["scenario_type"]
    _cn = _env["codename"]
    _belief = _meta["belief_condition"]
    _goal = _meta["goal_condition"]
    _belief_short = {"shared_truth": "shared", "ignorance": "ignorance", "false_belief": "false"}[_belief]
    _cond = f"{_belief_short}_{_goal}"
    # Extract codename_prefix: everything before _{belief_short}_{goal}
    _suffix = f"_{_belief_short}_{_goal}"
    _prefix = _cn[: -len(_suffix)] if _cn.endswith(_suffix) else _cn
    _CODENAME_MAP[_cn] = (_prefix, _cond)
    SCENARIO_PREFIX_MAP[_st] = _prefix

SCENARIO_TYPES = sorted(set(SCENARIO_PREFIX_MAP.values()))

# Build TRUE/FALSE token maps and TRUE_LOC_INDEX from envs + vocab
TRUE_LOC_INDEX: Dict[str, int] = {}
TRUE_TOKENS: Dict[str, str] = {}
FALSE_TOKENS: Dict[str, str] = {}
for _st, _prefix in SCENARIO_PREFIX_MAP.items():
    _vocab = _SCENARIO_VOCAB[_st]
    _locs = _vocab["locations"]
    # Find true_location from the first env of this type
    for _env in _ENVS.values():
        if _env["_meta"]["scenario_type"] == _st:
            _true_loc_id = _env["_meta"]["true_location"]
            _true_loc_display = _true_loc_id.replace("_", " ")
            _true_idx = _locs.index(_true_loc_display)
            _false_idx = 1 - _true_idx
            TRUE_LOC_INDEX[_prefix] = _true_idx
            TRUE_TOKENS[_prefix] = _locs[_true_idx].split()[0].lower()
            FALSE_TOKENS[_prefix] = _locs[_false_idx].split()[0].lower()
            break

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


def _pair_full_label(pair_dir_name: str) -> str:
    """Full HF-style model ids from results folder name (no _None_0_ / false_belief suffix)."""
    parts = pair_dir_name.split("_None_0_")
    if len(parts) < 2:
        return pair_dir_name
    m1, rest = parts[0], parts[1]
    m2 = rest.split("_false_belief")[0]
    return f"{m1} × {m2}"


def _condition_from_codename(codename: str) -> Tuple[str, str]:
    """Return (scenario_short, condition) e.g. ('key', 'shared_help')."""
    if codename in _CODENAME_MAP:
        return _CODENAME_MAP[codename]
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
            con_path = ep_dir / "contrastive_summary_compact.json"
            stem_path = ep_dir / "stem_summary_compact.json"

            rec: Dict[str, Any] = {
                "pair": _pair_full_label(pair_name),
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
            "pair": r["pair"],
            "scenario": r["scenario"],
            "condition": r["condition"],
            "correct": int(beh["correct"]),
        })
    return pd.DataFrame(rows)


def plot_behavioral(df: pd.DataFrame) -> None:
    pair_cond = df.groupby(["pair", "condition"])["correct"].mean().reset_index()
    means = pair_cond.groupby("condition")["correct"].mean().reindex(CONDITIONS)
    sems = pair_cond.groupby("condition")["correct"].sem().reindex(CONDITIONS).fillna(0)
    n_pairs = pair_cond["pair"].nunique()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CONDITIONS))
    colors = plt.cm.Set2(np.linspace(0, 1, len(CONDITIONS)))
    bars = ax.bar(x, means.values, yerr=sems.values, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_title(f"Behavioral Accuracy by Condition\n(averaged over scenarios & {n_pairs} model pairs)")
    ax.set_ylim(0, 1.05)
    for bar, val, sem in zip(bars, means.values, sems.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="chance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavioral_accuracy.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'behavioral_accuracy.png'}")


def _pair_title_for_subplot(pair: str) -> str:
    if " × " in pair:
        left, right = pair.split(" × ", 1)
        return f"{left}\n× {right}"
    return pair


def plot_behavioral_per_pair(df: pd.DataFrame) -> None:
    pairs = sorted(df["pair"].unique())
    n_pairs = len(pairs)
    ncols = 6
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.2 * nrows),
                             squeeze=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(CONDITIONS)))
    x = np.arange(len(CONDITIONS))

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["pair"] == pair]
        agg = sub.groupby("condition")["correct"].mean().reindex(CONDITIONS).fillna(0)
        ax.bar(x, agg.values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=5)
        ax.set_ylim(0, 1.05)
        ax.set_title(_pair_title_for_subplot(pair), fontsize=5)
        ax.tick_params(axis="y", labelsize=6)
        ax.axhline(0.5, ls="--", color="gray", lw=0.5)

    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Behavioral Accuracy by Condition — per Model Pair", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "behavioral_accuracy_per_pair.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'behavioral_accuracy_per_pair.png'}")


# ─────────────────────────────────────────────────────────
# 2. Contrastive log-prob diffs  (per turn × agent, one subplot per class)
# ─────────────────────────────────────────────────────────

AGENT_ROLES = {1: "Mia (Guide)", 2: "Ava (Seeker)"}

LAYER_POSITION_LABELS = {0: "30%", 1: "70%", 2: "Last"}


def build_contrastive_df(records: List[Dict]) -> pd.DataFrame:
    """One row per (episode, class, turn, agent, layer) with the log-prob diff."""
    rows = []
    for r in records:
        con = r.get("contrastive")
        if con is None:
            continue
        scenario = r["scenario"]
        true_idx = TRUE_LOC_INDEX[scenario]
        false_idx = 1 - true_idx
        compact_layers = con.get("compact_layers", {})
        for agent_num in (1, 2):
            agent_key = f"agent_{agent_num}"
            agent_data = con.get(agent_key, {})
            agent_layer_indices = compact_layers.get(agent_key, [])
            for cls_name, cls_data in agent_data.items():
                for turn_key in sorted(cls_data.keys(), key=int):
                    layers = cls_data[turn_key]
                    for pos, layer_scores in enumerate(layers):
                        layer_idx = agent_layer_indices[pos] if pos < len(agent_layer_indices) else pos
                        layer_label = LAYER_POSITION_LABELS.get(pos, f"pos{pos}")
                        if cls_name in LOCATION_CLASSES:
                            diff = layer_scores[true_idx] - layer_scores[false_idx]
                        else:
                            diff = layer_scores[0] - layer_scores[1]
                        rows.append({
                            "pair": r["pair"],
                            "scenario": scenario,
                            "condition": r["condition"],
                            "class": cls_name,
                            "turn": int(turn_key),
                            "agent": agent_num,
                            "layer_index": layer_idx,
                            "layer_label": layer_label,
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

_CANONICAL_STEM_RE = re.compile(
    r'(the \w+ is (?:in|on|at|by) the)',
    re.IGNORECASE,
)


def _canonical_stem(stem: str) -> str:
    """Normalize agent-specific and scenario-specific stems for aggregation."""
    for name in ("Ava", "Mia"):
        stem = stem.replace(name, "[Agent]")
    stem = _CANONICAL_STEM_RE.sub("the [item] is [prep]", stem)
    return stem


def build_stem_df(records: List[Dict]) -> pd.DataFrame:
    """One row per (episode, stem, turn, agent, layer) with true/false counts."""
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
                for pos, pred in enumerate(preds):
                    layer_idx = pred["layer_index"]
                    layer_label = LAYER_POSITION_LABELS.get(pos, f"pos{pos}")
                    pred_clean = pred["predicted_token_clean"].lower().strip("·▁ ")
                    tc = 1 if true_tok in pred_clean else 0
                    fc = 1 if (not tc and false_tok in pred_clean) else 0
                    rows.append({
                        "pair": r["pair"],
                        "scenario": scenario,
                        "condition": r["condition"],
                        "stem": stem_name,
                        "stem_canonical": _canonical_stem(stem_name),
                        "turn": turn_idx,
                        "agent": agent_num,
                        "layer_index": layer_idx,
                        "layer_label": layer_label,
                        "true_count": tc,
                        "false_count": fc,
                    })
    return pd.DataFrame(rows)


def plot_stem_for_turn_agent(
    df: pd.DataFrame, turn: int, agent: int,
    layer_label: str, out_dir: Path,
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
        f"Stem Predictions ({layer_label} Layer) — Turn {turn}, Agent {agent} ({role})\n"
        f"(n={n_episodes} episodes, averaged over scenarios & model pairs)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    safe_layer = layer_label.lower().replace(" ", "_").replace("%", "pct")
    fname = f"stem_{safe_layer}_turn{turn}_agent{agent}.png"
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
        plot_behavioral_per_pair(beh_df)
    else:
        print("  No behavioral data found.")

    # ── 2. Contrastive ──
    print("\n" + "=" * 60)
    print("2. CONTRASTIVE LOG-PROB DIFFS (per turn × agent × layer)")
    print("=" * 60)

    con_subdir = OUT_DIR / "contrastive"
    con_subdir.mkdir(exist_ok=True)

    con_df = build_contrastive_df(records)
    if not con_df.empty:
        con_df.sort_values(["layer_label", "turn", "agent", "class", "condition", "scenario", "pair"]) \
              .to_csv(OUT_DIR / "contrastive_compact_individual.csv", index=False)
        print(f"  {len(con_df)} rows saved to contrastive_compact_individual.csv")

        for ll in sorted(con_df["layer_label"].unique(), key=lambda s: list(LAYER_POSITION_LABELS.values()).index(s)):
            layer_df = con_df[con_df["layer_label"] == ll]
            layer_title = f"{ll} Layer"
            print(f"\n  --- {layer_title} ---")
            turn_agents = sorted(layer_df[["turn", "agent"]].drop_duplicates().values.tolist())
            print(f"  Generating plots for {len(turn_agents)} turn×agent combos...")
            for turn, agent in turn_agents:
                plot_contrastive_for_turn_agent(layer_df, turn, agent, layer_title, con_subdir)
    else:
        print("  No contrastive data found.")

    # ── 3. Stem ──
    print("\n" + "=" * 60)
    print("3. STEM TOKEN MENTIONS (per turn × agent × layer)")
    print("=" * 60)

    stem_subdir = OUT_DIR / "stem"
    stem_subdir.mkdir(exist_ok=True)

    stem_df = build_stem_df(records)
    if not stem_df.empty:
        stem_df.sort_values(["layer_label", "turn", "agent", "stem_canonical", "condition", "scenario", "pair"]) \
               .to_csv(OUT_DIR / "stem_compact_individual.csv", index=False)
        print(f"  {len(stem_df)} rows saved to stem_compact_individual.csv")

        for ll in sorted(stem_df["layer_label"].unique(), key=lambda s: list(LAYER_POSITION_LABELS.values()).index(s)):
            layer_df = stem_df[stem_df["layer_label"] == ll]
            layer_title = f"{ll} Layer"
            print(f"\n  --- {layer_title} ---")
            turn_agents = sorted(layer_df[["turn", "agent"]].drop_duplicates().values.tolist())
            print(f"  Generating plots for {len(turn_agents)} turn×agent combos...")
            for turn, agent in turn_agents:
                plot_stem_for_turn_agent(layer_df, turn, agent, layer_title, stem_subdir)
    else:
        print("  No stem data found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
