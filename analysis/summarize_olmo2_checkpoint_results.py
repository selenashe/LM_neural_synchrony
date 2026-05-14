#!/usr/bin/env python3
"""
Aggregate and plot results from OLMo checkpoint false-belief v3
single-prompt evaluations.

Supports multiple model families via --model_family:
  olmo2           (default) OLMo-2 training stages
  olmo3-instruct  OLMo-3 Instruct track (Base -> SFT -> DPO -> RLVR)
  olmo3-think     OLMo-3 Think track (SFT -> DPO -> RLVR)

Usage:
  # OLMo-2 (default, same as before):
  python analysis/summarize_olmo2_checkpoint_results.py

  # OLMo-3 Instruct:
  python analysis/summarize_olmo2_checkpoint_results.py \
      --model_family olmo3-instruct \
      --results_dir sotopia_results_olmo3_instruct_checkpoints_v3 \
      --output_dir summary_plots_olmo3_instruct_checkpoints_v3

  # OLMo-3 Think:
  python analysis/summarize_olmo2_checkpoint_results.py \
      --model_family olmo3-think \
      --results_dir sotopia_results_olmo3_think_checkpoints_v3 \
      --output_dir summary_plots_olmo3_think_checkpoints_v3
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOTOPIA_DATA = REPO_ROOT / "sotopia_utils" / "sotopia_data"

# ── Training stage ordering (per model family) ─────────────────────────────

MODEL_FAMILIES = {
    "olmo2": {
        "title": "OLMo-2",
        "stage_order": [
            "OLMo-2-1124-7B-Base",
            "OLMo-2-1124-7B-SFT",
            "OLMo-2-1124-7B-DPO",
            "OLMo-2-1124-7B-RLVR-step60",
            "OLMo-2-1124-7B-RLVR-step120",
            "OLMo-2-1124-7B-RLVR-step180",
            "OLMo-2-1124-7B-RLVR-step240",
            "OLMo-2-1124-7B-RLVR-step300",
            "OLMo-2-1124-7B-RLVR-step360",
            "OLMo-2-1124-7B-Instruct",
        ],
        "stage_labels": {
            "OLMo-2-1124-7B-Base": "Base",
            "OLMo-2-1124-7B-SFT": "SFT",
            "OLMo-2-1124-7B-DPO": "DPO",
            "OLMo-2-1124-7B-RLVR-step60": "RLVR\nstep60",
            "OLMo-2-1124-7B-RLVR-step120": "RLVR\nstep120",
            "OLMo-2-1124-7B-RLVR-step180": "RLVR\nstep180",
            "OLMo-2-1124-7B-RLVR-step240": "RLVR\nstep240",
            "OLMo-2-1124-7B-RLVR-step300": "RLVR\nstep300",
            "OLMo-2-1124-7B-RLVR-step360": "RLVR\nstep360",
            "OLMo-2-1124-7B-Instruct": "Instruct\n(final)",
        },
        "stage_short": {
            "OLMo-2-1124-7B-Base": "Base",
            "OLMo-2-1124-7B-SFT": "SFT",
            "OLMo-2-1124-7B-DPO": "DPO",
            "OLMo-2-1124-7B-RLVR-step60": "R-60",
            "OLMo-2-1124-7B-RLVR-step120": "R-120",
            "OLMo-2-1124-7B-RLVR-step180": "R-180",
            "OLMo-2-1124-7B-RLVR-step240": "R-240",
            "OLMo-2-1124-7B-RLVR-step300": "R-300",
            "OLMo-2-1124-7B-RLVR-step360": "R-360",
            "OLMo-2-1124-7B-Instruct": "Instruct",
        },
        "default_results_dir": "sotopia_results_olmo2_checkpoints_v3",
        "default_output_dir": "summary_plots_olmo2_checkpoints_v3",
    },
    "olmo3-instruct": {
        "title": "OLMo-3 Instruct",
        "stage_order": [
            "OLMo-3-1025-7B",
            "OLMo-3-7B-Instruct-SFT",
            "OLMo-3-7B-Instruct-DPO",
            "OLMo-3-7B-Instruct-RLVR-step400",
        ],
        "stage_labels": {
            "OLMo-3-1025-7B": "Base",
            "OLMo-3-7B-Instruct-SFT": "SFT",
            "OLMo-3-7B-Instruct-DPO": "DPO",
            "OLMo-3-7B-Instruct-RLVR-step400": "RLVR\nstep400",
        },
        "stage_short": {
            "OLMo-3-1025-7B": "Base",
            "OLMo-3-7B-Instruct-SFT": "SFT",
            "OLMo-3-7B-Instruct-DPO": "DPO",
            "OLMo-3-7B-Instruct-RLVR-step400": "R-400",
        },
        "default_results_dir": "sotopia_results_olmo3_instruct_checkpoints_v3",
        "default_output_dir": "summary_plots_olmo3_instruct_checkpoints_v3",
    },
    "olmo3-think": {
        "title": "OLMo-3 Think",
        "stage_order": [
            "OLMo-3-7B-Think-SFT",
            "OLMo-3-7B-Think-DPO",
            "OLMo-3-7B-Think-RLVR-step1375",
        ],
        "stage_labels": {
            "OLMo-3-7B-Think-SFT": "SFT",
            "OLMo-3-7B-Think-DPO": "DPO",
            "OLMo-3-7B-Think-RLVR-step1375": "RLVR\nstep1375",
        },
        "stage_short": {
            "OLMo-3-7B-Think-SFT": "SFT",
            "OLMo-3-7B-Think-DPO": "DPO",
            "OLMo-3-7B-Think-RLVR-step1375": "R-1375",
        },
        "default_results_dir": "sotopia_results_olmo3_think_checkpoints_v3",
        "default_output_dir": "summary_plots_olmo3_think_checkpoints_v3",
    },
}

# Default to olmo2 for backward compatibility when used as module globals
STAGE_ORDER = MODEL_FAMILIES["olmo2"]["stage_order"]
STAGE_LABELS = MODEL_FAMILIES["olmo2"]["stage_labels"]
STAGE_SHORT = MODEL_FAMILIES["olmo2"]["stage_short"]

# ── Condition definitions ────────────────────────────────────────────────────

CONDITIONS = [
    "mutual_C_cooperate",
    "mutual_A1_expert", "mutual_A2_unreliable",
    "mutual_A3_incentivized", "mutual_A4_malicious",
    "mutual_B1_forgetting", "mutual_B2_intervening", "mutual_B3_poor_vis",
    "asym_C_cooperate",
    "asym_A1_expert", "asym_A2_unreliable",
    "asym_A3_incentivized", "asym_A4_malicious",
    "asym_B1_forgetting", "asym_B2_intervening", "asym_B3_poor_vis",
    "conflict_C_cooperate",
    "conflict_A1_expert", "conflict_A2_unreliable",
    "conflict_A3_incentivized", "conflict_A4_malicious",
    "conflict_B1_forgetting", "conflict_B2_intervening", "conflict_B3_poor_vis",
]

CONDITION_LABELS = {
    "mutual_A1_expert": "Mutual\nA1-Expert",
    "mutual_A2_unreliable": "Mutual\nA2-Unreliable",
    "mutual_A3_incentivized": "Mutual\nA3-Incentiv.",
    "mutual_A4_malicious": "Mutual\nA4-Malicious",
    "mutual_B1_forgetting": "Mutual\nB1-Forgetting",
    "mutual_B2_intervening": "Mutual\nB2-Interven.",
    "mutual_B3_poor_vis": "Mutual\nB3-PoorVis",
    "mutual_C_cooperate": "Mutual\nC-Cooperate",
    "asym_A1_expert": "Asym\nA1-Expert",
    "asym_A2_unreliable": "Asym\nA2-Unreliable",
    "asym_A3_incentivized": "Asym\nA3-Incentiv.",
    "asym_A4_malicious": "Asym\nA4-Malicious",
    "asym_B1_forgetting": "Asym\nB1-Forgetting",
    "asym_B2_intervening": "Asym\nB2-Interven.",
    "asym_B3_poor_vis": "Asym\nB3-PoorVis",
    "asym_C_cooperate": "Asym\nC-Cooperate",
    "conflict_A1_expert": "Conflict\nA1-Expert",
    "conflict_A2_unreliable": "Conflict\nA2-Unreliable",
    "conflict_A3_incentivized": "Conflict\nA3-Incentiv.",
    "conflict_A4_malicious": "Conflict\nA4-Malicious",
    "conflict_B1_forgetting": "Conflict\nB1-Forgetting",
    "conflict_B2_intervening": "Conflict\nB2-Interven.",
    "conflict_B3_poor_vis": "Conflict\nB3-PoorVis",
    "conflict_C_cooperate": "Conflict\nC-Cooperate",
}

BELIEF_SHORT = {
    "mutual_knowledge": "mutual",
    "asymmetric_knowledge": "asym",
    "conflicting_evidence": "conflict",
}

# ── Load scenario data ───────────────────────────────────────────────────────

with open(SOTOPIA_DATA / "envs_false_belief_v3.json") as _f:
    _ENVS = json.load(_f)

with open(SOTOPIA_DATA / "scenario_vocab_false_belief_v2.json") as _f:
    _VOCAB = json.load(_f)

with open(
    SOTOPIA_DATA / "env_agent_combos_false_belief_v3_fixed_two_agents.json"
) as _f:
    _COMBOS = json.load(_f)

_ENV_BY_INDEX: Dict[int, dict] = {}
for idx, combo in enumerate(_COMBOS):
    env_id = combo["env_id"]
    _ENV_BY_INDEX[idx] = _ENVS[env_id]


# ── Parsing ──────────────────────────────────────────────────────────────────

def _extract_stage_from_dirname(dirname: str, stage_order: List[str]) -> Optional[str]:
    """Extract checkpoint name from directory like
    'OLMo-2-1124-7B-SFT_None_0_OLMo-2-1124-7B-SFT_None_0_false_belief_v3'.
    """
    parts = dirname.split("_None_0_")
    if len(parts) >= 2:
        candidate = parts[0]
        if candidate in stage_order:
            return candidate
    return None


def _parse_dialog(dialog_str: str) -> List[str]:
    try:
        return ast.literal_eval(dialog_str)
    except (ValueError, SyntaxError):
        return []


def _extract_ava_response(dialog: List[str]) -> Optional[str]:
    for turn in dialog:
        if "Ava Thompson said:" in turn:
            return turn.split("Ava Thompson said:")[-1].strip()
    return None


def _parse_ava_choice(ava_response: str, locations: List[str]) -> Optional[str]:
    if not ava_response:
        return None
    response_lower = ava_response.lower()

    # Prefer locations in non-question sentences. If a *different* location
    # appears in statement text, use that and ignore question-only mentions.
    # But if no location appears in statements (e.g. pronoun reference like
    # "I'll choose that location"), fall back to the full response.
    sentences = re.split(r"(?<=[.?!])\s*", response_lower)
    statement_text = " ".join(s for s in sentences if s and "?" not in s)

    statement_mentions = []
    if statement_text.strip():
        for loc in locations:
            idx = statement_text.rfind(loc.lower())
            if idx >= 0:
                statement_mentions.append((idx, loc))

    if statement_mentions:
        statement_mentions.sort(key=lambda x: x[0])
        return statement_mentions[-1][1]

    # No location in statement text. If the statement expresses doubt/negation,
    # the question-mentioned location is being rejected → null.
    # Otherwise fall back to full response (handles pronoun references like
    # "I'll choose that location").
    _DOUBT = re.compile(
        r"\b(don't|doesn't|didn't|not|no|never|can't|cannot|won't|n't"
        r"|doubt|wrong|mistaken|unsure|unlikely)\b"
    )
    if statement_text.strip() and _DOUBT.search(statement_text):
        return None

    mentions = []
    for loc in locations:
        idx = response_lower.rfind(loc.lower())
        if idx >= 0:
            mentions.append((idx, loc))
    if not mentions:
        return None
    mentions.sort(key=lambda x: x[0])
    return mentions[-1][1]


def load_results(results_root: Path, stage_order: List[str] = None) -> pd.DataFrame:
    if stage_order is None:
        stage_order = STAGE_ORDER
    dialogs_root = results_root / "dialogs"
    if not dialogs_root.exists():
        print(f"No dialogs directory at {dialogs_root}")
        return pd.DataFrame()

    rows = []
    for pair_dir in sorted(dialogs_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        stage = _extract_stage_from_dirname(pair_dir.name, stage_order)
        if stage is None:
            continue

        csv_files = sorted(pair_dir.glob("*_temp*_seed*.csv"))
        for csv_path in csv_files:
            if csv_path.parent.name == "prompt_records":
                continue
            stem = csv_path.stem
            match = re.match(r"(\d+)_temp[\d.]+_seed\d+", stem)
            if not match:
                continue
            ep_idx = int(match.group(1))
            env = _ENV_BY_INDEX.get(ep_idx)
            if env is None:
                continue

            meta = env["_meta"]
            scenario_type = meta["scenario_type"]
            vocab = _VOCAB.get(scenario_type)
            if vocab is None:
                continue

            try:
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    row_data = next(reader)
            except (StopIteration, KeyError):
                continue

            dialog = _parse_dialog(row_data.get("Dialog", "[]"))
            ava_response = _extract_ava_response(dialog)
            locations = vocab["locations"]
            true_location = meta["true_location"].replace("_", " ")

            b_gold = meta["b_gold"].replace("_", " ")
            ava_choice = _parse_ava_choice(ava_response, locations)

            condition = meta["condition"]
            belief_short = BELIEF_SHORT.get(meta["belief_condition"], "?")
            condition_key = f"{belief_short}_{condition.split('_', 1)[1]}" \
                if "_" in condition else condition

            a_gold = meta["a_gold"]
            if a_gold == "say_false":
                mia_location = [l for l in locations if l != true_location][0]
            else:
                mia_location = true_location

            correct = None
            if ava_choice is not None:
                correct = int(ava_choice == b_gold)

            follows_mia = None
            if ava_choice is not None:
                follows_mia = int(ava_choice == mia_location)

            rows.append({
                "stage": stage,
                "episode": ep_idx,
                "condition": condition_key,
                "belief_condition": meta["belief_condition"],
                "trust_manipulation": meta["trust_manipulation"],
                "goal_condition": meta["goal_condition"],
                "scenario_type": scenario_type,
                "true_location": true_location,
                "mia_location": mia_location,
                "ava_response": ava_response or "",
                "ava_choice": ava_choice or "",
                "b_gold": b_gold,
                "correct": correct,
                "follows_mia": follows_mia,
            })

    return pd.DataFrame(rows)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, stage_order: List[str] = None) -> pd.DataFrame:
    """Compute per-stage per-condition summary metrics."""
    if stage_order is None:
        stage_order = STAGE_ORDER
    records = []
    for stage in stage_order:
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            continue
        for cond in CONDITIONS:
            cond_df = stage_df[stage_df["condition"] == cond]
            n_total = len(cond_df)
            if n_total == 0:
                continue
            valid = cond_df[cond_df["correct"].notna()]
            n_valid = len(valid)
            n_null = n_total - n_valid
            accuracy = valid["correct"].mean() if n_valid > 0 else float("nan")
            follows_mia_rate = (
                valid["follows_mia"].mean() if n_valid > 0 else float("nan")
            )
            records.append({
                "stage": stage,
                "condition": cond,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_null": n_null,
                "null_rate": n_null / n_total if n_total > 0 else 0,
                "accuracy": accuracy,
                "follows_mia_rate": follows_mia_rate,
            })
    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trajectory(metrics_df: pd.DataFrame, out_dir: Path,
                    stage_order: List[str] = None, stage_short: Dict = None,
                    family_title: str = "OLMo-2") -> None:
    """Line plots: accuracy across training stages, one plot per
    (evidence_structure, condition_group) combination."""
    if stage_order is None:
        stage_order = STAGE_ORDER
    if stage_short is None:
        stage_short = STAGE_SHORT

    evidence_structures = {
        "mutual": [c for c in CONDITIONS if c.startswith("mutual_")],
        "asym": [c for c in CONDITIONS if c.startswith("asym_")],
        "conflict": [c for c in CONDITIONS if c.startswith("conflict_")],
    }
    condition_groups = {
        "A": lambda c: bool(re.match(r".*_A\d_", c)),
        "B": lambda c: bool(re.match(r".*_B\d_", c)),
    }

    evidence_labels = {
        "mutual": "Mutual Knowledge",
        "asym": "Asymmetric Knowledge",
        "conflict": "Conflicting Evidence",
    }
    group_labels = {
        "A": "Adversarial + Trust Manipulation",
        "B": "Adversarial + Evidence Manipulation",
    }

    x_positions = list(range(len(stage_order)))
    x_labels = [stage_short.get(s, s) for s in stage_order]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), sharey=True)

    for col_idx, ev_key in enumerate(["mutual", "asym", "conflict"]):
        ev_conds = evidence_structures[ev_key]
        for row_idx, (grp_key, grp_filter) in enumerate(condition_groups.items()):
            ax = axes[row_idx][col_idx]
            conds = [c for c in ev_conds if grp_filter(c)]
            for cond in conds:
                sub = metrics_df[metrics_df["condition"] == cond]
                if sub.empty:
                    continue
                y_vals = []
                for stage in stage_order:
                    row = sub[sub["stage"] == stage]
                    y_vals.append(row["accuracy"].values[0] if len(row) > 0 else float("nan"))
                label = CONDITION_LABELS.get(cond, cond).replace("\n", " ")
                ax.plot(x_positions, y_vals, marker="o", markersize=4, label=label)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{evidence_labels[ev_key]} — {group_labels[grp_key]}", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.6)
            ax.legend(fontsize=6, loc="lower right")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Behavioral Accuracy Across {family_title} Training Stages\n"
        "(single-prompt completion, false-belief v3)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_by_condition_group.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory_by_condition_group.png")



def plot_null_rate(metrics_df: pd.DataFrame, out_dir: Path,
                   stage_order: List[str] = None, stage_short: Dict = None) -> None:
    """Bar chart: null rate (unparseable responses) per training stage."""
    if stage_order is None:
        stage_order = STAGE_ORDER
    if stage_short is None:
        stage_short = STAGE_SHORT
    fig, ax = plt.subplots(figsize=(10, 5))

    stage_null = []
    for stage in stage_order:
        sub = metrics_df[metrics_df["stage"] == stage]
        if sub.empty:
            stage_null.append(0)
        else:
            total = sub["n_total"].sum()
            null = sub["n_null"].sum()
            stage_null.append(null / total if total > 0 else 0)

    x = np.arange(len(stage_order))
    labels = [stage_short.get(s, s) for s in stage_order]
    bars = ax.bar(x, stage_null, color="#e78ac3", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, stage_null):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Null Rate (fraction unparseable)")
    ax.set_title("Response Parse Failure Rate Across Training Stages")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "null_rate_by_stage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved null_rate_by_stage.png")


def plot_per_stage_conditions(metrics_df: pd.DataFrame, out_dir: Path,
                              stage_order: List[str] = None,
                              stage_labels: Dict = None) -> None:
    """Per-stage bar chart of accuracy across all 24 conditions."""
    if stage_order is None:
        stage_order = STAGE_ORDER
    if stage_labels is None:
        stage_labels = STAGE_LABELS
    stages_present = [s for s in stage_order if s in metrics_df["stage"].values]
    n_stages = len(stages_present)
    if n_stages == 0:
        return

    ncols = min(5, n_stages)
    nrows = (n_stages + ncols - 1) // ncols

    belief_colors = {"mutual": "#66c2a5", "asym": "#fc8d62", "conflict": "#8da0cb"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)
    x = np.arange(len(CONDITIONS))

    for idx, stage in enumerate(stages_present):
        ax = axes[idx // ncols][idx % ncols]
        sub = metrics_df[metrics_df["stage"] == stage]
        accs = sub.set_index("condition")["accuracy"].reindex(CONDITIONS)
        colors = [belief_colors.get(c.split("_")[0], "#999") for c in CONDITIONS]
        hatches = []
        for c in CONDITIONS:
            if re.match(r".*_A\d_", c):
                hatches.append("//")
            elif re.match(r".*_B\d_", c):
                hatches.append("..")
            else:
                hatches.append("")
        bars = ax.bar(x, accs.values, color=colors, edgecolor="black", linewidth=0.3)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c).replace("\n", " ") for c in CONDITIONS],
            fontsize=3.5, rotation=90,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_labels.get(stage, stage).replace("\n", " "),
                      fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.axhline(0.5, ls="--", color="gray", lw=0.5)
        for i in (8, 16):
            ax.axvline(i - 0.5, ls=":", color="gray", lw=0.3, alpha=0.7)

    for idx in range(n_stages, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66c2a5", label="Mutual Knowledge"),
        Patch(facecolor="#fc8d62", label="Asymmetric Knowledge"),
        Patch(facecolor="#8da0cb", label="Conflicting Evidence"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="A conditions (trust)"),
        Patch(facecolor="white", edgecolor="black", hatch="..", label="B conditions (evidence)"),
    ]
    fig.legend(handles=legend_elements, fontsize=8, loc="upper right")
    fig.suptitle(
        "Behavioral Accuracy by Condition — Per Training Stage",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_per_stage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved accuracy_per_stage.png")


def plot_adversarial_delta(metrics_df: pd.DataFrame, out_dir: Path,
                           stage_order: List[str] = None,
                           stage_short: Dict = None) -> None:
    """Grouped bar plot: delta (mean A accuracy - mean B accuracy) per evidence
    structure, for each checkpoint stage."""
    if stage_order is None:
        stage_order = STAGE_ORDER
    if stage_short is None:
        stage_short = STAGE_SHORT
    evidence_keys = ["mutual", "asym", "conflict"]
    evidence_labels = {
        "mutual": "Mutual",
        "asym": "Asymmetric",
        "conflict": "Conflicting",
    }
    evidence_colors = {
        "mutual": "#66c2a5",
        "asym": "#fc8d62",
        "conflict": "#8da0cb",
    }

    stages_present = [s for s in stage_order if s in metrics_df["stage"].values]
    n_stages = len(stages_present)
    if n_stages == 0:
        return

    deltas = {ev: [] for ev in evidence_keys}
    for stage in stages_present:
        sub = metrics_df[metrics_df["stage"] == stage]
        for ev in evidence_keys:
            a_conds = [c for c in CONDITIONS
                       if c.startswith(f"{ev}_") and re.match(r".*_A\d_", c)]
            b_conds = [c for c in CONDITIONS
                       if c.startswith(f"{ev}_") and re.match(r".*_B\d_", c)]
            a_acc = sub[sub["condition"].isin(a_conds)]["accuracy"].mean()
            b_acc = sub[sub["condition"].isin(b_conds)]["accuracy"].mean()
            delta = a_acc - b_acc if not (np.isnan(a_acc) or np.isnan(b_acc)) else float("nan")
            deltas[ev].append(delta)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_stages)
    n_groups = len(evidence_keys)
    bar_width = 0.25

    for i, ev in enumerate(evidence_keys):
        offset = (i - (n_groups - 1) / 2) * bar_width
        ax.bar(x + offset, deltas[ev], bar_width,
               label=evidence_labels[ev], color=evidence_colors[ev],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([stage_short.get(s, s) for s in stages_present],
                       fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Δ Accuracy (A − B)")
    ax.set_title("Adversarial Accuracy Delta (Trust Manipulation − Evidence Manipulation)\nby Evidence Structure")
    ax.axhline(0, ls="-", color="black", lw=0.8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "adversarial_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved adversarial_delta.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize OLMo checkpoint false-belief v3 results."
    )
    parser.add_argument(
        "--model_family", default="olmo2",
        choices=list(MODEL_FAMILIES.keys()),
        help="Model family (olmo2, olmo3-instruct, olmo3-think).",
    )
    parser.add_argument(
        "--results_dir", default=None,
        help="Results directory (relative to REPO_ROOT). "
             "Defaults to family-specific directory.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (relative to REPO_ROOT). "
             "Defaults to family-specific directory.",
    )
    args = parser.parse_args()

    family = MODEL_FAMILIES[args.model_family]
    stage_order = family["stage_order"]
    stage_labels = family["stage_labels"]
    stage_short = family["stage_short"]
    family_title = family["title"]

    results_dir = args.results_dir or family["default_results_dir"]
    output_dir = args.output_dir or family["default_output_dir"]

    results_root = REPO_ROOT / results_dir
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model family: {family_title}", flush=True)
    print("Loading results...", flush=True)
    df = load_results(results_root, stage_order)
    if df.empty:
        print("No results found. Exiting.")
        return

    stages_found = [s for s in stage_order if s in df["stage"].values]
    print(f"Found {len(df)} episodes across {len(stages_found)} stages:")
    for s in stages_found:
        n = len(df[df["stage"] == s])
        print(f"  {stage_short[s]:>10s}: {n} episodes")

    # Save raw per-episode data
    df.to_csv(out_dir / "all_episodes.csv", index=False)
    print(f"\nSaved all_episodes.csv ({len(df)} rows)")

    # Compute summary metrics
    print("\nComputing metrics...")
    metrics_df = compute_metrics(df, stage_order)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    print(f"Saved metrics_summary.csv ({len(metrics_df)} rows)")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"ACCURACY BY STAGE — {family_title} (averaged over all conditions)")
    print("=" * 80)
    for stage in stage_order:
        sub = metrics_df[metrics_df["stage"] == stage]
        if sub.empty:
            print(f"  {stage_short[stage]:>10s}: no data")
            continue
        acc = sub["accuracy"].mean()
        null = sub["null_rate"].mean()
        n = sub["n_total"].sum()
        print(f"  {stage_short[stage]:>10s}: acc={acc:.3f}  null_rate={null:.3f}  N={n}")

    # Generate plots
    print("\nGenerating plots...")
    plot_trajectory(metrics_df, out_dir, stage_order, stage_short, family_title)
    plot_null_rate(metrics_df, out_dir, stage_order, stage_short)
    plot_per_stage_conditions(metrics_df, out_dir, stage_order, stage_labels)
    plot_adversarial_delta(metrics_df, out_dir, stage_order, stage_short)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
