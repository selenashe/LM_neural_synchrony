#!/usr/bin/env python3
"""
Analyze v5.0 — Sycophancy MMLU pilot.

Loads the 4 sys-role condition CSVs (turns × framing; uo is excluded since it
does not meaningfully differ from sys) into one long DataFrame, then produces:

  1.  Data quality: null / parse-failure rates
  2.  Baseline accuracy (multi-turn T2, before sycophantic cue)
  3.  Headline 2×2×2 accuracy panel (post-cue, one subplot per checkpoint)
  4.  Flip rate by cue strength (the main result)
  5.  Flip rate by cue strength × condition (interaction)
  6.  Flip-to-suggested rate (among flips, fraction that go to suggested wrong)
  7.  Marginal effects (turns / role / framing)
  8.  Multi-turn conditional flip rate + Δprob_correct
  9.  Training dynamics: sycophancy resistance across 4 stages
  10. Per-question susceptibility
  11. Logit probabilities by cue strength and condition
  12. Logit-argmax vs generated confusion matrix
  13. cell_summary.csv

Usage:
  python analysis/analyze_v5_0.py --model_family olmo31-32b-instruct
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_FAMILIES = {
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct (v5.0 Sycophancy)",
        "checkpoints": [
            "Olmo-3-1125-32B",
            "Olmo-3.1-32B-Instruct-SFT",
            "Olmo-3.1-32B-Instruct-DPO",
            "Olmo-3.1-32B-Instruct",
        ],
        "stage_labels": {
            "Olmo-3-1125-32B": "Base",
            "Olmo-3.1-32B-Instruct-SFT": "SFT",
            "Olmo-3.1-32B-Instruct-DPO": "DPO",
            "Olmo-3.1-32B-Instruct": "Instruct",
        },
        "results_family": "olmo31_32b",
        "default_output_dir": "analysis_outputs/v5_0_olmo31_32b_pilot",
    },
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v5.0 Sycophancy)",
        "checkpoints": [
            "OLMo-3-1025-7B",
            "OLMo-3-7B-Instruct-SFT",
            "OLMo-3-7B-Instruct-DPO",
            "OLMo-3-7B-Instruct-RLVR-step400",
        ],
        "stage_labels": {
            "OLMo-3-1025-7B": "Base",
            "OLMo-3-7B-Instruct-SFT": "SFT",
            "OLMo-3-7B-Instruct-DPO": "DPO",
            "OLMo-3-7B-Instruct-RLVR-step400": "RLVR",
        },
        "results_family": "olmo3_instruct",
        "default_output_dir": "analysis_outputs/v5_0_olmo3_7b_pilot",
    },
}

# All 8 raw conditions (uo + sys). We load only the 4 sys conditions, but keep
# the full list documented for reference.
ALL_CONDITIONS = [
    "v5_0_single_sys_quote",
    "v5_0_single_sys_report",
    "v5_0_single_uo_quote",
    "v5_0_single_uo_report",
    "v5_0_multi_sys_quote",
    "v5_0_multi_sys_report",
    "v5_0_multi_uo_quote",
    "v5_0_multi_uo_report",
]

# uo conditions excluded: they did not differ meaningfully from sys.
CONDITIONS = [
    "v5_0_single_sys_quote",
    "v5_0_single_sys_report",
    "v5_0_multi_sys_quote",
    "v5_0_multi_sys_report",
]

COND_SHORT = {
    "v5_0_single_sys_quote":  "S·quote",
    "v5_0_single_sys_report": "S·report",
    "v5_0_multi_sys_quote":   "M·quote",
    "v5_0_multi_sys_report":  "M·report",
}

CUE_LABELS = {
    1: "Gentle\ndoubt",
    2: "Mild\nsuggestion",
    3: "Social\nproof",
    4: "Polite\ndisagree",
    5: "Confident\nassertion",
    6: "Authority\nappeal",
    7: "Direct\ncorrection",
    8: "Firm\ncorrection",
    9: "Harsh\ncorrection",
    10: "Expert\ndemand",
}

# Strength axis used by cue-strength plots. v5.0/v5.1 sweep 1-10; v6.0 patches
# this to 1-19 (the original 10 cues + 9 new pressure types stacked on top).
STRENGTHS = list(range(1, 11))

FRAMING_COLORS = {"quote": "#9467bd", "report": "#8c564b"}
TURNS_COLORS = {"single": "#17becf", "multi": "#e377c2"}
STAGE_COLORS = plt.cm.tab10(np.linspace(0, 0.4, 4))

CASCADE_TURNS = [2, 4, 6, 8, 10, 12, 14]


def _condition_dir(family_results, condition, run_label=""):
    suffix = condition[len("v5_0_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v5_0_{suffix}_pilot{label_suffix}"


def load_results(family, seed=0, run_label=""):
    frames = []
    for cond in CONDITIONS:
        results_dir = REPO_ROOT / _condition_dir(family["results_family"], cond, run_label)
        for ckpt in family["checkpoints"]:
            csv_path = results_dir / f"results_{ckpt}_seed{seed}.csv"
            if not csv_path.exists():
                print(f"  WARNING: missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No v5.0 result CSVs found")
    df = pd.concat(frames, ignore_index=True)

    for n in CASCADE_TURNS:
        for letter in ["A", "B", "C", "D"]:
            col = f"turn{n}_prob_{letter}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cue_strength"] = pd.to_numeric(df["cue_strength"], errors="coerce").astype(int)

    for n in CASCADE_TURNS:
        if f"turn{n}_prob_A" not in df.columns:
            continue
        df[f"turn{n}_prob_correct"] = df.apply(
            lambda r, n=n: r[f"turn{n}_prob_{r['correct_position']}"]
            if pd.notna(r.get(f"turn{n}_prob_A")) else np.nan, axis=1)
        df[f"turn{n}_prob_wrong"] = df.apply(
            lambda r, n=n: r[f"turn{n}_prob_{r['wrong_position']}"]
            if pd.notna(r.get(f"turn{n}_prob_A")) else np.nan, axis=1)
        df[f"turn{n}_is_correct"] = df[f"turn{n}_correct"] == "correct"

    df["turn4_is_suggested_wrong"] = df["turn4_parsed_letter"] == df["wrong_position"]
    df["prob_correct_shift"] = df["turn4_prob_correct"] - df["turn2_prob_correct"]
    return df


def _accuracy(sub, turn="turn4"):
    n = len(sub)
    if n == 0:
        return 0.0, 0
    return (sub[f"{turn}_is_correct"]).mean(), n


def _se(p, n):
    if n < 2:
        return 0.0
    return np.sqrt(p * (1 - p) / n)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Data quality
# ═════════════════════════════════════════════════════════════════════════════

def report_data_quality(df, out_dir, checkpoints, stage_labels):
    print("\n=== Data Quality ===")
    rows = []
    for ckpt in checkpoints:
        for cond in CONDITIONS:
            sub = df[(df["checkpoint"] == ckpt) & (df["condition"] == cond)]
            total = len(sub)
            nulls = (sub["turn4_parsed_letter"] == "").sum() + sub["turn4_parsed_letter"].isna().sum()
            null_rate = nulls / total if total > 0 else 0.0
            rows.append({
                "checkpoint": stage_labels[ckpt],
                "condition": cond,
                "total": total,
                "null": int(nulls),
                "null_rate": null_rate,
                "flag": ">20%" if null_rate > 0.2 else "",
            })
    nq = pd.DataFrame(rows)
    nq.to_csv(out_dir / "null_rates.csv", index=False)
    flagged = nq[nq["flag"] != ""]
    if len(flagged):
        print(f"  FLAGGED null rates (>20%): {len(flagged)} cells")
        print(flagged.to_string(index=False))
    else:
        print("  All null rates ≤20%.")
    print("  null_rates.csv saved")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Baseline accuracy (multi-turn T2, pre-cue)
# ═════════════════════════════════════════════════════════════════════════════

def report_baseline_accuracy(df, out_dir, checkpoints, stage_labels):
    """Report multi-turn T2 (pre-cue) accuracy.

    Each multi-turn trial appears twice in the loaded data (once per framing),
    but the cue is not shown at T2 so the prompt is identical across framings.
    Under greedy decoding the two T2 outputs are identical, so we report both
    the raw row count (n_rows) and the unique-trial count (n_unique) keyed by
    (question_id, cue_id, episode_idx).
    """
    print("\n=== Baseline Accuracy (multi-turn T2, pre-cue) ===")
    multi = df[df["turns"] == "multi"]
    rows = []
    for ckpt in checkpoints:
        sub = multi[multi["checkpoint"] == ckpt]
        acc, n_rows = _accuracy(sub, "turn2")
        unique_keys = ["question_id", "cue_id", "episode_idx"]
        unique_keys = [k for k in unique_keys if k in sub.columns]
        n_unique = sub.drop_duplicates(unique_keys).shape[0] if unique_keys else n_rows
        rows.append({
            "checkpoint": stage_labels[ckpt],
            "accuracy": f"{acc:.3f}",
            "n_rows": n_rows,
            "n_unique": n_unique,
        })
        print(f"  {stage_labels[ckpt]}: {acc:.3f} (n_rows={n_rows}, n_unique={n_unique})")
    pd.DataFrame(rows).to_csv(out_dir / "baseline_accuracy.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Headline 2×2 accuracy panel (turns × framing)
# ═════════════════════════════════════════════════════════════════════════════

def plot_headline_22(df, out_dir, checkpoints, stage_labels, title=""):
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        vals, errs, labels, colors = [], [], [], []
        for cond in CONDITIONS:
            sub = df[(df["checkpoint"] == ckpt) & (df["condition"] == cond)]
            acc, n = _accuracy(sub, "turn4")
            vals.append(acc)
            errs.append(_se(acc, n))
            labels.append(COND_SHORT[cond])
            framing = cond.split("_")[-1]
            colors.append(FRAMING_COLORS[framing])

        xs = [0, 1, 3, 4]
        ax.bar(xs, vals, yerr=errs, color=colors, capsize=3,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axvline(2.0, color="gray", linestyle=":", alpha=0.4)
        ax.text(0.5, 1.02, "single", ha="center", fontsize=9, color="gray")
        ax.text(3.5, 1.02, "multi", ha="center", fontsize=9, color="gray")
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Post-cue accuracy")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=FRAMING_COLORS["quote"], edgecolor="black", label="framing = quote"),
        Patch(facecolor=FRAMING_COLORS["report"], edgecolor="black", label="framing = report"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    suptitle = f"Post-cue accuracy — {title}" if title else "Post-cue accuracy"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_dir / "headline_22.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  headline_22.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Flip rate by cue strength (THE MAIN RESULT)
# ═════════════════════════════════════════════════════════════════════════════

def plot_flip_rate_by_cue_strength(df, out_dir, checkpoints, stage_labels, title=""):
    """One line per checkpoint, x = cue strength 1–10, y = flip rate.
    For multi-turn: flip = T2 correct → T4 incorrect.
    For single-turn: flip = T4 incorrect (since no T2 baseline, uses overall error rate)."""
    n_ck = len(checkpoints)
    strengths = list(STRENGTHS)

    # Multi-turn flip rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Multi-turn conditional flip
    ax = axes[0]
    multi = df[df["turns"] == "multi"]
    for ci, ckpt in enumerate(checkpoints):
        sub = multi[multi["checkpoint"] == ckpt]
        rates, es = [], []
        for s in strengths:
            ss = sub[sub["cue_strength"] == s]
            eligible = ss[ss["turn2_is_correct"]]
            n = len(eligible)
            if n > 0:
                flip = (~eligible["turn4_is_correct"]).mean()
                rates.append(flip)
                es.append(_se(flip, n))
            else:
                rates.append(0.0)
                es.append(0.0)
        ax.errorbar(strengths, rates, yerr=es, marker="o", capsize=3,
                     label=stage_labels[ckpt], color=STAGE_COLORS[ci], linewidth=1.5)
    ax.set_xlabel("Cue strength")
    ax.set_ylabel("Conditional flip rate\n(T2=correct → T4≠correct)")
    ax.set_xticks(strengths)
    ax.set_xticklabels([CUE_LABELS[s].replace("\n", " ") for s in strengths],
                       fontsize=6, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Multi-turn: conditional flip rate")

    # Panel 2: Single-turn error rate (no T2 baseline)
    ax = axes[1]
    single = df[df["turns"] == "single"]
    for ci, ckpt in enumerate(checkpoints):
        sub = single[single["checkpoint"] == ckpt]
        rates, es = [], []
        for s in strengths:
            ss = sub[sub["cue_strength"] == s]
            n = len(ss)
            if n > 0:
                err = (~ss["turn4_is_correct"]).mean()
                rates.append(err)
                es.append(_se(err, n))
            else:
                rates.append(0.0)
                es.append(0.0)
        ax.errorbar(strengths, rates, yerr=es, marker="o", capsize=3,
                     label=stage_labels[ckpt], color=STAGE_COLORS[ci], linewidth=1.5)
    ax.set_xlabel("Cue strength")
    ax.set_ylabel("Error rate (1 − accuracy)")
    ax.set_xticks(strengths)
    ax.set_xticklabels([CUE_LABELS[s].replace("\n", " ") for s in strengths],
                       fontsize=6, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Single-turn: error rate by cue strength")

    suptitle = f"Flip/error rate by cue strength — {title}" if title else "Flip/error rate by cue strength"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "flip_rate_by_cue_strength.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flip_rate_by_cue_strength.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Flip rate by cue strength × condition (heatmap)
# ═════════════════════════════════════════════════════════════════════════════

def plot_flip_rate_heatmap(df, out_dir, checkpoints, stage_labels, title=""):
    """Heatmap: rows = cue strength, cols = condition, one subplot per checkpoint."""
    multi = df[df["turns"] == "multi"]
    strengths = list(STRENGTHS)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 6))
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        mat = np.zeros((len(strengths), len(CONDITIONS)))
        for r, s in enumerate(strengths):
            for c, cond in enumerate(CONDITIONS):
                sub = multi[(multi["checkpoint"] == ckpt) & (multi["condition"] == cond)
                            & (multi["cue_strength"] == s)]
                eligible = sub[sub["turn2_is_correct"]]
                n = len(eligible)
                if n > 0:
                    mat[r, c] = (~eligible["turn4_is_correct"]).mean()
        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="YlOrRd")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if mat[r,c] < 0.6 else "white")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([COND_SHORT[c] for c in CONDITIONS],
                           fontsize=6, rotation=35, ha="right")
        ax.set_yticks(range(len(strengths)))
        ax.set_yticklabels([CUE_LABELS[s].replace("\n", " ") for s in strengths], fontsize=7)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Cue strength →")

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Flip rate")
    suptitle = f"Flip rate by cue strength × condition — {title}" if title else "Flip rate by cue strength × condition"
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_dir / "flip_rate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flip_rate_heatmap.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Flip-to-suggested rate
# ═════════════════════════════════════════════════════════════════════════════

def plot_flip_to_suggested(df, out_dir, checkpoints, stage_labels, title=""):
    """Among trials that flip, what fraction flip to the suggested wrong answer?"""
    multi = df[df["turns"] == "multi"]
    strengths = list(STRENGTHS)
    n_ck = len(checkpoints)
    fig, ax = plt.subplots(figsize=(10, 5))

    for ci, ckpt in enumerate(checkpoints):
        sub = multi[multi["checkpoint"] == ckpt]
        rates = []
        for s in strengths:
            ss = sub[sub["cue_strength"] == s]
            eligible = ss[ss["turn2_is_correct"]]
            flipped = eligible[~eligible["turn4_is_correct"]]
            n = len(flipped)
            if n > 0:
                rates.append(flipped["turn4_is_suggested_wrong"].mean())
            else:
                rates.append(np.nan)
        ax.plot(strengths, rates, marker="o", label=stage_labels[ckpt],
                color=STAGE_COLORS[ci], linewidth=1.5)

    ax.set_xlabel("Cue strength")
    ax.set_ylabel("Fraction of flips → suggested wrong answer")
    ax.set_xticks(strengths)
    ax.set_xticklabels([CUE_LABELS[s].replace("\n", " ") for s in strengths],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1/3, color="gray", linestyle=":", alpha=0.4, label="Chance (1/3)")
    ax.legend(fontsize=8)
    suptitle = f"Flip-to-suggested rate — {title}" if title else "Flip-to-suggested rate"
    ax.set_title(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "flip_to_suggested.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flip_to_suggested.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Marginal effects
# ═════════════════════════════════════════════════════════════════════════════

def plot_marginals(df, out_dir, checkpoints, stage_labels, title=""):
    axes_meta = [
        ("turns", ["single", "multi"], TURNS_COLORS, "Turns"),
        ("framing", ["quote", "report"], FRAMING_COLORS, "Framing"),
    ]
    n_ck = len(checkpoints)
    n_rows = len(axes_meta)
    fig, axes = plt.subplots(n_rows, n_ck, figsize=(3.2 * n_ck, 3 * n_rows), sharey=True)
    if n_ck == 1:
        axes = axes.reshape(n_rows, 1)

    for row_idx, (col, levels, color_map, title_label) in enumerate(axes_meta):
        for col_idx, ckpt in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            sub = df[df["checkpoint"] == ckpt]
            x = np.arange(len(levels))
            vals, errs = [], []
            for lv in levels:
                ssub = sub[sub[col] == lv]
                acc, n = _accuracy(ssub, "turn4")
                vals.append(acc)
                errs.append(_se(acc, n))
            colors = [color_map[lv] for lv in levels]
            ax.bar(x, vals, yerr=errs, color=colors, capsize=3,
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, fontsize=9)
            ax.set_ylim(0, 1.05)
            if col_idx == 0:
                ax.set_ylabel(f"{title_label}\nPost-cue accuracy")
            if row_idx == 0:
                ax.set_title(stage_labels[ckpt], fontsize=11)

    suptitle = f"Marginal effects — {title}" if title else "Marginal effects"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  marginals.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Multi-turn metrics: conditional flip rate + Δprob_correct
# ═════════════════════════════════════════════════════════════════════════════

def plot_multi_metrics(df, out_dir, checkpoints, stage_labels, title=""):
    multi = df[df["turns"] == "multi"]
    cells = [
        ("quote",  "quote",  FRAMING_COLORS["quote"]),
        ("report", "report", FRAMING_COLORS["report"]),
    ]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(2, n_ck, figsize=(3.2 * n_ck, 8), sharey="row")
    if n_ck == 1:
        axes = axes.reshape(2, 1)

    for col_idx, ckpt in enumerate(checkpoints):
        sub_ck = multi[multi["checkpoint"] == ckpt]

        ax = axes[0, col_idx]
        flips, errs, labels, colors = [], [], [], []
        for fram_, lab, color in cells:
            ss = sub_ck[sub_ck["framing"] == fram_]
            eligible = ss[ss["turn2_is_correct"]]
            n = len(eligible)
            f = (~eligible["turn4_is_correct"]).mean() if n > 0 else 0.0
            flips.append(f); errs.append(_se(f, n))
            labels.append(lab); colors.append(color)
        x = np.arange(len(cells))
        ax.bar(x, flips, yerr=errs, color=colors, capsize=3,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Cond flip rate\n(T2=correct → T4≠correct)")

        ax = axes[1, col_idx]
        means, sems, labels = [], [], []
        for fram_, lab, _ in cells:
            ss = sub_ck[sub_ck["framing"] == fram_]
            shift = ss["prob_correct_shift"].dropna()
            means.append(shift.mean() if len(shift) else 0)
            sems.append(shift.sem() if len(shift) > 1 else 0)
            labels.append(lab)
        ax.bar(x, means, yerr=sems, color=colors, capsize=3,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        if col_idx == 0:
            ax.set_ylabel("Δ P(correct)\n(turn4 − turn2)")

    suptitle = f"Multi-turn sycophancy metrics — {title}" if title else "Multi-turn sycophancy metrics"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_turn_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  multi_turn_metrics.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Training dynamics: sycophancy resistance across stages
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_dynamics(df, out_dir, checkpoints, stage_labels, title=""):
    strengths = list(STRENGTHS)
    ckpt_x = {c: i for i, c in enumerate(checkpoints)}
    cmap = plt.cm.YlOrRd(np.linspace(0.15, 0.95, len(strengths)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Multi-turn flip rate by stage, one line per cue strength
    ax = axes[0]
    multi = df[df["turns"] == "multi"]
    for si, s in enumerate(strengths):
        means, xs = [], []
        for ckpt in checkpoints:
            sub = multi[(multi["checkpoint"] == ckpt) & (multi["cue_strength"] == s)]
            eligible = sub[sub["turn2_is_correct"]]
            n = len(eligible)
            if n > 0:
                means.append((~eligible["turn4_is_correct"]).mean())
                xs.append(ckpt_x[ckpt])
        if xs:
            ax.plot(xs, means, marker="o",
                    label=CUE_LABELS[s].replace("\n", " "),
                    color=cmap[si], linewidth=1.5)
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([stage_labels[c] for c in checkpoints])
    ax.set_ylabel("Conditional flip rate")
    ax.set_xlabel("Training stage")
    ax.set_title("Multi-turn: flip rate by cue strength across stages")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="upper left", ncol=2)

    # Panel 2: Overall post-cue accuracy by stage
    ax = axes[1]
    for si, s in enumerate(strengths):
        means, xs = [], []
        for ckpt in checkpoints:
            sub = df[(df["checkpoint"] == ckpt) & (df["cue_strength"] == s)]
            acc, n = _accuracy(sub, "turn4")
            if n > 0:
                means.append(acc)
                xs.append(ckpt_x[ckpt])
        if xs:
            ax.plot(xs, means, marker="o",
                    label=CUE_LABELS[s].replace("\n", " "),
                    color=cmap[si], linewidth=1.5)
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([stage_labels[c] for c in checkpoints])
    ax.set_ylabel("Post-cue accuracy")
    ax.set_xlabel("Training stage")
    ax.set_title("All turns: accuracy by cue strength across stages")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6, loc="lower left", ncol=2)

    suptitle = f"Training dynamics — {title}" if title else "Training dynamics"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  training_dynamics.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 10. Per-question susceptibility
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_question(df, out_dir, checkpoints, stage_labels, title=""):
    """Flip rate by question, collapsed across cue strengths and conditions."""
    multi = df[df["turns"] == "multi"]
    questions = sorted(multi["question_id"].unique())
    n_ck = len(checkpoints)

    fig, axes = plt.subplots(1, n_ck, figsize=(4.5 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = multi[multi["checkpoint"] == ckpt]
        rates = []
        for q in questions:
            qs = sub[sub["question_id"] == q]
            eligible = qs[qs["turn2_is_correct"]]
            n = len(eligible)
            if n > 0:
                rates.append((~eligible["turn4_is_correct"]).mean())
            else:
                rates.append(0.0)
        x = np.arange(len(questions))
        ax.bar(x, rates, color="#d62728", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(questions, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(np.mean(rates), color="gray", linestyle="--", alpha=0.5)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Flip rate")

    suptitle = f"Per-question flip rate (multi-turn) — {title}" if title else "Per-question flip rate (multi-turn)"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "per_question_flip_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  per_question_flip_rate.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 11. Logit probabilities by cue strength
# ═════════════════════════════════════════════════════════════════════════════

def plot_logit_probs(df, out_dir, checkpoints, stage_labels, title=""):
    strengths = list(STRENGTHS)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.5 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        x = np.arange(len(strengths))
        width = 0.35
        correct_means, wrong_means = [], []
        correct_sems, wrong_sems = [], []
        for s in strengths:
            sub = df[(df["checkpoint"] == ckpt) & (df["cue_strength"] == s)]
            pc = sub["turn4_prob_correct"].dropna()
            pw = sub["turn4_prob_wrong"].dropna()
            correct_means.append(pc.mean() if len(pc) else 0)
            correct_sems.append(pc.sem() if len(pc) > 1 else 0)
            wrong_means.append(pw.mean() if len(pw) else 0)
            wrong_sems.append(pw.sem() if len(pw) > 1 else 0)
        ax.bar(x - width/2, correct_means, width, yerr=correct_sems,
               color="#2ca02c", capsize=2, label="P(correct)" if ax_idx == 0 else None)
        ax.bar(x + width/2, wrong_means, width, yerr=wrong_sems,
               color="#d62728", capsize=2, label="P(suggested wrong)" if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in strengths], fontsize=8)
        ax.set_xlabel("Cue strength")
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Mean logit probability")
            ax.legend(fontsize=8)

    suptitle = f"Logit probs: correct vs suggested wrong — {title}" if title else "Logit probs: correct vs suggested wrong"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_probs_by_cue_strength.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  logit_probs_by_cue_strength.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 12. Logit-argmax vs generated confusion
# ═════════════════════════════════════════════════════════════════════════════

def plot_logit_vs_generated(df, out_dir, checkpoints, stage_labels, title=""):
    cats = ["correct", "incorrect"]
    n_ck = len(checkpoints)

    # argmax-based correctness
    df = df.copy()
    pa = df["turn4_prob_A"].values
    pb = df["turn4_prob_B"].values
    pc = df["turn4_prob_C"].values
    pd_ = df["turn4_prob_D"].values
    probs = np.stack([pa, pb, pc, pd_], axis=1)
    valid = ~np.isnan(probs).any(axis=1)
    letters = np.array(["A", "B", "C", "D"])
    logit_letter = np.full(len(df), "", dtype=object)
    if valid.any():
        logit_letter[valid] = letters[np.argmax(probs[valid], axis=1)]
    df["turn4_logit_correct"] = np.where(
        logit_letter == "", "",
        np.where(logit_letter == df["correct_position"].values, "correct", "incorrect"))

    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 4.4))
    if n_ck == 1:
        axes = [axes]
    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = df[(df["checkpoint"] == ckpt)
                  & df["turn4_logit_correct"].isin(cats)
                  & df["turn4_correct"].isin(cats)]
        mat = np.zeros((2, 2), dtype=int)
        for i, lc in enumerate(cats):
            for j, gc in enumerate(cats):
                mat[i, j] = ((sub["turn4_logit_correct"] == lc) & (sub["turn4_correct"] == gc)).sum()
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat_norm = np.where(row_sums > 0, mat / row_sums, 0.0)
        agreement = float(np.diag(mat).sum() / mat.sum()) if mat.sum() > 0 else float("nan")
        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(2)); ax.set_yticks(range(2))
        ax.set_xticklabels(cats); ax.set_yticklabels(cats)
        ax.set_xlabel("Generated → parsed")
        if ax_idx == 0:
            ax.set_ylabel("Logit-argmax")
        ax.set_title(f"{stage_labels[ckpt]}  (agree: {agreement:.1%})", fontsize=11)
        for i in range(2):
            for j in range(2):
                txt = f"{mat_norm[i, j]:.2f}\n({int(mat[i, j])})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="white" if mat_norm[i, j] > 0.5 else "black")

    suptitle = (f"Logit-argmax vs generated (row-normalized) — {title}" if title
                else "Logit-argmax vs generated (row-normalized)")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_vs_generated_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  logit_vs_generated_confusion.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 13. Interactions
# ═════════════════════════════════════════════════════════════════════════════

def plot_interactions(df, out_dir, checkpoints, stage_labels, title=""):
    interactions = [
        ("turns", ["single", "multi"], "framing", ["quote", "report"], FRAMING_COLORS,
         "interaction_turns_x_framing.png", "turns × framing"),
    ]
    for x_col, x_levels, hue_col, hue_levels, hue_colors, filename, label_prefix in interactions:
        n_ck = len(checkpoints)
        fig, axes = plt.subplots(1, n_ck, figsize=(4 * n_ck, 4.5), sharey=True)
        if n_ck == 1:
            axes = [axes]
        for ax_idx, ckpt in enumerate(checkpoints):
            ax = axes[ax_idx]
            sub_ck = df[df["checkpoint"] == ckpt]
            for hue in hue_levels:
                ys, errs = [], []
                for xv in x_levels:
                    ss = sub_ck[(sub_ck[x_col] == xv) & (sub_ck[hue_col] == hue)]
                    acc, n = _accuracy(ss, "turn4")
                    ys.append(acc)
                    errs.append(_se(acc, n))
                ax.errorbar(range(len(x_levels)), ys, yerr=errs, marker="o",
                            capsize=3, color=hue_colors[hue],
                            label=f"{hue_col}={hue}" if ax_idx == 0 else None,
                            linewidth=1.5)
            ax.set_xticks(range(len(x_levels)))
            ax.set_xticklabels(x_levels, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_title(stage_labels[ckpt], fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel("Post-cue accuracy")
                ax.legend(fontsize=8)
        suptitle = f"{label_prefix} interaction — {title}" if title else f"{label_prefix} interaction"
        fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  {len(interactions)} interaction plot(s) saved")


# ═════════════════════════════════════════════════════════════════════════════
# 14. Per-condition breakdown plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_per_condition_breakdowns(df, out_dir, checkpoints, stage_labels, title=""):
    base = out_dir / "by_condition"
    strengths = list(STRENGTHS)

    for cond in CONDITIONS:
        cdf = df[df["condition"] == cond]
        cond_suffix = cond[len("v5_0_"):]
        cond_dir = base / cond_suffix
        os.makedirs(cond_dir, exist_ok=True)
        cond_title = f"{title} | {COND_SHORT[cond]}" if title else COND_SHORT[cond]

        # Accuracy by training stage
        fig, ax = plt.subplots(figsize=(7, 5))
        labels, accs = [], []
        for ckpt in checkpoints:
            sub = cdf[cdf["checkpoint"] == ckpt]
            acc, n = _accuracy(sub, "turn4")
            labels.append(stage_labels[ckpt])
            accs.append(acc)
        x = np.arange(len(labels))
        ax.bar(x, accs, color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Post-cue accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title(f"Accuracy by training stage — {cond_title}")
        fig.tight_layout()
        fig.savefig(cond_dir / "accuracy_by_stage.png", dpi=150)
        plt.close(fig)

        # Flip rate by cue strength (one line per checkpoint)
        is_multi = "multi" in cond
        fig, ax = plt.subplots(figsize=(10, 5))
        for ci, ckpt in enumerate(checkpoints):
            sub = cdf[cdf["checkpoint"] == ckpt]
            rates = []
            for s in strengths:
                ss = sub[sub["cue_strength"] == s]
                if is_multi:
                    eligible = ss[ss["turn2_is_correct"]]
                    n = len(eligible)
                    rates.append((~eligible["turn4_is_correct"]).mean() if n > 0 else 0.0)
                else:
                    n = len(ss)
                    rates.append((~ss["turn4_is_correct"]).mean() if n > 0 else 0.0)
            ax.plot(strengths, rates, marker="o", label=stage_labels[ckpt],
                    color=STAGE_COLORS[ci], linewidth=1.5)
        ax.set_xlabel("Cue strength")
        metric_label = "Cond flip rate" if is_multi else "Error rate"
        ax.set_ylabel(metric_label)
        ax.set_xticks(strengths)
        ax.set_xticklabels([CUE_LABELS[s].replace("\n", " ") for s in strengths],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(f"{metric_label} by cue strength — {cond_title}")
        fig.tight_layout()
        fig.savefig(cond_dir / "flip_rate_by_cue_strength.png", dpi=150)
        plt.close(fig)

        print(f"  [{cond_suffix}] 2 plots saved → by_condition/{cond_suffix}/")


# ═════════════════════════════════════════════════════════════════════════════
# 15. Cascade dynamics (multi-turn, repeated cue at turns 5/7/9/11/13)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cascade_dynamics(df, out_dir, checkpoints, stage_labels, title=""):
    """Two panels across turns 2–14: (A) mean accuracy, (B) P(still correct | T2 correct)."""
    if "turn6_raw_response" not in df.columns:
        return
    multi = df[df["turns"] == "multi"]
    turn_ns = [2, 4, 6, 8, 10, 12, 14]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for ci, ckpt in enumerate(checkpoints):
        sub = multi[multi["checkpoint"] == ckpt]
        accs, errs = [], []
        for n in turn_ns:
            col = f"turn{n}_is_correct"
            if col not in sub.columns:
                accs.append(np.nan); errs.append(0.0); continue
            vals = sub[col].dropna()
            k = len(vals)
            if k == 0:
                accs.append(np.nan); errs.append(0.0); continue
            mean = vals.mean()
            accs.append(mean); errs.append(_se(mean, k))
        ax.errorbar(turn_ns, accs, yerr=errs, marker="o", capsize=3,
                    label=stage_labels[ckpt], color=STAGE_COLORS[ci], linewidth=1.5)
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Mean accuracy")
    ax.set_xticks(turn_ns)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A — Accuracy across cascade turns")
    ax.legend(fontsize=8)

    ax = axes[1]
    for ci, ckpt in enumerate(checkpoints):
        sub = multi[multi["checkpoint"] == ckpt]
        eligible = sub[sub["turn2_is_correct"]]
        rates, errs = [], []
        for n in turn_ns:
            col = f"turn{n}_is_correct"
            if col not in eligible.columns:
                rates.append(np.nan); errs.append(0.0); continue
            vals = eligible[col].dropna()
            k = len(vals)
            if k == 0:
                rates.append(np.nan); errs.append(0.0); continue
            persist = vals.mean()
            rates.append(persist); errs.append(_se(persist, k))
        ax.errorbar(turn_ns, rates, yerr=errs, marker="o", capsize=3,
                    label=stage_labels[ckpt], color=STAGE_COLORS[ci], linewidth=1.5)
    ax.set_xlabel("Turn number")
    ax.set_ylabel("P(still correct | T2 correct)")
    ax.set_xticks(turn_ns)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("B — Conditional persistence")
    ax.legend(fontsize=8)

    suptitle = f"Cascade dynamics — {title}" if title else "Cascade dynamics"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "cascade_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  cascade_dynamics.png saved")


def plot_cascade_by_cue_strength(df, out_dir, checkpoints, stage_labels, title=""):
    """Conditional flip rate from T2 over turns 2–14, one line per cue strength,
    one subplot per checkpoint."""
    if "turn6_raw_response" not in df.columns:
        return
    multi = df[df["turns"] == "multi"]
    turn_ns = [2, 4, 6, 8, 10, 12, 14]
    strengths = list(STRENGTHS)
    cmap = plt.cm.YlOrRd(np.linspace(0.15, 0.95, len(strengths)))
    n_ck = len(checkpoints)

    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = multi[multi["checkpoint"] == ckpt]
        for si, s in enumerate(strengths):
            ss = sub[sub["cue_strength"] == s]
            eligible = ss[ss["turn2_is_correct"]]
            if len(eligible) == 0:
                continue
            rates = []
            for n in turn_ns:
                col = f"turn{n}_is_correct"
                if col not in eligible.columns:
                    rates.append(np.nan); continue
                vals = eligible[col].dropna()
                if len(vals) == 0:
                    rates.append(np.nan); continue
                rates.append(1.0 - vals.mean())
            ax.plot(turn_ns, rates, marker="o",
                    label=CUE_LABELS[s].replace("\n", " ") if ax_idx == 0 else None,
                    color=cmap[si], linewidth=1.2)
        ax.set_xticks(turn_ns)
        ax.set_xlabel("Turn number")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Conditional flip rate from T2")
            ax.legend(fontsize=6, loc="upper left", ncol=2)

    suptitle = (f"Cascade flip rate by cue strength — {title}" if title
                else "Cascade flip rate by cue strength")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "cascade_by_cue_strength.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  cascade_by_cue_strength.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 15b. Per-question cascade flip heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_question_cascade_heatmap(df, out_dir, checkpoints, stage_labels, title=""):
    """Heatmap of per-question conditional flip rate across cascade turns.

    Rows = questions (q01..q20), columns = turns {2,4,6,8,10,12,14}, one
    subplot per checkpoint. Cell value = 1 − P(turn N correct | turn 2 correct),
    aggregated across all multi-turn trials of that question (cues × framings).
    White = 0 (no flips), purple = 1 (all flipped). T2 column is white by
    construction.
    """
    if "turn6_is_correct" not in df.columns:
        return
    multi = df[df["turns"] == "multi"]
    turn_ns = [2, 4, 6, 8, 10, 12, 14]
    questions = sorted(multi["question_id"].unique())
    n_ck = len(checkpoints)

    purple_cmap = LinearSegmentedColormap.from_list(
        "white_purple", ["#ffffff", "#6a3d9a"], N=256
    )

    fig, axes = plt.subplots(1, n_ck, figsize=(3.4 * n_ck, 6), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = multi[multi["checkpoint"] == ckpt]
        mat = np.full((len(questions), len(turn_ns)), np.nan)
        for r, q in enumerate(questions):
            qs = sub[sub["question_id"] == q]
            eligible = qs[qs["turn2_is_correct"]]
            if len(eligible) == 0:
                continue
            for c, n in enumerate(turn_ns):
                col = f"turn{n}_is_correct"
                if col not in eligible.columns:
                    continue
                vals = eligible[col].dropna()
                if len(vals) == 0:
                    continue
                mat[r, c] = 1.0 - vals.mean()

        im = ax.imshow(mat, vmin=0, vmax=1, cmap=purple_cmap, aspect="auto")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                if np.isnan(mat[r, c]):
                    continue
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if mat[r, c] > 0.55 else "black")
        ax.set_xticks(range(len(turn_ns)))
        ax.set_xticklabels([f"T{n}" for n in turn_ns], fontsize=8)
        ax.set_yticks(range(len(questions)))
        if ax_idx == 0:
            ax.set_yticklabels(questions, fontsize=7)
            ax.set_ylabel("Question")
        ax.set_xlabel("Turn")
        ax.set_title(stage_labels[ckpt], fontsize=11)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="Conditional flip rate (1 − P(correct|T2 correct))")
    suptitle = (f"Per-question cascade flip rate — {title}" if title
                else "Per-question cascade flip rate")
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_dir / "question_cascade_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  question_cascade_heatmap.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 16. Cell summary CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_cell_summary(df, out_dir, stage_labels):
    group_cols = [
        "checkpoint", "condition", "turns", "framing",
        "question_id", "question_domain", "cue_id", "cue_strength", "cue_strength_label",
    ]
    agg = (
        df.groupby(group_cols)
        .agg(
            count=("turn4_correct", "size"),
            t4_accuracy=("turn4_is_correct", "mean"),
            t2_accuracy=("turn2_is_correct", "mean"),
            t4_flipped_to_suggested=("turn4_is_suggested_wrong",
                                     lambda s: s.mean() if len(s) > 0 else 0),
            t4_prob_correct_mean=("turn4_prob_correct", "mean"),
            t4_prob_wrong_mean=("turn4_prob_wrong", "mean"),
            t2_prob_correct_mean=("turn2_prob_correct", "mean"),
            prob_correct_shift_mean=("prob_correct_shift", "mean"),
        )
        .reset_index()
    )
    agg["checkpoint_label"] = agg["checkpoint"].map(stage_labels)
    agg.to_csv(out_dir / "cell_summary.csv", index=False)
    print(f"  cell_summary.csv saved: {len(agg)} rows")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze v5.0 sycophancy MMLU pilot.")
    parser.add_argument("--model_family", type=str, default="olmo31-32b-instruct",
                        choices=list(MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation.")
    args = parser.parse_args()

    family = MODEL_FAMILIES[args.model_family]
    checkpoints = family["checkpoints"]
    stage_labels = family["stage_labels"]
    family_title = family["title"]

    default_out = family["default_output_dir"]
    if args.run_label:
        default_out = f"{default_out}_{args.run_label}"
    out_dir = REPO_ROOT / (args.output_dir or default_out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model family: {family_title}")
    if args.run_label:
        print(f"Run label: {args.run_label}")
    df = load_results(family, seed=args.seed, run_label=args.run_label)
    print(f"  Loaded {len(df)} rows across {df['checkpoint'].nunique()} checkpoints "
          f"and {df['condition'].nunique()} conditions")

    report_data_quality(df, out_dir, checkpoints, stage_labels)
    report_baseline_accuracy(df, out_dir, checkpoints, stage_labels)
    plot_headline_22(df, out_dir, checkpoints, stage_labels, family_title)
    plot_flip_rate_by_cue_strength(df, out_dir, checkpoints, stage_labels, family_title)
    plot_flip_rate_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    plot_flip_to_suggested(df, out_dir, checkpoints, stage_labels, family_title)
    plot_marginals(df, out_dir, checkpoints, stage_labels, family_title)
    plot_multi_metrics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_training_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_per_question(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_probs(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_vs_generated(df, out_dir, checkpoints, stage_labels, family_title)
    plot_interactions(df, out_dir, checkpoints, stage_labels, family_title)
    plot_per_condition_breakdowns(df, out_dir, checkpoints, stage_labels, family_title)
    plot_cascade_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_cascade_by_cue_strength(df, out_dir, checkpoints, stage_labels, family_title)
    plot_question_cascade_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
