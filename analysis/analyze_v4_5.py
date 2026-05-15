#!/usr/bin/env python3
"""
Analyze v4.5 — 2×2×2 factorial (turns × role × framing).

Loads all 8 condition CSVs into one long DataFrame (with turns, role, framing
columns), then produces:

  1. Data quality: null rates per (checkpoint, condition); flag > 20%
  2. Cooperate baseline by condition
  3. Headline 2×2×2 maintain rate panel (one subplot per checkpoint)
  4. Marginal effects: maintain rate by turns / role / framing
  5. Interaction plots: framing × role (one row per turns), etc.
  6. Per-locus heatmap (4 epistemic loci × 8 conditions, one panel per checkpoint)
  7. Sycophancy across the 4 unique role-by-turns cells
  8. Multi-turn-only metrics: conditional flip rate, Δprob_truth
  9. Logit probabilities by condition
 10. cell_summary.csv per (condition × cell_id)

Usage:
  python analysis/analyze_v4_5.py --model_family olmo31-32b-instruct
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_FAMILIES = {
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct (v4.5)",
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
        "default_output_dir": "analysis_outputs/v4_5_olmo31_32b",
    },
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v4.5)",
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
        "default_output_dir": "analysis_outputs/v4_5_olmo3_7b",
    },
    "olmo3-7b-think": {
        "title": "OLMo-3 7B Think (v4.5)",
        "checkpoints": [
            "OLMo-3-7B-Think-SFT",
            "OLMo-3-7B-Think-DPO",
            "OLMo-3-7B-Think-RLVR-step1375",
        ],
        "stage_labels": {
            "OLMo-3-7B-Think-SFT": "SFT",
            "OLMo-3-7B-Think-DPO": "DPO",
            "OLMo-3-7B-Think-RLVR-step1375": "RLVR",
        },
        "results_family": "olmo3_think",
        "default_output_dir": "analysis_outputs/v4_5_olmo3_think",
    },
    # Gemini via Vertex AI — single model, so checkpoints list has one entry.
    # Use --model_key to control the string used in both dir names and CSV.
    "gemini": {
        "title": "Gemini (v4.5)",
        "checkpoints": ["gemini"],
        "stage_labels": {"gemini": "Gemini"},
        "results_family": "gemini",
        "default_output_dir": "analysis_outputs/v4_5_gemini",
    },
}

CONDITIONS = [
    "v4_5_single_sys_quote",
    "v4_5_single_sys_report",
    "v4_5_single_uo_quote",
    "v4_5_single_uo_report",
    "v4_5_multi_sys_quote",
    "v4_5_multi_sys_report",
    "v4_5_multi_uo_quote",
    "v4_5_multi_uo_report",
]

# pretty short labels
COND_SHORT = {
    "v4_5_single_sys_quote":  "S·sys·quote",
    "v4_5_single_sys_report": "S·sys·report",
    "v4_5_single_uo_quote":   "S·uo·quote",
    "v4_5_single_uo_report":  "S·uo·report",
    "v4_5_multi_sys_quote":   "M·sys·quote",
    "v4_5_multi_sys_report":  "M·sys·report",
    "v4_5_multi_uo_quote":    "M·uo·quote",
    "v4_5_multi_uo_report":   "M·uo·report",
}

LOCUS_ORDER = ["source-capacity", "source-motivation", "seeker-capacity", "environmental"]
LOCUS_SHORT = {
    "source-capacity": "Src cap", "source-motivation": "Src motiv",
    "seeker-capacity": "Skr cap", "environmental": "Environ",
    "sycophancy": "Sycoph", "no-cue": "No cue",
}
LOCUS_LABELS = ["Source\ncapacity", "Source\nmotivation", "Seeker\ncapacity", "Environmental"]
LOCUS_COLORS = {
    "source-capacity": "#1f77b4", "source-motivation": "#d62728",
    "seeker-capacity": "#e377c2", "environmental": "#17becf",
}
SPEC_ORDER = ["direct", "statistical", "mechanism", "inferential", "behavioral"]
PRIOR_ORDER = ["none", "stale-time", "stale-quality", "fresh"]
PRIOR_LABELS = ["None", "Stale\n(time)", "Stale\n(quality)", "Fresh"]
DOSE_ORDER = ["weak", "medium", "strong"]

CHOICE_COLORS = {"truth": "#2ca02c", "other": "#d62728", "unsure": "#7f7f7f"}
ROLE_COLORS = {"sys": "#1f77b4", "uo": "#ff7f0e"}
FRAMING_COLORS = {"quote": "#9467bd", "report": "#8c564b"}
TURNS_COLORS = {"single": "#17becf", "multi": "#e377c2"}


def _condition_dir(family_results, condition, run_label=""):
    suffix = condition[len("v4_5_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v4_5_{suffix}{label_suffix}"


def load_results(family, seed=0, run_label=""):
    """Load all 8 conditions into one DataFrame."""
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
        raise FileNotFoundError("No v4.5 result CSVs found")
    df = pd.concat(frames, ignore_index=True)

    for col in ["turn2_prob_A", "turn2_prob_B", "turn2_prob_C",
                "turn4_prob_A", "turn4_prob_B", "turn4_prob_C"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for turn in ["turn2", "turn4"]:
        df[f"{turn}_prob_truth"] = np.where(
            df["truth_position"] == "A", df[f"{turn}_prob_A"], df[f"{turn}_prob_B"])
        df[f"{turn}_prob_other"] = np.where(
            df["truth_position"] == "A", df[f"{turn}_prob_B"], df[f"{turn}_prob_A"])
        df[f"{turn}_prob_unsure"] = df[f"{turn}_prob_C"]

    df["prob_truth_shift"] = df["turn4_prob_truth"] - df["turn2_prob_truth"]
    return df


def _rates(sub, turn="turn4"):
    n = len(sub)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    col = f"{turn}_choice"
    truth = (sub[col] == "truth").sum() / n
    other = (sub[col] == "other").sum() / n
    unsure = (sub[col] == "unsure").sum() / n
    return truth, other, unsure, n


def _maintain_se(sub, turn="turn4"):
    n = len(sub)
    if n < 2:
        return 0.0
    p = (sub[f"{turn}_choice"] == "truth").sum() / n
    return np.sqrt(p * (1 - p) / n)


def _adv_cued(df):
    """Adversarial trials with the 4 epistemic cue loci (excludes sycophancy & no-cue)."""
    return df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]


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

    # cooperate baseline by condition (turn4 only)
    coop = df[df["goal_condition"] == "cooperate"]
    if len(coop) > 0:
        coop_rows = []
        for ckpt in checkpoints:
            for cond in CONDITIONS:
                sub = coop[(coop["checkpoint"] == ckpt) & (coop["condition"] == cond)]
                t, o, u, n = _rates(sub, "turn4")
                coop_rows.append({
                    "checkpoint": stage_labels[ckpt], "condition": cond,
                    "truth": f"{t:.3f}", "other": f"{o:.3f}", "unsure": f"{u:.3f}", "n": n,
                })
        pd.DataFrame(coop_rows).to_csv(out_dir / "cooperate_baseline.csv", index=False)
        print("  null_rates.csv + cooperate_baseline.csv saved")
    else:
        print("  null_rates.csv saved (no cooperate data)")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Headline: 2×2×2 maintain rate panel (one subplot per checkpoint)
# ═════════════════════════════════════════════════════════════════════════════

def plot_headline_222(df, out_dir, checkpoints, stage_labels, title=""):
    """One row of 4 checkpoint subplots. Each subplot has 8 bars in canonical order.
    Color = role; hatch = framing; gap between single/multi blocks."""
    adv = _adv_cued(df)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        vals, errs, labels, colors, hatches = [], [], [], [], []
        for i, cond in enumerate(CONDITIONS):
            sub = adv[(adv["checkpoint"] == ckpt) & (adv["condition"] == cond)]
            t, _, _, _ = _rates(sub, "turn4")
            vals.append(t)
            errs.append(_maintain_se(sub, "turn4"))
            labels.append(COND_SHORT[cond])
            parts = cond.split("_")  # v4_5_{turns}_{role}_{framing}
            role = parts[3]
            framing = parts[4]
            colors.append(ROLE_COLORS[role])
            hatches.append("//" if framing == "quote" else "")

        # x positions with a small gap between single (i=0..3) and multi (i=4..7)
        xs = list(range(4)) + [5, 6, 7, 8]
        bars = ax.bar(xs, vals, yerr=errs, color=colors, capsize=3,
                       edgecolor="black", linewidth=0.5)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axvline(4.5, color="gray", linestyle=":", alpha=0.4)
        ax.text(1.5, 1.02, "single", ha="center", fontsize=9, color="gray")
        ax.text(6.5, 1.02, "multi", ha="center", fontsize=9, color="gray")
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Maintain rate (final answer)")

    # custom legend for role + framing
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=ROLE_COLORS["sys"], edgecolor="black", label="role = sys"),
        Patch(facecolor=ROLE_COLORS["uo"], edgecolor="black", label="role = uo"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="framing = quote"),
        Patch(facecolor="white", edgecolor="black", label="framing = report"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    suptitle = f"Headline maintain rate — {title}" if title else "Headline maintain rate"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(out_dir / "headline_222.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  headline_222.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Marginal effects (collapse the other two axes)
# ═════════════════════════════════════════════════════════════════════════════

def plot_marginals(df, out_dir, checkpoints, stage_labels, title=""):
    adv = _adv_cued(df)
    axes_meta = [
        ("turns", ["single", "multi"], TURNS_COLORS, "Turns"),
        ("role",  ["sys", "uo"],       ROLE_COLORS,   "Role"),
        ("framing", ["quote", "report"], FRAMING_COLORS, "Framing"),
    ]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(3, n_ck, figsize=(3.2 * n_ck, 9), sharey=True)
    if n_ck == 1:
        axes = axes.reshape(3, 1)

    for row_idx, (col, levels, color_map, title_label) in enumerate(axes_meta):
        for col_idx, ckpt in enumerate(checkpoints):
            ax = axes[row_idx, col_idx]
            sub = adv[adv["checkpoint"] == ckpt]
            x = np.arange(len(levels))
            vals, errs = [], []
            for lv in levels:
                ssub = sub[sub[col] == lv]
                t, _, _, _ = _rates(ssub, "turn4")
                vals.append(t)
                errs.append(_maintain_se(ssub, "turn4"))
            colors = [color_map[lv] for lv in levels]
            ax.bar(x, vals, yerr=errs, color=colors, capsize=3,
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, fontsize=9)
            ax.set_ylim(0, 1.05)
            if col_idx == 0:
                ax.set_ylabel(f"{title_label}\nMaintain rate")
            if row_idx == 0:
                ax.set_title(stage_labels[ckpt], fontsize=11)

    suptitle = f"Marginal effects — {title}" if title else "Marginal effects"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  marginals.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Interactions (2x2 plots for each pair of axes)
# ═════════════════════════════════════════════════════════════════════════════

def _plot_interaction(df, out_dir, checkpoints, stage_labels, title,
                      x_col, x_levels, hue_col, hue_levels, hue_colors,
                      filename, label_prefix):
    """Lines: x = x_levels, color = hue_levels. One subplot per checkpoint,
    averaged over the third axis."""
    adv = _adv_cued(df)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4 * n_ck, 4.5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub_ck = adv[adv["checkpoint"] == ckpt]
        for hue in hue_levels:
            ys, errs = [], []
            for xv in x_levels:
                ss = sub_ck[(sub_ck[x_col] == xv) & (sub_ck[hue_col] == hue)]
                t, _, _, _ = _rates(ss, "turn4")
                ys.append(t)
                errs.append(_maintain_se(ss, "turn4"))
            ax.errorbar(range(len(x_levels)), ys, yerr=errs, marker="o",
                        capsize=3, color=hue_colors[hue],
                        label=f"{hue_col}={hue}" if ax_idx == 0 else None,
                        linewidth=1.5)
        ax.set_xticks(range(len(x_levels)))
        ax.set_xticklabels(x_levels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Maintain rate")
            ax.set_xlabel(x_col)
            ax.legend(fontsize=8)
    suptitle = f"{label_prefix} interaction — {title}" if title else f"{label_prefix} interaction"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_interactions(df, out_dir, checkpoints, stage_labels, title=""):
    _plot_interaction(df, out_dir, checkpoints, stage_labels, title,
                      x_col="framing", x_levels=["quote", "report"],
                      hue_col="role", hue_levels=["sys", "uo"], hue_colors=ROLE_COLORS,
                      filename="interaction_framing_x_role.png",
                      label_prefix="framing × role")
    _plot_interaction(df, out_dir, checkpoints, stage_labels, title,
                      x_col="turns", x_levels=["single", "multi"],
                      hue_col="framing", hue_levels=["quote", "report"], hue_colors=FRAMING_COLORS,
                      filename="interaction_turns_x_framing.png",
                      label_prefix="turns × framing")
    _plot_interaction(df, out_dir, checkpoints, stage_labels, title,
                      x_col="turns", x_levels=["single", "multi"],
                      hue_col="role", hue_levels=["sys", "uo"], hue_colors=ROLE_COLORS,
                      filename="interaction_turns_x_role.png",
                      label_prefix="turns × role")
    print("  3 interaction plots saved")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Per-locus heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_locus_heatmap(df, out_dir, checkpoints, stage_labels, title=""):
    """Heatmap: rows = locus, cols = condition, value = maintain rate.
    One subplot per checkpoint."""
    adv = _adv_cued(df)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 4.5))
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        mat = np.zeros((len(LOCUS_ORDER), len(CONDITIONS)))
        for r, loc in enumerate(LOCUS_ORDER):
            for c, cond in enumerate(CONDITIONS):
                sub = adv[(adv["checkpoint"] == ckpt) & (adv["condition"] == cond)
                          & (adv["locus"] == loc)]
                t, _, _, _ = _rates(sub, "turn4")
                mat[r, c] = t
        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([COND_SHORT[c] for c in CONDITIONS],
                           fontsize=7, rotation=35, ha="right")
        ax.set_yticks(range(len(LOCUS_ORDER)))
        ax.set_yticklabels([LOCUS_SHORT[l] for l in LOCUS_ORDER], fontsize=8)
        ax.set_title(stage_labels[ckpt], fontsize=11)

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Maintain rate")
    suptitle = f"Maintain rate by locus × condition — {title}" if title else "Maintain rate by locus × condition"
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_dir / "locus_x_condition_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  locus_x_condition_heatmap.png saved")


def plot_locus_specificity_heatmap(df, out_dir, checkpoints, stage_labels, title=""):
    """3D heatmap: rows = checkpoints, cols = specificities.
    Each subplot is a 4-locus × 8-condition heatmap."""
    adv = _adv_cued(df)
    specs = [s for s in SPEC_ORDER if s in adv["specificity"].unique()]
    if not specs:
        print("  skipping locus_x_condition_x_specificity_heatmap (no matching specificities)")
        return
    spec_labels = {s: s.capitalize() for s in SPEC_ORDER}

    n_ck = len(checkpoints)
    n_sp = len(specs)
    cbar_space = 0.06
    fig, axes = plt.subplots(n_ck, n_sp,
                             figsize=(4.5 * n_sp + 1.0, 3.5 * n_ck),
                             squeeze=False)

    for row, ckpt in enumerate(checkpoints):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            mat = np.full((len(LOCUS_ORDER), len(CONDITIONS)), np.nan)
            for r, loc in enumerate(LOCUS_ORDER):
                for c, cond in enumerate(CONDITIONS):
                    sub = adv[(adv["checkpoint"] == ckpt) & (adv["condition"] == cond)
                              & (adv["locus"] == loc) & (adv["specificity"] == spec)]
                    if len(sub) > 0:
                        t, _, _, _ = _rates(sub, "turn4")
                        mat[r, c] = t
            im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    if not np.isnan(mat[r, c]):
                        ax.text(c, r, f"{mat[r,c]:.2f}", ha="center", va="center",
                                fontsize=6, color="black")
            if row == n_ck - 1:
                ax.set_xticks(range(len(CONDITIONS)))
                ax.set_xticklabels([COND_SHORT[c] for c in CONDITIONS],
                                   fontsize=6, rotation=35, ha="right")
            else:
                ax.set_xticks([])
            ax.set_yticks(range(len(LOCUS_ORDER)))
            ax.set_yticklabels([LOCUS_SHORT[l] for l in LOCUS_ORDER], fontsize=7)
            if row == 0:
                ax.set_title(spec_labels[spec], fontsize=10)
            if col == n_sp - 1:
                ax.annotate(stage_labels[ckpt], xy=(1.02, 0.5),
                            xycoords="axes fraction", fontsize=10,
                            ha="left", va="center", rotation=270)

    suptitle = (f"Maintain rate by locus × condition × specificity — {title}"
                if title else "Maintain rate by locus × condition × specificity")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1 - cbar_space, 0.96])
    cbar_ax = fig.add_axes([1 - cbar_space + 0.01, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Maintain rate")
    fig.savefig(out_dir / "locus_x_condition_x_specificity_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  locus_x_condition_x_specificity_heatmap.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Sycophancy across conditions
# ═════════════════════════════════════════════════════════════════════════════

def plot_sycophancy(df, out_dir, checkpoints, stage_labels, title=""):
    """Sycophancy maintain rate by 4 unique cells (turns × role).
    Quote/report are degenerate; collapse them."""
    syc = df[df["locus"] == "sycophancy"]
    cells = [
        ("single", "sys", "single·sys"),
        ("single", "uo",  "single·uo"),
        ("multi",  "sys", "multi·sys"),
        ("multi",  "uo",  "multi·uo"),
    ]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(3.2 * n_ck, 4.5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        vals, errs, labels = [], [], []
        for turns_, role_, lab in cells:
            sub = syc[(syc["checkpoint"] == ckpt) & (syc["turns"] == turns_)
                      & (syc["role"] == role_)]
            t, _, _, _ = _rates(sub, "turn4")
            vals.append(t)
            errs.append(_maintain_se(sub, "turn4"))
            labels.append(lab)
        x = np.arange(len(cells))
        colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Sycophancy maintain rate")
    suptitle = f"Sycophancy maintain rate (no Mia, no cue) — {title}" if title else "Sycophancy maintain rate"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "sycophancy_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  sycophancy_by_condition.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Multi-turn metrics: conditional flip rate, Δprob_truth
# ═════════════════════════════════════════════════════════════════════════════

def _conditional_flip(sub):
    """Among trials where turn2_choice == 'truth', fraction where turn4_choice != 'truth'."""
    eligible = sub[sub["turn2_choice"] == "truth"]
    n = len(eligible)
    if n == 0:
        return 0.0, 0
    flips = (eligible["turn4_choice"] != "truth").sum()
    return flips / n, n


def plot_multi_metrics(df, out_dir, checkpoints, stage_labels, title=""):
    """For the 4 multi-turn conditions: conditional flip rate + Δprob_truth, by role × framing."""
    multi = df[(df["turns"] == "multi") & (df["goal_condition"] == "adversarial")
               & (df["locus"].isin(LOCUS_ORDER))]
    cells = [
        ("sys", "quote", "sys·quote", ROLE_COLORS["sys"], "//"),
        ("sys", "report", "sys·report", ROLE_COLORS["sys"], ""),
        ("uo",  "quote", "uo·quote", ROLE_COLORS["uo"], "//"),
        ("uo",  "report", "uo·report", ROLE_COLORS["uo"], ""),
    ]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(2, n_ck, figsize=(3.2 * n_ck, 8), sharey="row")
    if n_ck == 1:
        axes = axes.reshape(2, 1)

    for col_idx, ckpt in enumerate(checkpoints):
        sub_ck = multi[multi["checkpoint"] == ckpt]
        # row 0: conditional flip rate
        ax = axes[0, col_idx]
        flips, errs, labels, colors, hatches = [], [], [], [], []
        for role_, fram_, lab, color, hatch in cells:
            ss = sub_ck[(sub_ck["role"] == role_) & (sub_ck["framing"] == fram_)]
            f, n = _conditional_flip(ss)
            flips.append(f)
            errs.append(np.sqrt(f * (1 - f) / n) if n > 0 else 0)
            labels.append(lab); colors.append(color); hatches.append(hatch)
        x = np.arange(len(cells))
        bars = ax.bar(x, flips, yerr=errs, color=colors, capsize=3,
                      edgecolor="black", linewidth=0.5)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Cond flip rate\n(turn2=truth → turn4≠truth)")

        # row 1: Δprob_truth
        ax = axes[1, col_idx]
        means, sems, labels = [], [], []
        for role_, fram_, lab, _, _ in cells:
            ss = sub_ck[(sub_ck["role"] == role_) & (sub_ck["framing"] == fram_)]
            shift = ss["prob_truth_shift"].dropna()
            means.append(shift.mean() if len(shift) else 0)
            sems.append(shift.sem() if len(shift) > 1 else 0)
            labels.append(lab)
        bars = ax.bar(x, means, yerr=sems, color=colors, capsize=3,
                      edgecolor="black", linewidth=0.5)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        if col_idx == 0:
            ax.set_ylabel("Δ P(truth)\n(turn4 − turn2)")

    suptitle = f"Multi-turn belief-change metrics — {title}" if title else "Multi-turn belief-change metrics"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_turn_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  multi_turn_metrics.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Logit probabilities by condition
# ═════════════════════════════════════════════════════════════════════════════

def plot_logit_probs(df, out_dir, checkpoints, stage_labels, title=""):
    adv = _adv_cued(df)
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.5 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        x = np.arange(len(CONDITIONS))
        width = 0.27
        truth_means, other_means, unsure_means = [], [], []
        truth_sems, other_sems, unsure_sems = [], [], []
        for cond in CONDITIONS:
            sub = adv[(adv["checkpoint"] == ckpt) & (adv["condition"] == cond)]
            for arr_m, arr_s, col in [
                (truth_means, truth_sems, "turn4_prob_truth"),
                (other_means, other_sems, "turn4_prob_other"),
                (unsure_means, unsure_sems, "turn4_prob_unsure"),
            ]:
                vals = sub[col].dropna()
                arr_m.append(vals.mean() if len(vals) else 0)
                arr_s.append(vals.sem() if len(vals) > 1 else 0)
        ax.bar(x - width, truth_means, width, yerr=truth_sems,
               color=CHOICE_COLORS["truth"], capsize=2,
               label="P(truth)" if ax_idx == 0 else None)
        ax.bar(x, other_means, width, yerr=other_sems,
               color=CHOICE_COLORS["other"], capsize=2,
               label="P(other)" if ax_idx == 0 else None)
        ax.bar(x + width, unsure_means, width, yerr=unsure_sems,
               color=CHOICE_COLORS["unsure"], capsize=2,
               label="P(unsure)" if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_SHORT[c] for c in CONDITIONS],
                           fontsize=7, rotation=35, ha="right")
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Mean final-answer logit probability")
            ax.legend(fontsize=8)

    suptitle = f"Logit probabilities by condition (adversarial cued) — {title}" if title else "Logit probabilities by condition"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_probs_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  logit_probs_by_condition.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 8b. Sanity check: does argmax(P(A),P(B),P(C)) match the parsed letter?
# ═════════════════════════════════════════════════════════════════════════════

def _add_logit_pred(df):
    """Add columns 'turn4_logit_letter' and 'turn4_logit_choice' derived from
    argmax over P(A), P(B), P(C). Skips rows where any of the three is NaN."""
    pa, pb, pc = df["turn4_prob_A"].values, df["turn4_prob_B"].values, df["turn4_prob_C"].values
    probs = np.stack([pa, pb, pc], axis=1)
    valid = ~np.isnan(probs).any(axis=1)
    letters = np.array(["A", "B", "C"])
    out_letter = np.full(len(df), "", dtype=object)
    if valid.any():
        out_letter[valid] = letters[np.argmax(probs[valid], axis=1)]
    truth_pos = df["truth_position"].astype(str).values
    out_choice = np.where(out_letter == "", "",
                  np.where(out_letter == "C", "unsure",
                  np.where(out_letter == truth_pos, "truth", "other")))
    df = df.copy()
    df["turn4_logit_letter"] = out_letter
    df["turn4_logit_choice"] = out_choice
    return df


def plot_logit_vs_generated_maintain(df, out_dir, checkpoints, stage_labels, title=""):
    """Side-by-side bars per condition: maintain rate from generation vs from
    argmax(P(A),P(B),P(C)). If they diverge, generation is sampling tokens that
    don't match the immediate-next-token argmax."""
    df = _add_logit_pred(df)
    adv = df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.5 * n_ck, 5), sharey=True)
    if n_ck == 1:
        axes = [axes]
    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        gen_vals, logit_vals, labels = [], [], []
        for cond in CONDITIONS:
            sub = adv[(adv["checkpoint"] == ckpt) & (adv["condition"] == cond)]
            n_gen = (sub["turn4_choice"].isin(["truth", "other", "unsure"])).sum()
            n_log = (sub["turn4_logit_choice"].isin(["truth", "other", "unsure"])).sum()
            gen_vals.append((sub["turn4_choice"] == "truth").sum() / n_gen if n_gen else 0)
            logit_vals.append((sub["turn4_logit_choice"] == "truth").sum() / n_log if n_log else 0)
            labels.append(COND_SHORT[cond])
        x = np.arange(len(CONDITIONS))
        width = 0.4
        ax.bar(x - width/2, gen_vals,   width, color=CHOICE_COLORS["truth"],
               label="Generated → parsed" if ax_idx == 0 else None)
        ax.bar(x + width/2, logit_vals, width, color="#ff7f0e",
               label="argmax(P(A),P(B),P(C))" if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Maintain rate (final answer)")
            ax.legend(fontsize=8, loc="upper right")
    suptitle = (f"Maintain rate: generated vs logit-argmax — {title}" if title
                else "Maintain rate: generated vs logit-argmax")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_vs_generated_maintain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  logit_vs_generated_maintain.png saved")


def plot_logit_vs_generated_confusion(df, out_dir, checkpoints, stage_labels, title=""):
    """3×3 confusion matrix per checkpoint: row = argmax(P(A),P(B),P(C)) → choice,
    column = generated/parsed choice. Rows are normalized so each row sums to 1.
    Diagonal mass = how often the immediate-next-token argmax predicts what the
    model actually generated."""
    df = _add_logit_pred(df)
    adv = df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]
    cats = ["truth", "other", "unsure"]
    n_ck = len(checkpoints)
    fig, axes = plt.subplots(1, n_ck, figsize=(4.2 * n_ck, 4.4))
    if n_ck == 1:
        axes = [axes]
    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = adv[(adv["checkpoint"] == ckpt)
                  & adv["turn4_logit_choice"].isin(cats)
                  & adv["turn4_choice"].isin(cats)]
        mat = np.zeros((3, 3), dtype=int)
        for i, lc in enumerate(cats):
            for j, gc in enumerate(cats):
                mat[i, j] = ((sub["turn4_logit_choice"] == lc) & (sub["turn4_choice"] == gc)).sum()
        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mat_norm = np.where(row_sums > 0, mat / row_sums, 0.0)
        agreement = float(np.diag(mat).sum() / mat.sum()) if mat.sum() > 0 else float("nan")
        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(cats); ax.set_yticklabels(cats)
        ax.set_xlabel("Generated → parsed choice")
        if ax_idx == 0:
            ax.set_ylabel("Logit-argmax choice")
        ax.set_title(f"{stage_labels[ckpt]}  (agreement: {agreement:.1%})", fontsize=11)
        for i in range(3):
            for j in range(3):
                txt = f"{mat_norm[i, j]:.2f}\n({int(mat[i, j])})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if mat_norm[i, j] > 0.5 else "black")
    suptitle = (f"Logit-argmax vs generated choice (row-normalized) — {title}" if title
                else "Logit-argmax vs generated choice (row-normalized)")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "logit_vs_generated_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  logit_vs_generated_confusion.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Per-condition breakdown plots (locus / specificity / prior / dosage)
#    Mirrors the classic v4_3choice breakdown suite; one subfolder per condition.
# ═════════════════════════════════════════════════════════════════════════════

def _plot_choice_by_stage(df, out_dir, checkpoints, stage_labels, title=""):
    adversarial = df[(df["goal_condition"] == "adversarial") & (df["locus"] != "no-cue")]
    cooperate   = df[df["goal_condition"] == "cooperate"]

    panels = [("Adversarial (with cues)", adversarial)]
    if len(cooperate) > 0:
        panels.append(("Cooperate", cooperate))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax_idx, (cond_label, sub_df) in enumerate(panels):
        ax = axes[ax_idx]
        labels, maintain_vals, defer_vals, unsure_vals = [], [], [], []
        for ckpt in checkpoints:
            sub = sub_df[sub_df["checkpoint"] == ckpt]
            t, d, u, _ = _rates(sub, "turn4")
            labels.append(stage_labels[ckpt])
            maintain_vals.append(t); defer_vals.append(d); unsure_vals.append(u)
        x = np.arange(len(labels))
        ax.bar(x, maintain_vals, label="Maintain (truth)", color=CHOICE_COLORS["truth"])
        ax.bar(x, defer_vals, bottom=maintain_vals, label="Defer (other)", color=CHOICE_COLORS["other"])
        ax.bar(x, unsure_vals,
               bottom=[m + d for m, d in zip(maintain_vals, defer_vals)],
               label="Unsure", color=CHOICE_COLORS["unsure"])
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Proportion"); ax.set_ylim(0, 1.05); ax.set_title(cond_label)
        ax.legend(fontsize=8, loc="upper right")
    suptitle = f"Choice distribution by training stage — {title}" if title else "Choice distribution by training stage"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "choice_by_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_choice_by_locus(df, out_dir, checkpoints, stage_labels, title=""):
    adversarial = df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]
    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes_flat = [axes] if len(panels) == 1 else axes.flatten()
    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = adversarial if ckpt == "all" else adversarial[adversarial["checkpoint"] == ckpt]
        maintain_vals, defer_vals, unsure_vals = [], [], []
        for locus in LOCUS_ORDER:
            t, d, u, _ = _rates(sub[sub["locus"] == locus], "turn4")
            maintain_vals.append(t); defer_vals.append(d); unsure_vals.append(u)
        x = np.arange(len(LOCUS_ORDER))
        ax.bar(x, maintain_vals, color=CHOICE_COLORS["truth"],   label="Maintain" if ax_idx == 0 else None)
        ax.bar(x, defer_vals,    bottom=maintain_vals,            color=CHOICE_COLORS["other"],   label="Defer"    if ax_idx == 0 else None)
        ax.bar(x, unsure_vals,   bottom=[m + d for m, d in zip(maintain_vals, defer_vals)],
               color=CHOICE_COLORS["unsure"], label="Unsure" if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([LOCUS_SHORT[l] for l in LOCUS_ORDER], fontsize=8, rotation=20, ha="right")
        ax.set_title(label, fontsize=11); ax.set_ylim(0, 1.05)
        if ax_idx % ncols == 0:
            ax.set_ylabel("Proportion")
    for i in range(len(panels), len(axes_flat)):
        axes_flat[i].set_visible(False)
    axes_flat[0].legend(fontsize=8)
    suptitle = f"Choice distribution by locus (adversarial) — {title}" if title else "Choice distribution by locus (adversarial)"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "choice_by_locus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_a(df, out_dir, checkpoints, stage_labels, title=""):
    """Maintain rate by locus × specificity (slice A = fresh-prior cells)."""
    slice_a = df[(df["slice"] == "A") & (df["locus"].isin(LOCUS_ORDER))]
    no_cue  = df[(df["slice"] == "A") & (df["locus"] == "no-cue")]
    n_loci = len(LOCUS_ORDER)
    width = 0.18
    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()
    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = slice_a if ckpt == "all" else slice_a[slice_a["checkpoint"] == ckpt]
        nc  = no_cue  if ckpt == "all" else no_cue[no_cue["checkpoint"] == ckpt]
        nc_maintain, _, _, _ = _rates(nc, "turn4")
        x = np.arange(len(SPEC_ORDER))
        for i, locus in enumerate(LOCUS_ORDER):
            locus_data = sub[sub["locus"] == locus]
            vals, errs = [], []
            for spec in SPEC_ORDER:
                sd = locus_data[locus_data["specificity"] == spec]
                m, _, _, _ = _rates(sd, "turn4")
                vals.append(m); errs.append(_maintain_se(sd, "turn4"))
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=LOCUS_SHORT[locus] if ax_idx == 0 else None,
                   color=LOCUS_COLORS[locus], capsize=2)
        ax.axhline(y=nc_maintain, color="gray", linestyle="--", alpha=0.6,
                   label="No cue" if ax_idx == 0 else None)
        ax.set_xticks(x + width * (n_loci - 1) / 2)
        ax.set_xticklabels([s.capitalize() for s in SPEC_ORDER], fontsize=8, rotation=30, ha="right")
        ax.set_title(label, fontsize=11); ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")
    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="lower left")
    suptitle = f"Slice A: Maintain rate by specificity × locus — {title}" if title else "Slice A: Maintain rate by specificity × locus"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_A_maintain_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_b(df, out_dir, checkpoints, stage_labels, title=""):
    """Maintain rate by locus × prior (inferential cues across slices A+B)."""
    slice_b = df[
        (df["slice"].isin(["A", "B"])) &
        (
            ((df["locus"] != "no-cue") & (df["specificity"] == "inferential")) |
            (df["locus"] == "no-cue")
        ) &
        (df["goal_condition"] == "adversarial")
    ]
    all_loci = LOCUS_ORDER + ["no-cue"]
    locus_colors = {**LOCUS_COLORS, "no-cue": "#7f7f7f"}
    locus_labels = {**LOCUS_SHORT, "no-cue": "No cue"}
    n_loci = len(all_loci)
    width = 0.15
    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()
    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = slice_b if ckpt == "all" else slice_b[slice_b["checkpoint"] == ckpt]
        x = np.arange(len(PRIOR_ORDER))
        for i, locus in enumerate(all_loci):
            locus_data = sub[sub["locus"] == locus]
            vals, errs = [], []
            for prior in PRIOR_ORDER:
                pd_ = locus_data[locus_data["prior"] == prior]
                m, _, _, _ = _rates(pd_, "turn4")
                vals.append(m); errs.append(_maintain_se(pd_, "turn4"))
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=locus_labels[locus] if ax_idx == 0 else None,
                   color=locus_colors[locus], capsize=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_xticks(x + width * (n_loci - 1) / 2)
        ax.set_xticklabels(PRIOR_LABELS, fontsize=8)
        ax.set_title(label, fontsize=11); ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")
    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="lower left")
    suptitle = f"Slice B: Maintain rate by prior × locus — {title}" if title else "Slice B: Maintain rate by prior × locus"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_B_maintain_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slice_c(df, out_dir, checkpoints, stage_labels, title=""):
    """Dose-response: maintain rate vs dosage (statistical cues, fresh prior)."""
    slice_c = df[
        (df["slice"].isin(["A", "C"])) &
        (df["specificity"] == "statistical") &
        (df["locus"].isin(LOCUS_ORDER)) &
        (df["prior"] == "fresh")
    ]
    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()
    dose_x = {d: i for i, d in enumerate(DOSE_ORDER)}
    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = slice_c if ckpt == "all" else slice_c[slice_c["checkpoint"] == ckpt]
        for locus in LOCUS_ORDER:
            locus_data = sub[sub["locus"] == locus]
            means, xs = [], []
            for dose in DOSE_ORDER:
                m, _, _, n = _rates(locus_data[locus_data["dosage"] == dose], "turn4")
                if n > 0:
                    means.append(m); xs.append(dose_x[dose])
            if xs:
                ax.plot(xs, means, marker="o",
                        label=LOCUS_SHORT[locus] if ax_idx == 0 else None,
                        color=LOCUS_COLORS[locus], linewidth=1.5)
        ax.set_xticks(range(len(DOSE_ORDER)))
        ax.set_xticklabels([d.capitalize() for d in DOSE_ORDER])
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(label, fontsize=11); ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")
    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="best")
    suptitle = f"Slice C: Dose-response maintain rate — {title}" if title else "Slice C: Dose-response maintain rate"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_C_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_training_dynamics(df, out_dir, checkpoints, stage_labels, title=""):
    """Maintain rate over training stages, one line per locus (slice A)."""
    slice_a = df[(df["slice"] == "A") & (df["locus"].isin(LOCUS_ORDER))]
    no_cue  = df[(df["slice"] == "A") & (df["locus"] == "no-cue")]
    ckpt_x  = {c: i for i, c in enumerate(checkpoints)}
    colors  = plt.cm.tab10(np.linspace(0, 1, len(LOCUS_ORDER)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, locus in enumerate(LOCUS_ORDER):
        locus_data = slice_a[slice_a["locus"] == locus]
        means, xs = [], []
        for ckpt in checkpoints:
            m, _, _, n = _rates(locus_data[locus_data["checkpoint"] == ckpt], "turn4")
            if n > 0:
                means.append(m); xs.append(ckpt_x[ckpt])
        if xs:
            ax.plot(xs, means, marker="o",
                    label=LOCUS_LABELS[i].replace("\n", " "),
                    color=colors[i], linewidth=1.5)
    nc_means, nc_xs = [], []
    for ckpt in checkpoints:
        m, _, _, n = _rates(no_cue[no_cue["checkpoint"] == ckpt], "turn4")
        if n > 0:
            nc_means.append(m); nc_xs.append(ckpt_x[ckpt])
    if nc_xs:
        ax.plot(nc_xs, nc_means, marker="s", linestyle="--", color="gray",
                label="No cue", linewidth=1.2, alpha=0.7)
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([stage_labels[c] for c in checkpoints])
    ax.set_ylabel("Maintain rate (slice A)"); ax.set_xlabel("Training stage")
    dyn_title = f"Training dynamics: maintain rate by locus — {title}" if title else "Training dynamics: maintain rate by locus"
    ax.set_title(dyn_title)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics.png", dpi=150)
    plt.close(fig)


def _plot_logit_by(df, out_dir, checkpoints, stage_labels, title,
                   group_col, group_order, group_labels, filename, fig_title_suffix):
    adversarial = df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]
    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    axes_flat = axes.flatten()
    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = adversarial if ckpt == "all" else adversarial[adversarial["checkpoint"] == ckpt]
        x = np.arange(len(group_order))
        width = 0.25
        truth_means, other_means, unsure_means = [], [], []
        truth_sems,  other_sems,  unsure_sems  = [], [], []
        for val in group_order:
            grp = sub[sub[group_col] == val]
            truth_means.append(grp["turn4_prob_truth"].mean()  if len(grp) > 0 else 0)
            truth_sems .append(grp["turn4_prob_truth"].sem()   if len(grp) > 1 else 0)
            other_means.append(grp["turn4_prob_other"].mean()  if len(grp) > 0 else 0)
            other_sems .append(grp["turn4_prob_other"].sem()   if len(grp) > 1 else 0)
            unsure_means.append(grp["turn4_prob_unsure"].mean() if len(grp) > 0 else 0)
            unsure_sems .append(grp["turn4_prob_unsure"].sem()  if len(grp) > 1 else 0)
        ax.bar(x - width, truth_means,  width, yerr=truth_sems,
               label="P(truth)"  if ax_idx == 0 else None, color=CHOICE_COLORS["truth"],  capsize=3)
        ax.bar(x,          other_means,  width, yerr=other_sems,
               label="P(other)"  if ax_idx == 0 else None, color=CHOICE_COLORS["other"],  capsize=3)
        ax.bar(x + width,  unsure_means, width, yerr=unsure_sems,
               label="P(unsure)" if ax_idx == 0 else None, color=CHOICE_COLORS["unsure"], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(label, fontsize=11)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Mean probability")
    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8)
    suptitle = f"Logit probabilities by {fig_title_suffix} (adversarial) — {title}" if title else f"Logit probabilities by {fig_title_suffix} (adversarial)"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_logit_breakdowns(df, out_dir, checkpoints, stage_labels, title="",
                           slice_a_balanced=False):
    _plot_logit_by(df, out_dir, checkpoints, stage_labels, title,
                   group_col="locus", group_order=LOCUS_ORDER,
                   group_labels=[LOCUS_SHORT[l] for l in LOCUS_ORDER],
                   filename="logit_probabilities_by_locus.png", fig_title_suffix="locus")
    spec_order = [s for s in SPEC_ORDER if s != "statistical"] if slice_a_balanced else SPEC_ORDER
    _plot_logit_by(df, out_dir, checkpoints, stage_labels, title,
                   group_col="specificity", group_order=spec_order,
                   group_labels=[s.capitalize() for s in spec_order],
                   filename="logit_probabilities_by_specificity.png", fig_title_suffix="specificity")
    if not slice_a_balanced:
        _plot_logit_by(df, out_dir, checkpoints, stage_labels, title,
                       group_col="prior", group_order=PRIOR_ORDER,
                       group_labels=[l.replace("\n", " ") for l in PRIOR_LABELS],
                       filename="logit_probabilities_by_prior.png", fig_title_suffix="prior")
        _plot_logit_by(df, out_dir, checkpoints, stage_labels, title,
                       group_col="dosage", group_order=DOSE_ORDER,
                       group_labels=[d.capitalize() for d in DOSE_ORDER],
                       filename="logit_probabilities_by_dosage.png", fig_title_suffix="dosage")


def plot_all_configs_averaged(df, out_dir, checkpoints, stage_labels, title="",
                              slice_a_balanced=False):
    """Top-level plots averaged across all 8 system configs."""
    avg_title = f"{title} | All configs" if title else "All configs"
    _plot_choice_by_locus(df, out_dir, checkpoints, stage_labels, avg_title)
    _plot_training_dynamics(df, out_dir, checkpoints, stage_labels, avg_title)
    _plot_logit_breakdowns(df, out_dir, checkpoints, stage_labels, avg_title,
                           slice_a_balanced=slice_a_balanced)
    print("  all-configs averaged: choice_by_locus, training_dynamics, "
          "logit_by_locus, logit_by_specificity saved to top-level dir")


def plot_per_condition_breakdowns(df, out_dir, checkpoints, stage_labels, title="",
                                  slice_a_balanced=False):
    """For each of the 8 conditions, generate breakdown plots in by_condition/{cond}/.
    When slice_a_balanced=True, skip Slice B/C and dosage/prior logit plots."""
    base = out_dir / "by_condition"
    for cond in CONDITIONS:
        cdf = df[df["condition"] == cond]
        cond_suffix = cond[len("v4_5_"):]  # e.g. "single_sys_quote"
        cond_dir = base / cond_suffix
        os.makedirs(cond_dir, exist_ok=True)
        cond_title = f"{title} | {COND_SHORT[cond]}" if title else COND_SHORT[cond]
        _plot_choice_by_stage(cdf, cond_dir, checkpoints, stage_labels, cond_title)
        _plot_choice_by_locus(cdf, cond_dir, checkpoints, stage_labels, cond_title)
        _plot_slice_a(cdf, cond_dir, checkpoints, stage_labels, cond_title)
        if not slice_a_balanced:
            _plot_slice_b(cdf, cond_dir, checkpoints, stage_labels, cond_title)
            _plot_slice_c(cdf, cond_dir, checkpoints, stage_labels, cond_title)
        _plot_training_dynamics(cdf, cond_dir, checkpoints, stage_labels, cond_title)
        _plot_logit_breakdowns(cdf, cond_dir, checkpoints, stage_labels, cond_title,
                               slice_a_balanced=slice_a_balanced)
        n_plots = 7 if slice_a_balanced else 10
        print(f"  [{cond_suffix}] {n_plots} plots saved → by_condition/{cond_suffix}/")


# ═════════════════════════════════════════════════════════════════════════════
# 10. Cell summary CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_cell_summary(df, out_dir, stage_labels):
    group_cols = [
        "checkpoint", "condition", "turns", "role", "framing",
        "cell_id", "slice", "locus", "specificity", "dosage", "prior", "goal_condition",
    ]
    agg = (
        df.groupby(group_cols)
        .agg(
            count=("turn4_choice", "size"),
            t4_truth=("turn4_choice", lambda s: (s == "truth").mean()),
            t4_other=("turn4_choice", lambda s: (s == "other").mean()),
            t4_unsure=("turn4_choice", lambda s: (s == "unsure").mean()),
            t2_truth=("turn2_choice", lambda s: (s == "truth").mean()),
            t2_other=("turn2_choice", lambda s: (s == "other").mean()),
            t2_unsure=("turn2_choice", lambda s: (s == "unsure").mean()),
            t4_prob_truth_mean=("turn4_prob_truth", "mean"),
            t4_prob_other_mean=("turn4_prob_other", "mean"),
            t4_prob_unsure_mean=("turn4_prob_unsure", "mean"),
            t2_prob_truth_mean=("turn2_prob_truth", "mean"),
            prob_truth_shift_mean=("prob_truth_shift", "mean"),
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
    parser = argparse.ArgumentParser(description="Analyze v4.5 2x2x2 factorial.")
    parser.add_argument("--model_family", type=str, default="olmo31-32b-instruct",
                        choices=list(MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation, e.g. 'full'. "
                             "Appended to results dirs and output dir.")
    parser.add_argument("--slice_a_balanced", action="store_true",
                        help="Filter to Slice A, fresh prior, non-statistical specificity only. "
                             "Produces a balanced 4 loci × 4 specificities × 100 scenarios subset.")
    args = parser.parse_args()

    family = MODEL_FAMILIES[args.model_family]
    checkpoints = family["checkpoints"]
    stage_labels = family["stage_labels"]
    family_title = family["title"]

    default_out = family["default_output_dir"]
    if args.run_label:
        default_out = f"{default_out}_{args.run_label}"
    if args.slice_a_balanced:
        default_out += "_sliceA_balanced"
    out_dir = REPO_ROOT / (args.output_dir or default_out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model family: {family_title}")
    if args.run_label:
        print(f"Run label: {args.run_label}")
    df = load_results(family, seed=args.seed, run_label=args.run_label)
    print(f"  Loaded {len(df)} rows across {df['checkpoint'].nunique()} checkpoints "
          f"and {df['condition'].nunique()} conditions")

    if args.slice_a_balanced:
        before = len(df)
        df = df[
            ((df["slice"] == "A") & (df["specificity"] != "statistical"))
            | (df["slice"] == "E")
        ]
        df = df[df["prior"] == "fresh"]
        print(f"  --slice_a_balanced: {before} → {len(df)} rows "
              f"(Slice A non-statistical + Slice E sycophancy, fresh prior only; "
              f"Slice D cooperate excluded pending report-framing bug fix)")
        family_title += " [Slice A balanced]"

    report_data_quality(df, out_dir, checkpoints, stage_labels)
    plot_headline_222(df, out_dir, checkpoints, stage_labels, family_title)
    plot_marginals(df, out_dir, checkpoints, stage_labels, family_title)
    plot_interactions(df, out_dir, checkpoints, stage_labels, family_title)
    plot_locus_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    plot_locus_specificity_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    plot_sycophancy(df, out_dir, checkpoints, stage_labels, family_title)
    plot_multi_metrics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_probs(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_vs_generated_maintain(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_vs_generated_confusion(df, out_dir, checkpoints, stage_labels, family_title)
    plot_all_configs_averaged(df, out_dir, checkpoints, stage_labels, family_title,
                              slice_a_balanced=args.slice_a_balanced)
    plot_per_condition_breakdowns(df, out_dir, checkpoints, stage_labels, family_title,
                                  slice_a_balanced=args.slice_a_balanced)
    save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
