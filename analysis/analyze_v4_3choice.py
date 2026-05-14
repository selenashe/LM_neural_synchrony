#!/usr/bin/env python3
"""
Analyze false-belief v4 3-choice forced-choice results.

Reads per-checkpoint CSVs and produces:
  1. Data quality: null rate + cooperate baseline
  2. Choice distribution stacked bars by training stage
  3. Choice distribution stacked bars by locus
  4. Slice A: Maintain rate by locus x specificity
  5. Slice B: Maintain rate by locus x prior
  6. Slice C: Dose-response maintain rate
  7. Training dynamics: maintain rate across checkpoints
  8. Logit probability analysis
  9. Cell-level summary CSV

Usage:
  python analysis/analyze_v4_3choice.py --model_family olmo3-7b-instruct
  python analysis/analyze_v4_3choice.py --model_family olmo31-32b-instruct
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
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct",
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
        "default_results_dir": "sotopia_results_olmo3_instruct_v4_3choice",
        "default_output_dir": "analysis_outputs/v4_3choice",
    },
    "olmo3-7b-think": {
        "title": "OLMo-3 7B Think",
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
        "default_results_dir": "sotopia_results_olmo3_think_v4_3choice",
        "default_output_dir": "analysis_outputs/v4_3choice_think",
    },
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct",
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
        "default_results_dir": "sotopia_results_olmo31_32b_v4_3choice",
        "default_output_dir": "analysis_outputs/v4_3choice_olmo31_32b",
    },
    "olmo3-7b-instruct-full": {
        "title": "OLMo-3 7B Instruct (full 100)",
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
        "default_results_dir": "sotopia_results_olmo3_instruct_v4_3choice_full",
        "default_output_dir": "analysis_outputs/v4_3choice_full",
    },
    "olmo3-7b-think-full": {
        "title": "OLMo-3 7B Think (full 100)",
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
        "default_results_dir": "sotopia_results_olmo3_think_v4_3choice_full",
        "default_output_dir": "analysis_outputs/v4_3choice_think_full",
    },
    "olmo31-32b-instruct-full": {
        "title": "OLMo-3.1 32B Instruct (full 100)",
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
        "default_results_dir": "sotopia_results_olmo31_32b_v4_3choice_full",
        "default_output_dir": "analysis_outputs/v4_3choice_olmo31_32b_full",
    },
}

LOCUS_ORDER = ["source-capacity", "source-motivation", "seeker-capacity", "environmental"]
LOCUS_LABELS = ["Source\ncapacity", "Source\nmotivation", "Seeker\ncapacity", "Environmental"]
LOCUS_SHORT = {
    "source-capacity": "Src cap", "source-motivation": "Src motiv",
    "seeker-capacity": "Skr cap", "environmental": "Environ",
}
SPEC_ORDER = ["direct", "statistical", "mechanism", "inferential", "behavioral"]
PRIOR_ORDER = ["none", "stale-time", "stale-quality", "fresh"]
PRIOR_LABELS = ["None", "Stale\n(time)", "Stale\n(quality)", "Fresh"]
DOSE_ORDER = ["weak", "medium", "strong"]

CHOICE_COLORS = {"maintain": "#2ca02c", "defer": "#d62728", "unsure": "#7f7f7f"}
LOCUS_COLORS = {
    "source-capacity": "#1f77b4", "source-motivation": "#d62728",
    "seeker-capacity": "#e377c2", "environmental": "#17becf",
}


def load_results(results_dir, seed=0, checkpoints=None):
    frames = []
    for ckpt in checkpoints:
        csv_path = results_dir / f"results_{ckpt}_seed{seed}.csv"
        if not csv_path.exists():
            print(f"  WARNING: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No result CSVs found in {results_dir}")
    df = pd.concat(frames, ignore_index=True)

    df["prob_A"] = pd.to_numeric(df["prob_A"], errors="coerce")
    df["prob_B"] = pd.to_numeric(df["prob_B"], errors="coerce")
    df["prob_C"] = pd.to_numeric(df["prob_C"], errors="coerce")
    df["prob_truth"] = np.where(df["truth_position"] == "A", df["prob_A"], df["prob_B"])
    df["prob_other"] = np.where(df["truth_position"] == "A", df["prob_B"], df["prob_A"])
    df["prob_unsure"] = df["prob_C"]

    return df


def _choice_rates(sub):
    n = len(sub)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    maintain = (sub["choice"] == "truth").sum() / n
    defer = (sub["choice"] == "other").sum() / n
    unsure = (sub["choice"] == "unsure").sum() / n
    return maintain, defer, unsure, n


def _maintain_se(sub):
    n = len(sub)
    if n < 2:
        return 0.0
    p = (sub["choice"] == "truth").sum() / n
    return np.sqrt(p * (1 - p) / n)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Data quality
# ═════════════════════════════════════════════════════════════════════════════

def report_data_quality(df, out_dir, checkpoints, stage_labels):
    print("\n=== Data Quality ===")

    null_rates = []
    for ckpt in checkpoints:
        sub = df[df["checkpoint"] == ckpt]
        total = len(sub)
        nulls = (sub["parsed_letter"] == "").sum() + sub["parsed_letter"].isna().sum()
        null_rates.append({
            "checkpoint": stage_labels[ckpt],
            "total": total,
            "null": int(nulls),
            "null_rate": nulls / total if total > 0 else 0,
        })
    null_df = pd.DataFrame(null_rates)
    print("\nNull rates:")
    print(null_df.to_string(index=False))
    null_df.to_csv(out_dir / "null_rates.csv", index=False)

    print("\nCooperate baseline (choice distribution):")
    coop = df[df["goal_condition"] == "cooperate"]
    coop_rows = []
    for ckpt in checkpoints:
        sub = coop[coop["checkpoint"] == ckpt]
        m, d, u, n = _choice_rates(sub)
        coop_rows.append({
            "checkpoint": stage_labels[ckpt],
            "maintain": f"{m:.3f}", "defer": f"{d:.3f}", "unsure": f"{u:.3f}",
            "n": n,
        })
    coop_df = pd.DataFrame(coop_rows)
    print(coop_df.to_string(index=False))
    coop_df.to_csv(out_dir / "cooperate_baseline.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Choice distribution by training stage (stacked bars)
# ═════════════════════════════════════════════════════════════════════════════

def plot_choice_by_stage(df, out_dir, checkpoints, stage_labels, title=""):
    adversarial = df[(df["goal_condition"] == "adversarial") & (df["locus"] != "no-cue")]
    cooperate = df[df["goal_condition"] == "cooperate"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (cond_label, sub_df) in enumerate([
        ("Adversarial (with cues)", adversarial),
        ("Cooperate", cooperate),
    ]):
        ax = axes[ax_idx]
        labels = []
        maintain_vals, defer_vals, unsure_vals = [], [], []

        for ckpt in checkpoints:
            sub = sub_df[sub_df["checkpoint"] == ckpt]
            m, d, u, _ = _choice_rates(sub)
            labels.append(stage_labels[ckpt])
            maintain_vals.append(m)
            defer_vals.append(d)
            unsure_vals.append(u)

        x = np.arange(len(labels))
        ax.bar(x, maintain_vals, label="Maintain (truth)",
               color=CHOICE_COLORS["maintain"])
        ax.bar(x, defer_vals, bottom=maintain_vals, label="Defer (other)",
               color=CHOICE_COLORS["defer"])
        ax.bar(x, unsure_vals,
               bottom=[m + d for m, d in zip(maintain_vals, defer_vals)],
               label="Unsure", color=CHOICE_COLORS["unsure"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1.05)
        ax.set_title(cond_label)
        ax.legend(fontsize=8, loc="upper right")

    suptitle = f"Choice distribution by training stage — {title}" if title else "Choice distribution by training stage"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "choice_by_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Choice by stage plot saved")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Choice distribution by locus (stacked bars)
# ═════════════════════════════════════════════════════════════════════════════

def plot_choice_by_locus(df, out_dir, checkpoints, stage_labels, title=""):
    adversarial = df[(df["goal_condition"] == "adversarial") & (df["locus"].isin(LOCUS_ORDER))]

    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    if n_panels == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = adversarial if ckpt == "all" else adversarial[adversarial["checkpoint"] == ckpt]

        maintain_vals, defer_vals, unsure_vals = [], [], []
        for locus in LOCUS_ORDER:
            m, d, u, _ = _choice_rates(sub[sub["locus"] == locus])
            maintain_vals.append(m)
            defer_vals.append(d)
            unsure_vals.append(u)

        x = np.arange(len(LOCUS_ORDER))
        ax.bar(x, maintain_vals, color=CHOICE_COLORS["maintain"],
               label="Maintain" if ax_idx == 0 else None)
        ax.bar(x, defer_vals, bottom=maintain_vals, color=CHOICE_COLORS["defer"],
               label="Defer" if ax_idx == 0 else None)
        ax.bar(x, unsure_vals,
               bottom=[m + d for m, d in zip(maintain_vals, defer_vals)],
               color=CHOICE_COLORS["unsure"],
               label="Unsure" if ax_idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([LOCUS_SHORT[l] for l in LOCUS_ORDER], fontsize=8, rotation=20, ha="right")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
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
    print("  Choice by locus plot saved")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Slice A: Maintain rate by locus x specificity
# ═════════════════════════════════════════════════════════════════════════════

def plot_slice_a(df, out_dir, checkpoints, stage_labels, title=""):
    slice_a = df[(df["slice"] == "A") & (df["locus"].isin(LOCUS_ORDER))]

    no_cue = df[(df["slice"] == "A") & (df["locus"] == "no-cue")]

    n_loci = len(LOCUS_ORDER)
    width = 0.18

    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()

    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = slice_a if ckpt == "all" else slice_a[slice_a["checkpoint"] == ckpt]
        nc = no_cue if ckpt == "all" else no_cue[no_cue["checkpoint"] == ckpt]

        nc_maintain, _, _, _ = _choice_rates(nc)

        x = np.arange(len(SPEC_ORDER))
        for i, locus in enumerate(LOCUS_ORDER):
            locus_data = sub[sub["locus"] == locus]
            vals = []
            errs = []
            for spec in SPEC_ORDER:
                spec_data = locus_data[locus_data["specificity"] == spec]
                m, _, _, _ = _choice_rates(spec_data)
                vals.append(m)
                errs.append(_maintain_se(spec_data))
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=LOCUS_SHORT[locus] if ax_idx == 0 else None,
                   color=LOCUS_COLORS[locus], capsize=2)

        ax.axhline(y=nc_maintain, color="gray", linestyle="--", alpha=0.6,
                   label="No cue" if ax_idx == 0 else None)
        ax.set_xticks(x + width * (n_loci - 1) / 2)
        ax.set_xticklabels([s.capitalize() for s in SPEC_ORDER],
                           fontsize=8, rotation=30, ha="right")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")

    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="lower left")
    suptitle = f"Slice A: Maintain rate by specificity x locus — {title}" if title else "Slice A: Maintain rate by specificity x locus"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_A_maintain_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Slice A maintain rate chart saved")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Slice B: Maintain rate by locus x prior
# ═════════════════════════════════════════════════════════════════════════════

def plot_slice_b(df, out_dir, checkpoints, stage_labels, title=""):
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
            vals = []
            errs = []
            for prior in PRIOR_ORDER:
                prior_data = locus_data[locus_data["prior"] == prior]
                m, _, _, _ = _choice_rates(prior_data)
                vals.append(m)
                errs.append(_maintain_se(prior_data))
            ax.bar(x + i * width, vals, width, yerr=errs,
                   label=locus_labels[locus] if ax_idx == 0 else None,
                   color=locus_colors[locus], capsize=2)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_xticks(x + width * (n_loci - 1) / 2)
        ax.set_xticklabels(PRIOR_LABELS, fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")

    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="lower left")
    suptitle = f"Slice B: Maintain rate by prior x locus — {title}" if title else "Slice B: Maintain rate by prior x locus"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_B_maintain_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Slice B maintain rate chart saved")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Slice C: Dose-response
# ═════════════════════════════════════════════════════════════════════════════

def plot_slice_c(df, out_dir, checkpoints, stage_labels, title=""):
    slice_c = df[
        (df["slice"].isin(["A", "C"])) &
        (df["specificity"] == "statistical") &
        (df["locus"].isin(LOCUS_ORDER)) &
        (df["prior"] == "fresh")
    ]

    panels = [(ckpt, stage_labels[ckpt]) for ckpt in checkpoints] + [("all", "All checkpoints")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()

    for ax_idx, (ckpt, label) in enumerate(panels):
        ax = axes_flat[ax_idx]
        sub = slice_c if ckpt == "all" else slice_c[slice_c["checkpoint"] == ckpt]

        dose_x = {d: i for i, d in enumerate(DOSE_ORDER)}
        for locus in LOCUS_ORDER:
            locus_data = sub[sub["locus"] == locus]
            means, xs = [], []
            for dose in DOSE_ORDER:
                m, _, _, n = _choice_rates(locus_data[locus_data["dosage"] == dose])
                if n > 0:
                    means.append(m)
                    xs.append(dose_x[dose])
            if xs:
                ax.plot(xs, means, marker="o",
                        label=LOCUS_SHORT[locus] if ax_idx == 0 else None,
                        color=LOCUS_COLORS[locus], linewidth=1.5)

        ax.set_xticks(range(len(DOSE_ORDER)))
        ax.set_xticklabels([d.capitalize() for d in DOSE_ORDER])
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1.05)
        if ax_idx % 3 == 0:
            ax.set_ylabel("Maintain rate")

    axes_flat[5].set_visible(False)
    axes_flat[0].legend(fontsize=8, loc="best")
    suptitle = f"Slice C: Dose-response maintain rate — {title}" if title else "Slice C: Dose-response maintain rate"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "slice_C_dose_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Slice C dose-response chart saved")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Training dynamics
# ═════════════════════════════════════════════════════════════════════════════

def plot_training_dynamics(df, out_dir, checkpoints, stage_labels, title=""):
    slice_a = df[(df["slice"] == "A") & (df["locus"].isin(LOCUS_ORDER))]
    no_cue = df[(df["slice"] == "A") & (df["locus"] == "no-cue")]

    ckpt_x = {c: i for i, c in enumerate(checkpoints)}
    colors = plt.cm.tab10(np.linspace(0, 1, len(LOCUS_ORDER)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, locus in enumerate(LOCUS_ORDER):
        locus_data = slice_a[slice_a["locus"] == locus]
        means, xs = [], []
        for ckpt in checkpoints:
            m, _, _, n = _choice_rates(locus_data[locus_data["checkpoint"] == ckpt])
            if n > 0:
                means.append(m)
                xs.append(ckpt_x[ckpt])
        if xs:
            ax.plot(xs, means, marker="o",
                    label=LOCUS_LABELS[i].replace("\n", " "),
                    color=colors[i], linewidth=1.5)

    nc_means, nc_xs = [], []
    for ckpt in checkpoints:
        m, _, _, n = _choice_rates(no_cue[no_cue["checkpoint"] == ckpt])
        if n > 0:
            nc_means.append(m)
            nc_xs.append(ckpt_x[ckpt])
    if nc_xs:
        ax.plot(nc_xs, nc_means, marker="s", linestyle="--", color="gray",
                label="No cue", linewidth=1.2, alpha=0.7)

    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([stage_labels[c] for c in checkpoints])
    ax.set_ylabel("Maintain rate (slice A)")
    ax.set_xlabel("Training stage")
    dyn_title = f"Training dynamics: maintain rate by locus — {title}" if title else "Training dynamics: maintain rate by locus"
    ax.set_title(dyn_title)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "training_dynamics.png", dpi=150)
    plt.close(fig)
    print("  Training dynamics plot saved")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Logit probability analysis
# ═════════════════════════════════════════════════════════════════════════════

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
        truth_sems, other_sems, unsure_sems = [], [], []
        for val in group_order:
            grp = sub[sub[group_col] == val]
            truth_means.append(grp["prob_truth"].mean() if len(grp) > 0 else 0)
            truth_sems.append(grp["prob_truth"].sem() if len(grp) > 1 else 0)
            other_means.append(grp["prob_other"].mean() if len(grp) > 0 else 0)
            other_sems.append(grp["prob_other"].sem() if len(grp) > 1 else 0)
            unsure_means.append(grp["prob_unsure"].mean() if len(grp) > 0 else 0)
            unsure_sems.append(grp["prob_unsure"].sem() if len(grp) > 1 else 0)

        ax.bar(x - width, truth_means, width, yerr=truth_sems,
               label="P(truth)" if ax_idx == 0 else None,
               color=CHOICE_COLORS["maintain"], capsize=3)
        ax.bar(x, other_means, width, yerr=other_sems,
               label="P(other)" if ax_idx == 0 else None,
               color=CHOICE_COLORS["defer"], capsize=3)
        ax.bar(x + width, unsure_means, width, yerr=unsure_sems,
               label="P(unsure)" if ax_idx == 0 else None,
               color=CHOICE_COLORS["unsure"], capsize=3)

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


def plot_logit_analysis(df, out_dir, checkpoints, stage_labels, title=""):
    _plot_logit_by(
        df, out_dir, checkpoints, stage_labels, title,
        group_col="locus", group_order=LOCUS_ORDER,
        group_labels=[LOCUS_SHORT[l] for l in LOCUS_ORDER],
        filename="logit_probabilities_by_locus.png", fig_title_suffix="locus",
    )
    print("  Logit probabilities by locus saved")

    _plot_logit_by(
        df, out_dir, checkpoints, stage_labels, title,
        group_col="specificity", group_order=SPEC_ORDER,
        group_labels=[s.capitalize() for s in SPEC_ORDER],
        filename="logit_probabilities_by_specificity.png", fig_title_suffix="specificity",
    )
    print("  Logit probabilities by specificity saved")

    _plot_logit_by(
        df, out_dir, checkpoints, stage_labels, title,
        group_col="prior", group_order=PRIOR_ORDER,
        group_labels=[l.replace("\n", " ") for l in PRIOR_LABELS],
        filename="logit_probabilities_by_prior.png", fig_title_suffix="prior",
    )
    print("  Logit probabilities by prior saved")

    _plot_logit_by(
        df, out_dir, checkpoints, stage_labels, title,
        group_col="dosage", group_order=DOSE_ORDER,
        group_labels=[d.capitalize() for d in DOSE_ORDER],
        filename="logit_probabilities_by_dosage.png", fig_title_suffix="dosage",
    )
    print("  Logit probabilities by dosage saved")


# ═════════════════════════════════════════════════════════════════════════════
# 9. Summary CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_cell_summary(df, out_dir, stage_labels):
    group_cols = [
        "checkpoint", "cell_id", "slice", "locus", "specificity",
        "dosage", "prior", "goal_condition",
    ]
    agg = (
        df.groupby(group_cols)
        .agg(
            maintain_rate=("choice", lambda s: (s == "truth").mean()),
            defer_rate=("choice", lambda s: (s == "other").mean()),
            unsure_rate=("choice", lambda s: (s == "unsure").mean()),
            prob_truth_mean=("prob_truth", "mean"),
            prob_other_mean=("prob_other", "mean"),
            prob_unsure_mean=("prob_unsure", "mean"),
            count=("choice", "size"),
        )
        .reset_index()
    )
    agg["checkpoint_label"] = agg["checkpoint"].map(stage_labels)
    agg.to_csv(out_dir / "cell_summary.csv", index=False)
    print(f"  Cell summary saved: {len(agg)} rows")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze v4 3-choice forced-choice results."
    )
    parser.add_argument(
        "--model_family", type=str, default="olmo3-7b-instruct",
        choices=list(MODEL_FAMILIES.keys()),
    )
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    family = MODEL_FAMILIES[args.model_family]
    checkpoints = family["checkpoints"]
    stage_labels = family["stage_labels"]
    family_title = family["title"]

    results_dir = REPO_ROOT / (args.results_dir or family["default_results_dir"])
    out_dir = REPO_ROOT / (args.output_dir or family["default_output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model family: {family_title}")
    print(f"Loading results from {results_dir}")
    df = load_results(results_dir, seed=args.seed, checkpoints=checkpoints)
    print(f"  Loaded {len(df)} rows across {df['checkpoint'].nunique()} checkpoints")

    report_data_quality(df, out_dir, checkpoints, stage_labels)
    plot_choice_by_stage(df, out_dir, checkpoints, stage_labels, family_title)
    plot_choice_by_locus(df, out_dir, checkpoints, stage_labels, family_title)
    plot_slice_a(df, out_dir, checkpoints, stage_labels, family_title)
    plot_slice_b(df, out_dir, checkpoints, stage_labels, family_title)
    plot_slice_c(df, out_dir, checkpoints, stage_labels, family_title)
    plot_training_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_logit_analysis(df, out_dir, checkpoints, stage_labels, family_title)
    save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
