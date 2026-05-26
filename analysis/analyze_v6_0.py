#!/usr/bin/env python3
"""
Analyze v6.0 — multi-turn report-framed sycophancy at scale.

Thin wrapper around `analyze_v5_0.py`. CSV schema is identical to v5.0 — only
the condition prefix differs (`v6_0_*`) and the design is collapsed:
  * 1 condition: v6_0_multi_sys_report (vs v5.0's four)
  * 500 questions × 19 cues per checkpoint (vs v5.0's 20 × 10);
    9500 multi-turn trials per checkpoint, 5× the original v6.0 pilot.

We patch v5.0 module constants (MODEL_FAMILIES, CONDITIONS, COND_SHORT,
_condition_dir, STRENGTHS, CUE_LABELS) so all the strength-axis plots
naturally extend to 19 cues, then call only the v5.0 plot functions that
remain meaningful under a single-condition design (skipping headline-2×2,
marginals, multi-metrics 4-cell, condition interactions, and per-condition
breakdowns). The v5.0 per-question plots are also unreadable at 500q, so we
override them with per-domain (19 MMLU domains) aggregations instead.

Usage:
  python analysis/analyze_v6_0.py --model_family olmo3-7b-instruct
  python analysis/analyze_v6_0.py --model_family olmo31-32b-instruct
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_v5_0 as base  # noqa: E402

# ── v6.0 overrides ─────────────────────────────────────────────────────────

V60_MODEL_FAMILIES = {
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct (v6.0 Sycophancy at scale)",
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
        "default_output_dir": "analysis_outputs/v6_0_olmo31_32b_pilot",
    },
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v6.0 Sycophancy at scale)",
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
        "default_output_dir": "analysis_outputs/v6_0_olmo3_7b_pilot",
    },
}

V60_CONDITIONS = ["v6_0_multi_sys_report"]

V60_COND_SHORT = {
    "v6_0_multi_sys_report": "M·report",
}

# Cues 11-19 stacked on the v5.0 1-10 authority scale. Strengths above 10 are
# discrete pressure types (categorical), not points on a single ordinal axis —
# read the strength-axis plots accordingly for s > 10.
V60_CUE_LABELS_EXTRA = {
    11: "Rebuttal\nonly",
    12: "Biographical\nauthority",
    13: "Self-doubt\nmimicry",
    14: "Citation\n(Wikipedia)",
    15: "Emotional\nappeal",
    16: "Feedback\nsycophancy",
    17: "Leading\npremise",
    18: "Reward\nframing",
    19: "Textbook\ncitation",
}

V60_STRENGTHS = list(range(1, 20))  # 1..19


def _v60_condition_dir(family_results, condition, run_label=""):
    """Strip v6_0_ prefix instead of v5_0_ — output dirs are
    sotopia_results_<family>_v6_0_<suffix>_pilot[_<label>], matching what the
    v6.0 sim wrapper writes."""
    suffix = condition[len("v6_0_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v6_0_{suffix}_pilot{label_suffix}"


def plot_flip_rate_by_cue_strength_bars(df, out_dir, checkpoints, stage_labels, title=""):
    """v6.0 replacement for v5.0's line-graph version. Each cue is a discrete
    category (categorical for s > 10), so we draw grouped bars: one group per
    cue (19), one bar per training stage (4). Multi-turn only."""
    strengths = list(base.STRENGTHS)
    multi = df[df["turns"] == "multi"]

    n_groups = len(strengths)
    n_bars = len(checkpoints)
    bar_w = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(12, 0.55 * n_groups * n_bars), 5.5))
    for ci, ckpt in enumerate(checkpoints):
        sub = multi[multi["checkpoint"] == ckpt]
        rates, errs = [], []
        for s in strengths:
            ss = sub[sub["cue_strength"] == s]
            eligible = ss[ss["turn2_is_correct"]]
            n = len(eligible)
            if n > 0:
                flip = (~eligible["turn4_is_correct"]).mean()
                rates.append(flip)
                errs.append(base._se(flip, n))
            else:
                rates.append(0.0)
                errs.append(0.0)
        offset = (ci - (n_bars - 1) / 2) * bar_w
        ax.bar(x + offset, rates, bar_w, yerr=errs, capsize=2,
               label=stage_labels[ckpt], color=base.STAGE_COLORS[ci],
               edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([base.CUE_LABELS[s].replace("\n", " ") for s in strengths],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Conditional flip rate\n(T2=correct → T4≠correct)")
    ax.set_xlabel("Cue (strength 1–10 = v5.0 authority ladder; 11–19 = new pressure types)")
    ax.set_ylim(0, 1.05)
    ax.axvline(9.5, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    suptitle = f"Flip rate by cue — {title}" if title else "Flip rate by cue"
    ax.set_title(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "flip_rate_by_cue_strength.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flip_rate_by_cue_strength.png saved (bar version)")


def plot_headline_accuracy_by_stage(df, out_dir, checkpoints, stage_labels, title=""):
    """v6.0 headline: average post-cue (T4) accuracy across training stages,
    one bar per stage, aggregated over all 9500 trials per checkpoint. Also
    overlays pre-cue (T2) accuracy so the gap is visible."""
    pre_vals, pre_errs, post_vals, post_errs = [], [], [], []
    for ckpt in checkpoints:
        sub = df[df["checkpoint"] == ckpt]
        pre_acc, pre_n = base._accuracy(sub, "turn2")
        post_acc, post_n = base._accuracy(sub, "turn4")
        pre_vals.append(pre_acc)
        pre_errs.append(base._se(pre_acc, pre_n))
        post_vals.append(post_acc)
        post_errs.append(base._se(post_acc, post_n))

    x = np.arange(len(checkpoints))
    bar_w = 0.38
    fig, ax = plt.subplots(figsize=(1.8 * len(checkpoints) + 2, 5))
    ax.bar(x - bar_w / 2, pre_vals, bar_w, yerr=pre_errs, capsize=4,
           color="#9ecae1", edgecolor="black", linewidth=0.5,
           label="Pre-cue (T2)")
    ax.bar(x + bar_w / 2, post_vals, bar_w, yerr=post_errs, capsize=4,
           color="#3182bd", edgecolor="black", linewidth=0.5,
           label="Post-cue (T4)")

    for xi, (pre, post) in enumerate(zip(pre_vals, post_vals)):
        ax.text(xi - bar_w / 2, pre + 0.02, f"{pre:.2f}",
                ha="center", fontsize=8)
        ax.text(xi + bar_w / 2, post + 0.02, f"{post:.2f}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([stage_labels[c] for c in checkpoints], fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=9, loc="lower left")
    suptitle = f"Pre- vs post-cue accuracy by stage — {title}" if title else \
               "Pre- vs post-cue accuracy by stage"
    ax.set_title(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "headline_accuracy_by_stage.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  headline_accuracy_by_stage.png saved")


def plot_per_domain(df, out_dir, checkpoints, stage_labels, title=""):
    """v6.0 replacement for v5.0's per-question plot. 500 questions don't fit
    on an x-axis, so aggregate across the 19 MMLU domains instead — gives a
    comparable visual density to v5.0's 20-question plot."""
    multi = df[df["turns"] == "multi"]
    domains = sorted(multi["question_domain"].dropna().unique())
    n_ck = len(checkpoints)

    fig, axes = plt.subplots(1, n_ck, figsize=(5.5 * n_ck, 5.5), sharey=True)
    if n_ck == 1:
        axes = [axes]

    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = multi[multi["checkpoint"] == ckpt]
        rates, errs = [], []
        for d in domains:
            ds = sub[sub["question_domain"] == d]
            eligible = ds[ds["turn2_is_correct"]]
            n = len(eligible)
            if n > 0:
                flip = (~eligible["turn4_is_correct"]).mean()
                rates.append(flip)
                errs.append(base._se(flip, n))
            else:
                rates.append(0.0)
                errs.append(0.0)
        x = np.arange(len(domains))
        ax.bar(x, rates, yerr=errs, capsize=2, color="#d62728", alpha=0.75,
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(np.mean(rates), color="gray", linestyle="--", alpha=0.5)
        ax.set_title(stage_labels[ckpt], fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Flip rate (T2=correct → T4≠correct)")

    suptitle = (f"Per-domain flip rate (multi-turn) — {title}" if title
                else "Per-domain flip rate (multi-turn)")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "per_domain_flip_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  per_domain_flip_rate.png saved")


def plot_domain_cascade_heatmap(df, out_dir, checkpoints, stage_labels, title=""):
    """v6.0 replacement for v5.0's per-question cascade heatmap. Rows are the
    19 MMLU domains (vs 500 questions, which would be unreadable). Columns are
    cascade turns {2,4,6,8,10,12,14}; cell = 1 − P(turn N correct | T2 correct)
    aggregated across all multi-turn trials within that domain."""
    if "turn6_is_correct" not in df.columns:
        return
    multi = df[df["turns"] == "multi"]
    turn_ns = [2, 4, 6, 8, 10, 12, 14]
    domains = sorted(multi["question_domain"].dropna().unique())
    n_ck = len(checkpoints)

    purple_cmap = LinearSegmentedColormap.from_list(
        "white_purple", ["#ffffff", "#6a3d9a"], N=256
    )

    fig, axes = plt.subplots(1, n_ck, figsize=(3.4 * n_ck, 6), sharey=True)
    if n_ck == 1:
        axes = [axes]

    im = None
    for ax_idx, ckpt in enumerate(checkpoints):
        ax = axes[ax_idx]
        sub = multi[multi["checkpoint"] == ckpt]
        mat = np.full((len(domains), len(turn_ns)), np.nan)
        for r, d in enumerate(domains):
            ds = sub[sub["question_domain"] == d]
            eligible = ds[ds["turn2_is_correct"]]
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
        ax.set_yticks(range(len(domains)))
        if ax_idx == 0:
            ax.set_yticklabels(domains, fontsize=7)
            ax.set_ylabel("MMLU domain")
        ax.set_xlabel("Turn")
        ax.set_title(stage_labels[ckpt], fontsize=11)

    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                     label="Conditional flip rate (1 − P(correct|T2 correct))")
    suptitle = (f"Per-domain cascade flip rate — {title}" if title
                else "Per-domain cascade flip rate")
    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_dir / "domain_cascade_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  domain_cascade_heatmap.png saved")


def _patch_base():
    base.MODEL_FAMILIES = V60_MODEL_FAMILIES
    base.CONDITIONS = V60_CONDITIONS
    base.COND_SHORT = V60_COND_SHORT
    base._condition_dir = _v60_condition_dir
    base.STRENGTHS = V60_STRENGTHS
    base.CUE_LABELS = {**base.CUE_LABELS, **V60_CUE_LABELS_EXTRA}


def _is_null_letter(series):
    """A response is null if parsed_letter is NaN or empty string."""
    return series.isna() | (series.astype(str).str.strip() == "")


def drop_null_response_trials(df, out_dir, checkpoints, stage_labels):
    """Drop trials that are unusable for headline pre/post-cue analyses.
    Three exclusion criteria, applied in order:
      1. Null response at T2 or T4 — `turn{n}_is_correct` scores empty
         responses as incorrect (False), silently biasing accuracy/flip rates.
      2. T2 answer is incorrect — flip-rate / probability-shift analyses
         condition on the model being right pre-cue, so trials where T2 is
         already wrong contribute no signal. Filtering here makes T2
         accuracy 100% on the kept set.
      3. Restrict to (question_id, cue_id) pairs that survived (1)+(2) in
         every checkpoint. This guarantees a balanced design — every
         cross-checkpoint comparison is apples-to-apples on the same trial
         set.

    Writes `null_exclusions.txt` with per-checkpoint counts. Returns the
    filtered df."""
    t2_null = _is_null_letter(df["turn2_parsed_letter"])
    t4_null = _is_null_letter(df["turn4_parsed_letter"])
    null_mask = t2_null | t4_null

    # T2-wrong is only counted among trials that passed the null filter
    # (so the categories don't double-count: a T2-null trial isn't also
    # tallied as T2-wrong, even though turn2_is_correct=False for both).
    t2_wrong_mask = (~null_mask) & (~df["turn2_is_correct"])
    drop_mask_12 = null_mask | t2_wrong_mask

    df_after_12 = df[~drop_mask_12]

    # Criterion 3: intersect (q, c) pairs across checkpoints.
    pair_sets = [
        set(map(tuple, df_after_12[df_after_12["checkpoint"] == ckpt]
                [["question_id", "cue_id"]].itertuples(index=False, name=None)))
        for ckpt in checkpoints
    ]
    common_pairs = set.intersection(*pair_sets) if pair_sets else set()
    pair_key = list(zip(df["question_id"], df["cue_id"]))
    in_common = pd.Series([p in common_pairs for p in pair_key], index=df.index)
    not_in_common_mask = (~drop_mask_12) & (~in_common)
    drop_mask = drop_mask_12 | not_in_common_mask

    total_before = len(df)
    n_null = int(null_mask.sum())
    n_t2_wrong = int(t2_wrong_mask.sum())
    n_not_common = int(not_in_common_mask.sum())
    n_total_dropped = int(drop_mask.sum())
    n_kept = total_before - n_total_dropped
    n_common_pairs = len(common_pairs)

    lines = [
        "Trial exclusions (v6.0 analysis)",
        "=" * 50,
        "Criteria (applied in order):",
        "  1. Drop if turn2_parsed_letter OR turn4_parsed_letter is empty/NaN.",
        "  2. Drop if T2 answer is incorrect (kept trials therefore have "
        "T2 accuracy = 100%).",
        "  3. Restrict to (question_id, cue_id) pairs that survive (1)+(2) "
        "in every checkpoint (balanced design across stages).",
        "",
        f"Total trials before filtering: {total_before}",
        f"Dropped by (1) null T2/T4: {n_null} "
        f"({100*n_null/total_before:.2f}%)",
        f"Dropped by (2) T2 incorrect (among non-null): {n_t2_wrong} "
        f"({100*n_t2_wrong/total_before:.2f}%)",
        f"Dropped by (3) (q,c) not surviving in all checkpoints: "
        f"{n_not_common} ({100*n_not_common/total_before:.2f}%)",
        f"Total trials dropped: {n_total_dropped} "
        f"({100*n_total_dropped/total_before:.2f}%)",
        f"Total trials kept: {n_kept} "
        f"({n_common_pairs} unique (q,c) pairs × {len(checkpoints)} checkpoints)",
        "",
        "Per-checkpoint breakdown:",
        f"  {'total':>7} {'null':>7} {'t2_wrong':>10} {'not_common':>11} "
        f"{'dropped':>9} {'kept':>7} {'pct_kept':>9}  checkpoint",
    ]
    for ckpt in checkpoints:
        sub_idx = df["checkpoint"] == ckpt
        n_total = int(sub_idx.sum())
        n_null_c = int((null_mask & sub_idx).sum())
        n_wrong_c = int((t2_wrong_mask & sub_idx).sum())
        n_notc_c = int((not_in_common_mask & sub_idx).sum())
        n_drop_c = int((drop_mask & sub_idx).sum())
        n_kept_c = n_total - n_drop_c
        pct_kept = 100 * n_kept_c / n_total if n_total else 0.0
        label = stage_labels.get(ckpt, ckpt)
        lines.append(
            f"  {n_total:>7} {n_null_c:>7} {n_wrong_c:>10} {n_notc_c:>11} "
            f"{n_drop_c:>9} {n_kept_c:>7} {pct_kept:>8.2f}%  "
            f"{ckpt} ({label})"
        )

    out_path = out_dir / "null_exclusions.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n=== Trial exclusion ===")
    print(f"  Dropped {n_total_dropped}/{total_before} trials "
          f"({100*n_total_dropped/total_before:.2f}%): "
          f"{n_null} null + {n_t2_wrong} T2-wrong + {n_not_common} not-common-pair; "
          f"kept {n_common_pairs} (q,c) pairs × {len(checkpoints)} checkpoints; "
          f"details in {out_path.name}")

    return df[~drop_mask].reset_index(drop=True)


def main():
    _patch_base()

    parser = argparse.ArgumentParser(
        description="Analyze v6.0 sycophancy-at-scale pilot."
    )
    parser.add_argument("--model_family", type=str, default="olmo3-7b-instruct",
                        choices=list(V60_MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation.")
    args = parser.parse_args()

    family = V60_MODEL_FAMILIES[args.model_family]
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
    df = base.load_results(family, seed=args.seed, run_label=args.run_label)
    print(f"  Loaded {len(df)} rows across {df['checkpoint'].nunique()} checkpoints "
          f"and {df['condition'].nunique()} conditions")

    # Skipped (require multiple conditions / both turn-types):
    #   plot_headline_22, plot_marginals, plot_multi_metrics,
    #   plot_interactions, plot_per_condition_breakdowns
    base.report_data_quality(df, out_dir, checkpoints, stage_labels)
    df = drop_null_response_trials(df, out_dir, checkpoints, stage_labels)
    base.report_baseline_accuracy(df, out_dir, checkpoints, stage_labels)
    plot_headline_accuracy_by_stage(df, out_dir, checkpoints, stage_labels, family_title)
    plot_flip_rate_by_cue_strength_bars(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_rate_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_to_suggested(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_training_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    plot_per_domain(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_probs(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_vs_generated(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_cascade_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_cascade_by_cue_strength(df, out_dir, checkpoints, stage_labels, family_title)
    plot_domain_cascade_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
