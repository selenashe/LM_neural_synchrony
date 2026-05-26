#!/usr/bin/env python3
"""Tier-1 exploratory analysis on the v6.0 filtered trial set.

Operates on the balanced cell_summary written by analyze_v6_0.py — every row
is one (checkpoint, question, cue) trial in which T2 was correct and T4 was
parseable, and (q, c) pairs are present for all 4 checkpoints. With this
input, the analysis can compare checkpoints on the same trials without
worrying about coverage drift.

Outcomes used throughout:
  flip       = 1 - t4_accuracy                 (binary)
  shift      = prob_correct_shift_mean         (continuous, t4-t2)
  shift_norm = shift / max_possible_shift_in_observed_direction (in [-1, 1])
               i.e. shift / (1 - t2_prob) when shift >= 0, else shift / t2_prob.
               Removes the mechanical ceiling/floor artifact baked into raw
               shift: a high-confidence trial cannot move much further up;
               a low-confidence trial cannot move much further down. Reported
               in parallel with `shift`.
  confidence = t2_prob_correct_mean            (raw correct-letter prob;
               NOT normalized over A/B/C/D — see goal 6 caveats)

Each (checkpoint, question, cue) cell has exactly one underlying trial
(count == 1 in cell_summary), so any "interaction" term is unidentified from
residual variance. We report it descriptively only.

Usage:
  python analysis/analyze_v6_0_filtered.py
  python analysis/analyze_v6_0_filtered.py \
    --input analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv \
    --output_dir analysis_outputs/v6_0_olmo3_7b_pilot/filtered_followup \
    --balanced_min_cues 18
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans  # noqa: F401  (kept available for ad-hoc)

REPO_ROOT = Path(__file__).resolve().parent.parent

STAGE_ORDER = ["Base", "SFT", "DPO", "RLVR"]
STAGE_COLORS = {"Base": "#4575b4", "SFT": "#74add1",
                "DPO": "#f46d43", "RLVR": "#d73027"}


# ── Setup ──────────────────────────────────────────────────────────────────

def load_cells(path):
    df = pd.read_csv(path)
    needed = {"checkpoint", "checkpoint_label", "question_id",
              "question_domain", "cue_id", "cue_strength",
              "cue_strength_label", "t4_accuracy",
              "t4_flipped_to_suggested", "t4_prob_correct_mean",
              "t4_prob_wrong_mean", "t2_prob_correct_mean",
              "prob_correct_shift_mean"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    df["flip"] = 1.0 - df["t4_accuracy"].astype(float)
    df["shift"] = df["prob_correct_shift_mean"].astype(float)
    df["confidence"] = df["t2_prob_correct_mean"].astype(float)

    # Bound-normalized shift: divide by the room the trial actually had to
    # move in the observed direction. Removes the mechanical ceiling/floor
    # artifact (high-conf trials can't shift up much; low-conf can't shift
    # down much). Result is in [-1, 1].
    shift_v = df["shift"].values
    conf_v = df["confidence"].values
    max_up = 1.0 - conf_v
    max_down = conf_v
    denom = np.where(shift_v >= 0, max_up, max_down)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift_norm = np.where(denom > 1e-9, shift_v / denom, 0.0)
    df["shift_norm"] = np.clip(shift_norm, -1.0, 1.0)

    if not (df.groupby("checkpoint_label").size().nunique() == 1):
        print("WARNING: row counts differ across checkpoints — input may not "
              "be the balanced filtered set.", file=sys.stderr)
    return df


# ── 1. Coverage check ─────────────────────────────────────────────────────

def coverage(df, out_dir, min_cues=18):
    """Per-question cue-coverage in the filtered set; flag a balanced
    subset (questions with ≥ min_cues cues retained for every checkpoint)
    for use downstream."""
    cov = (df.groupby(["checkpoint_label", "question_id"])["cue_id"]
             .nunique().rename("n_cues").reset_index())
    cov.to_csv(out_dir / "coverage_per_question.csv", index=False)

    pivot = cov.pivot(index="question_id", columns="checkpoint_label",
                      values="n_cues").fillna(0).astype(int)
    pivot = pivot.reindex(columns=[c for c in STAGE_ORDER if c in pivot.columns])
    min_across = pivot.min(axis=1)
    balanced_qids = set(min_across.index[min_across >= min_cues])

    n_total_q = pivot.shape[0]
    n_balanced = len(balanced_qids)
    summary_lines = [
        "Coverage summary (filtered cell_summary)",
        "=" * 50,
        f"Unique questions in filtered set: {n_total_q}",
        f"Threshold for 'balanced subset': ≥ {min_cues} cues "
        f"per question per checkpoint",
        f"Balanced subset: {n_balanced} questions "
        f"({100*n_balanced/n_total_q:.1f}%) "
        f"covering up to 19 cues × {n_balanced} ≈ "
        f"{19*n_balanced} cells per checkpoint",
        "",
        "Distribution of min(cues across checkpoints) per question:",
    ]
    counts = min_across.value_counts().sort_index()
    for n_cue, n_q in counts.items():
        summary_lines.append(f"  {n_cue:>3} cues: {n_q:>4} questions")

    # Domain composition: balanced vs not balanced — is the balanced subset
    # biased toward a particular domain?
    dom = df[["question_id", "question_domain"]].drop_duplicates()
    dom["balanced"] = dom["question_id"].isin(balanced_qids)
    dom_breakdown = (dom.groupby(["question_domain", "balanced"]).size()
                       .unstack(fill_value=0)
                       .rename(columns={True: "balanced", False: "other"}))
    if "balanced" not in dom_breakdown.columns:
        dom_breakdown["balanced"] = 0
    if "other" not in dom_breakdown.columns:
        dom_breakdown["other"] = 0
    dom_breakdown["total"] = dom_breakdown.sum(axis=1)
    dom_breakdown["pct_balanced"] = (
        100 * dom_breakdown["balanced"] / dom_breakdown["total"])
    dom_breakdown = dom_breakdown.sort_values("total", ascending=False)
    dom_breakdown.to_csv(out_dir / "coverage_by_domain.csv")

    summary_lines.append("")
    summary_lines.append("Domain composition (top 10 by total questions):")
    summary_lines.append(dom_breakdown.head(10).to_string())

    (out_dir / "coverage_summary.txt").write_text("\n".join(summary_lines) + "\n")
    pd.Series(sorted(balanced_qids), name="question_id").to_csv(
        out_dir / "balanced_subset_question_ids.csv", index=False)

    # Coverage heatmap
    fig, ax = plt.subplots(figsize=(6, max(4, 0.06 * pivot.shape[0])))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=19)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel(f"questions (n={pivot.shape[0]}, sorted by question_id)")
    ax.set_title(f"cues retained per question per checkpoint\n"
                 f"(balanced subset: ≥{min_cues} cues → {n_balanced} questions)")
    fig.colorbar(im, ax=ax, label="n cues")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  coverage: {n_total_q} questions; balanced subset "
          f"(≥{min_cues} cues) = {n_balanced}")
    return balanced_qids


# ── 2. Variance decomposition ─────────────────────────────────────────────

def _eta_squared(values, groups):
    """One-way variance decomposition: SS_between / SS_total. Returns η²
    and the F-test p-value (informative only — no within-cell replication,
    F uses (N - K) residual df)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = ~np.isnan(values)
    values, groups = values[mask], groups[mask]
    if values.size < 3:
        return np.nan, np.nan, 0
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0, np.nan, len(np.unique(groups))
    ss_between = 0.0
    levels = np.unique(groups)
    for lev in levels:
        sub = values[groups == lev]
        if sub.size:
            ss_between += sub.size * (sub.mean() - grand) ** 2
    eta2 = ss_between / ss_total
    K = len(levels); N = values.size
    if K > 1 and N > K:
        ss_within = ss_total - ss_between
        F = (ss_between / (K - 1)) / (ss_within / (N - K)) if ss_within > 0 else np.inf
        p = 1 - stats.f.cdf(F, K - 1, N - K)
    else:
        p = np.nan
    return eta2, p, K


def variance_decomposition(df, out_dir):
    """Per-checkpoint η² for each factor and each outcome. Factors are
    estimated independently (not partitioned) so they overlap — interpret
    as "how much variation aligns with this factor."

    Q×C is included as an upper-bound diagnostic: with one row per (q,c)
    cell, the additive question+cue model already saturates Q×C, so the
    "interaction" η² is whatever variance remains after factoring question
    and cue out — not separable from residual."""
    rows = []
    factors = ("question_id", "cue_id", "question_domain")
    for ckpt, ck_df in df.groupby("checkpoint_label"):
        for outcome in ("flip", "shift", "shift_norm"):
            for factor in factors:
                eta2, p, K = _eta_squared(ck_df[outcome].values,
                                          ck_df[factor].values)
                rows.append({
                    "checkpoint": ckpt, "outcome": outcome,
                    "factor": factor, "n_levels": K,
                    "eta_squared": eta2, "p_value": p,
                })

    var_df = pd.DataFrame(rows)
    var_df.to_csv(out_dir / "variance_decomposition.csv", index=False)

    # Plot: grouped bars (one panel per outcome, stages on x, factors as bars)
    factors_plot = list(factors)
    outcomes_plot = ("flip", "shift", "shift_norm")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    for ax, outcome in zip(axes, outcomes_plot):
        sub = var_df[var_df["outcome"] == outcome]
        stages = [s for s in STAGE_ORDER
                  if s in sub["checkpoint"].unique()]
        x = np.arange(len(stages))
        bar_w = 0.85 / len(factors_plot)
        for i, fac in enumerate(factors_plot):
            vals = [sub[(sub["checkpoint"] == s) &
                        (sub["factor"] == fac)]["eta_squared"].values
                    for s in stages]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - (len(factors_plot) - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, label=fac, edgecolor="black",
                   linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylabel("η² / R²")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Outcome: {outcome}")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Variance associated with each factor (per checkpoint, per outcome)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "variance_decomposition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print("  variance decomposition: variance_decomposition.csv/.png")
    return var_df


# ── 3. Per-question resistance profile ────────────────────────────────────

def per_question_profile(df, out_dir):
    g = (df.groupby(["checkpoint_label", "question_id"])
           .agg(n_obs=("flip", "size"),
                mean_flip=("flip", "mean"),
                mean_shift=("shift", "mean"),
                mean_shift_norm=("shift_norm", "mean"),
                mean_t2_conf=("confidence", "mean"),
                mean_t4_prob_correct=("t4_prob_correct_mean", "mean"),
                domain=("question_domain", "first"))
           .reset_index())
    g.to_csv(out_dir / "per_question_profile.csv", index=False)

    # Ranking plot: per checkpoint, top 20 most resistant + top 20 most flippable
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True)
    for ax, ckpt in zip(axes, STAGE_ORDER):
        sub = g[g["checkpoint_label"] == ckpt].sort_values("mean_flip")
        if sub.empty:
            ax.set_visible(False); continue
        n = len(sub)
        top_resist = sub.head(20)["question_id"].tolist()
        top_flip = sub.tail(20)["question_id"].tolist()
        show = pd.concat([sub.head(20), sub.tail(20)])
        y = np.arange(len(show))
        colors = ["#1a9850"] * 20 + ["#d73027"] * 20
        ax.barh(y, show["mean_flip"].values, color=colors, edgecolor="black",
                linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(show["question_id"].tolist(), fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("flip rate")
        ax.set_title(f"{ckpt} — most resistant (green) vs most flippable (red)")
        ax.axhline(19.5, color="gray", linestyle=":", linewidth=0.5)
    fig.suptitle(f"Per-question flip rate — top/bottom 20 (n_questions={g['question_id'].nunique()})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "per_question_resistance_ranking.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  per-question profile: {g.shape[0]} rows")
    return g


# ── 4. Per-cue effectiveness profile ──────────────────────────────────────

def per_cue_profile(df, out_dir):
    g = (df.groupby(["checkpoint_label", "cue_id"])
           .agg(n_obs=("flip", "size"),
                mean_flip=("flip", "mean"),
                mean_shift=("shift", "mean"),
                mean_shift_norm=("shift_norm", "mean"),
                mean_t2_conf=("confidence", "mean"),
                cue_strength=("cue_strength", "first"),
                cue_strength_label=("cue_strength_label", "first"))
           .reset_index())
    g.to_csv(out_dir / "per_cue_profile.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True)
    cue_label_order = (g.groupby("cue_id")["mean_flip"].mean()
                         .sort_values().index.tolist())
    for ax, ckpt in zip(axes, STAGE_ORDER):
        sub = g[g["checkpoint_label"] == ckpt].set_index("cue_id").reindex(
            cue_label_order).reset_index()
        if sub.empty:
            ax.set_visible(False); continue
        y = np.arange(len(sub))
        ax.barh(y, sub["mean_flip"].values, color="#d73027",
                edgecolor="black", linewidth=0.4)
        labels = [f"{cid} [{lbl}]" for cid, lbl
                  in zip(sub["cue_id"], sub["cue_strength_label"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("flip rate")
        ax.set_xlim(0, 1.0)
        ax.set_title(ckpt)
    fig.suptitle("Per-cue flip rate by checkpoint (cues sorted by avg flip across checkpoints)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "per_cue_effectiveness_ranking.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  per-cue profile: {g.shape[0]} rows")
    return g


# ── 5. Rank stability across checkpoints ──────────────────────────────────

def rank_stability(per_q, per_c, out_dir):
    def _spearman_matrix(profile, key, metric):
        pivot = profile.pivot(index=key, columns="checkpoint_label",
                              values=metric)
        pivot = pivot.reindex(columns=[s for s in STAGE_ORDER
                                       if s in pivot.columns])
        rho = pivot.corr(method="spearman")
        return pivot, rho

    rows = []
    metrics = ("mean_flip", "mean_shift", "mean_shift_norm")
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 4.5 * len(metrics)))

    for col, (profile, key) in enumerate([
            (per_q, "question_id"), (per_c, "cue_id")]):
        for row, metric in enumerate(metrics):
            pivot, rho = _spearman_matrix(profile, key, metric)
            ax = axes[row, col]
            im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(rho))); ax.set_yticks(range(len(rho)))
            ax.set_xticklabels(rho.columns); ax.set_yticklabels(rho.index)
            for i in range(len(rho)):
                for j in range(len(rho)):
                    ax.text(j, i, f"{rho.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="black" if abs(rho.values[i, j]) < 0.5
                                  else "white")
            ax.set_title(f"{key} — Spearman ρ on {metric}", fontsize=10)
            for i, c1 in enumerate(rho.columns):
                for j, c2 in enumerate(rho.columns):
                    if i < j:
                        rows.append({
                            "axis": key, "metric": metric,
                            "ckpt_a": c1, "ckpt_b": c2,
                            "spearman_rho": rho.values[i, j],
                        })
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Spearman ρ")
    fig.suptitle("Rank stability across checkpoints", fontsize=12)
    fig.savefig(out_dir / "rank_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rs_df = pd.DataFrame(rows)
    rs_df.to_csv(out_dir / "rank_stability.csv", index=False)
    print(f"  rank stability: {len(rs_df)} pairwise comparisons")
    return rs_df


# ── 6. Confidence vs outcome ──────────────────────────────────────────────

def confidence_vs_outcome(df, per_q, out_dir):
    """Pre-cue confidence vs post-cue outcome, at trial level and at
    per-question aggregate level. Trial-level uses the raw cell prob (one
    correct-letter probability per trial); the per-question version pools
    over cues. Cross-checkpoint comparison uses ranks/relative metrics
    because raw probability calibration may differ across checkpoints."""
    # Trial-level Pearson + Spearman of (confidence, flip) and (confidence, shift)
    rows = []
    for ckpt, sub in df.groupby("checkpoint_label"):
        for outcome in ("flip", "shift", "shift_norm"):
            valid = sub[["confidence", outcome]].dropna()
            if len(valid) > 2:
                pear = stats.pearsonr(valid["confidence"], valid[outcome])
                spear = stats.spearmanr(valid["confidence"], valid[outcome])
            else:
                pear = spear = (np.nan, np.nan)
            rows.append({
                "checkpoint": ckpt, "level": "trial",
                "outcome": outcome, "n": len(valid),
                "pearson_r": pear[0], "pearson_p": pear[1],
                "spearman_rho": spear[0], "spearman_p": spear[1],
            })
    # Per-question (within checkpoint)
    for ckpt, sub in per_q.groupby("checkpoint_label"):
        for outcome in ("mean_flip", "mean_shift", "mean_shift_norm"):
            valid = sub[["mean_t2_conf", outcome]].dropna()
            if len(valid) > 2:
                pear = stats.pearsonr(valid["mean_t2_conf"], valid[outcome])
                spear = stats.spearmanr(valid["mean_t2_conf"], valid[outcome])
            else:
                pear = spear = (np.nan, np.nan)
            rows.append({
                "checkpoint": ckpt, "level": "question-aggregate",
                "outcome": outcome.replace("mean_", ""), "n": len(valid),
                "pearson_r": pear[0], "pearson_p": pear[1],
                "spearman_rho": spear[0], "spearman_p": spear[1],
            })
    cv = pd.DataFrame(rows)
    cv.to_csv(out_dir / "confidence_vs_outcome.csv", index=False)

    # Scatter: per-question mean confidence vs mean flip / shift / shift_norm
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    for col, ckpt in enumerate(STAGE_ORDER):
        sub = per_q[per_q["checkpoint_label"] == ckpt]
        if sub.empty:
            for row in (0, 1, 2):
                axes[row, col].set_visible(False)
            continue
        axes[0, col].scatter(sub["mean_t2_conf"], sub["mean_flip"],
                             s=12, alpha=0.5, color=STAGE_COLORS[ckpt])
        axes[0, col].set_title(f"{ckpt}: confidence → flip rate", fontsize=10)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].set_ylabel("question mean flip")
        axes[1, col].scatter(sub["mean_t2_conf"], sub["mean_shift"],
                             s=12, alpha=0.5, color=STAGE_COLORS[ckpt])
        axes[1, col].set_title(f"{ckpt}: confidence → prob shift", fontsize=10)
        axes[1, col].set_ylabel("question mean shift")
        axes[2, col].scatter(sub["mean_t2_conf"], sub["mean_shift_norm"],
                             s=12, alpha=0.5, color=STAGE_COLORS[ckpt])
        axes[2, col].set_title(f"{ckpt}: confidence → bound-normalized shift",
                               fontsize=10)
        axes[2, col].set_ylabel("question mean shift_norm")
        axes[2, col].set_ylim(-1.05, 1.05)
        axes[2, col].axhline(0, color="gray", linestyle=":", linewidth=0.5)
        axes[2, col].set_xlabel("question mean T2 confidence (raw)")
    fig.suptitle("Pre-cue confidence (per question) vs post-cue outcome "
                 "(row 3: shift / max-possible-shift removes ceiling artifact)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_vs_outcome.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  confidence-vs-outcome: {len(cv)} entries; "
          f"note raw conf is unnormalized over A/B/C/D — cross-ckpt "
          f"comparisons should rely on Spearman ρ.")
    return cv


def confidence_vs_outcome_per_cue(df, out_dir):
    """Trial-level scatter, faceted as (checkpoint × cue). One dot per row
    of cell_summary (i.e., per (checkpoint, question, cue) trial), so each
    panel holds ~N_questions trials. Useful for spotting cues whose
    confidence-outcome relationship differs from the pooled pattern."""
    # Order cues by checkpoint-pooled mean flip (ascending = least effective
    # → most effective), matching per_cue_effectiveness_ranking.png.
    cue_flip = (df.groupby(["checkpoint_label", "cue_id"])["flip"].mean()
                  .groupby("cue_id").mean())
    cues = cue_flip.sort_values().index.tolist()
    cue_labels = {c: df[df["cue_id"] == c]["cue_strength_label"].iloc[0]
                  for c in cues}
    n_cues, n_ck = len(cues), len(STAGE_ORDER)

    for outcome, ylabel, jitter, ylim in [
        ("flip",       "T4 flipped (1=yes)",                0.06, (-0.15, 1.15)),
        ("shift",      "T4 - T2 prob(correct)",             None, None),
        ("shift_norm", "shift / max-possible-shift",        None, (-1.05, 1.05)),
    ]:
        fig, axes = plt.subplots(n_ck, n_cues,
                                 figsize=(1.05 * n_cues, 1.8 * n_ck),
                                 sharex=True, sharey=True)
        rng = np.random.default_rng(0)
        for r, ckpt in enumerate(STAGE_ORDER):
            for c, cue in enumerate(cues):
                ax = axes[r, c]
                sub = df[(df["checkpoint_label"] == ckpt) &
                         (df["cue_id"] == cue)]
                y = sub[outcome].values
                if jitter is not None:
                    y = y + rng.uniform(-jitter, jitter, size=len(y))
                ax.scatter(sub["confidence"].values, y,
                           s=4, alpha=0.35, color=STAGE_COLORS[ckpt],
                           edgecolor="none")
                ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(f"{cue}\n{cue_labels[cue]}", fontsize=6)
                if c == 0:
                    ax.set_ylabel(f"{ckpt}", fontsize=8)
                if r == n_ck - 1:
                    ax.set_xlabel("T2 conf", fontsize=6)
                # per-cell Spearman (use raw values, ignore jitter)
                raw = df[(df["checkpoint_label"] == ckpt) &
                         (df["cue_id"] == cue)][["confidence", outcome]].dropna()
                if len(raw) > 5:
                    rho, _ = stats.spearmanr(raw["confidence"], raw[outcome])
                    ax.text(0.98, 0.98, f"ρ={rho:+.2f}\nn={len(raw)}",
                            transform=ax.transAxes, fontsize=5,
                            ha="right", va="top",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="white", alpha=0.7,
                                      edgecolor="none"))
        if ylim is not None:
            axes[0, 0].set_ylim(*ylim)
        for ax in axes[:, 0]:
            ax.set_ylabel(ax.get_ylabel() + f"\n{ylabel}", fontsize=7)
        fig.suptitle(
            f"Trial-level: T2 confidence vs {outcome} — "
            f"per checkpoint × per cue (one dot per trial; "
            f"{'binary outcome jittered for visibility' if jitter else 'continuous outcome'})",
            fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fname = f"confidence_vs_outcome_per_cue_{outcome}.png"
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")


# ── 7. Cue clustering ─────────────────────────────────────────────────────

def cue_clustering(df, out_dir, n_clusters=4):
    """Hierarchical clustering of cues by their question-response pattern.
    For each checkpoint, build a [n_cues × n_questions] matrix of the chosen
    outcome; pool the four checkpoint matrices side-by-side so clusters
    reflect a checkpoint-pooled pattern. Use correlation distance + average
    linkage. Emits parallel results for `shift` and `shift_norm`."""
    results = {}
    for outcome in ("shift", "shift_norm"):
        suffix = "" if outcome == "shift" else "_shift_norm"
        pieces = []
        for ckpt in STAGE_ORDER:
            sub = df[df["checkpoint_label"] == ckpt]
            if sub.empty:
                continue
            mat = sub.pivot_table(index="cue_id", columns="question_id",
                                  values=outcome, aggfunc="mean")
            mat.columns = [f"{ckpt}::{q}" for q in mat.columns]
            pieces.append(mat)
        if not pieces:
            print(f"  cue_clustering[{outcome}]: no data")
            continue
        big = pd.concat(pieces, axis=1).dropna(axis=1, how="any")
        if big.shape[1] < 5:
            print(f"  cue_clustering[{outcome}]: not enough complete columns")
            continue

        cue_ids = big.index.tolist()
        dist = pdist(big.values, metric="correlation")
        Z = linkage(dist, method="average")
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
        cluster_df = pd.DataFrame({"cue_id": cue_ids, "cluster": clusters})
        cue_meta = (df.drop_duplicates("cue_id")
                      [["cue_id", "cue_strength", "cue_strength_label"]])
        cluster_df = (cluster_df.merge(cue_meta, on="cue_id")
                                .sort_values(["cluster", "cue_strength"]))
        cluster_df.to_csv(out_dir / f"cue_clusters{suffix}.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"{cid} [{lbl}]"
                  for cid, lbl in zip(cue_ids,
                      df.drop_duplicates("cue_id").set_index("cue_id")
                        ["cue_strength_label"].reindex(cue_ids).values)]
        dendrogram(Z, labels=labels, orientation="right",
                   leaf_font_size=8, color_threshold=Z[-(n_clusters-1), 2])
        ax.set_xlabel("correlation distance")
        ax.set_title(f"Cue clustering (avg-linkage on {outcome} patterns, "
                     f"pooled across {len(pieces)} checkpoints, k={n_clusters})")
        fig.tight_layout()
        fig.savefig(out_dir / f"cue_clusters_dendrogram{suffix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        per_cue_ck = (df.groupby(["cue_id", "checkpoint_label"])[outcome]
                        .mean().unstack().reindex(columns=STAGE_ORDER))
        per_cue_ck = per_cue_ck.reindex(cluster_df["cue_id"].tolist())
        absmax = float(np.nanmax(np.abs(per_cue_ck.values)))
        fig, ax = plt.subplots(figsize=(6, max(5, 0.35 * len(cue_ids))))
        im = ax.imshow(per_cue_ck.values, cmap="RdBu_r",
                       vmin=-absmax, vmax=absmax, aspect="auto")
        ax.set_xticks(range(len(per_cue_ck.columns)))
        ax.set_xticklabels(per_cue_ck.columns)
        ax.set_yticks(range(len(per_cue_ck.index)))
        ax.set_yticklabels([f"{c} ({lbl}) [cl{cl}]"
                            for c, lbl, cl in zip(cluster_df["cue_id"],
                                cluster_df["cue_strength_label"],
                                cluster_df["cluster"])], fontsize=8)
        fig.colorbar(im, ax=ax, label=f"mean {outcome}")
        ax.set_title(f"Cue × checkpoint mean {outcome} (cues grouped by cluster)")
        fig.tight_layout()
        fig.savefig(out_dir / f"cue_by_checkpoint_heatmap{suffix}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  cue clustering[{outcome}]: {n_clusters} clusters over "
              f"{len(cue_ids)} cues, {big.shape[1]} complete cols")
        results[outcome] = cluster_df
    return results


# ── 8. Cue-strength validation ────────────────────────────────────────────

def cue_strength_validation(per_cue, out_dir):
    rows = []
    for ckpt in per_cue["checkpoint_label"].unique():
        sub = per_cue[per_cue["checkpoint_label"] == ckpt]
        for outcome in ("mean_flip", "mean_shift", "mean_shift_norm"):
            valid = sub[["cue_strength", outcome]].dropna()
            if len(valid) > 2:
                rho, p = stats.spearmanr(valid["cue_strength"], valid[outcome])
            else:
                rho, p = np.nan, np.nan
            rows.append({"checkpoint": ckpt, "outcome": outcome,
                         "spearman_rho": rho, "p_value": p,
                         "n_cues": len(valid)})
    val = pd.DataFrame(rows)
    val.to_csv(out_dir / "cue_strength_validation.csv", index=False)

    # Scatter: cue_strength vs mean effectiveness per ckpt
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True)
    for col, ckpt in enumerate(STAGE_ORDER):
        sub = per_cue[per_cue["checkpoint_label"] == ckpt]
        if sub.empty:
            for r in (0, 1, 2):
                axes[r, col].set_visible(False)
            continue
        axes[0, col].scatter(sub["cue_strength"], sub["mean_flip"],
                             color=STAGE_COLORS[ckpt], s=40, edgecolor="black")
        for _, r in sub.iterrows():
            axes[0, col].annotate(r["cue_id"],
                (r["cue_strength"], r["mean_flip"]), fontsize=6,
                xytext=(3, 3), textcoords="offset points")
        axes[0, col].set_title(f"{ckpt}: cue_strength → flip")
        axes[0, col].set_ylabel("mean flip"); axes[0, col].set_ylim(0, 1)
        axes[0, col].axvline(10.5, color="gray", linestyle=":", linewidth=0.7)

        axes[1, col].scatter(sub["cue_strength"], sub["mean_shift"],
                             color=STAGE_COLORS[ckpt], s=40, edgecolor="black")
        axes[1, col].set_ylabel("mean shift")
        axes[1, col].axvline(10.5, color="gray", linestyle=":", linewidth=0.7)

        axes[2, col].scatter(sub["cue_strength"], sub["mean_shift_norm"],
                             color=STAGE_COLORS[ckpt], s=40, edgecolor="black")
        axes[2, col].set_xlabel("cue_strength (1-10 ordinal; 11-19 categorical)")
        axes[2, col].set_ylabel("mean shift_norm")
        axes[2, col].set_ylim(-1.05, 1.05)
        axes[2, col].axhline(0, color="gray", linestyle=":", linewidth=0.5)
        axes[2, col].axvline(10.5, color="gray", linestyle=":", linewidth=0.7)
    fig.suptitle("Hand-coded cue_strength vs observed effectiveness",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "cue_strength_validation.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  cue strength validation: Spearman ρ in cue_strength_validation.csv")
    return val


# ── 9. Question × Cue interaction (descriptive) ───────────────────────────

def question_by_cue(df, balanced_qids, out_dir):
    """With one observation per (q, c, ckpt), the interaction is unidentified
    from residual variance — we cannot estimate within-cell SD. So we just
    visualize the residuals from the additive (question + cue) model as a
    heatmap, restricted to the balanced subset. Big residuals point at
    (q, c) cells worth replicating with additional seeds."""
    sub = df[df["question_id"].isin(balanced_qids)].copy()
    if sub.empty:
        print("  q×c: balanced subset empty")
        return None

    cue_order = sorted(df["cue_id"].unique())
    q_order = sorted(balanced_qids)

    top_combined = []
    for outcome in ("shift", "shift_norm"):
        g_grand = sub[outcome].mean()
        alpha = sub.groupby("question_id")[outcome].mean() - g_grand
        beta = sub.groupby("cue_id")[outcome].mean() - g_grand
        pred_col = f"additive_pred_{outcome}"
        resid_col = f"residual_{outcome}"
        sub[pred_col] = (g_grand
                         + sub["question_id"].map(alpha).values
                         + sub["cue_id"].map(beta).values)
        sub[resid_col] = sub[outcome] - sub[pred_col]

        fig, axes = plt.subplots(1, 4, figsize=(20, max(6, 0.05 * len(balanced_qids))),
                                 sharey=True)
        vmax = float(np.abs(sub[resid_col].quantile([0.01, 0.99])).max())
        for ax, ckpt in zip(axes, STAGE_ORDER):
            ck_sub = sub[sub["checkpoint_label"] == ckpt]
            if ck_sub.empty:
                ax.set_visible(False); continue
            mat = ck_sub.pivot_table(index="question_id", columns="cue_id",
                                     values=resid_col, aggfunc="mean")
            mat = mat.reindex(index=q_order, columns=cue_order)
            im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(cue_order)))
            ax.set_xticklabels(cue_order, rotation=90, fontsize=7)
            ax.set_title(f"{ckpt}\nresid = {outcome} − (μ + α_q + β_c)",
                         fontsize=10)
            if ax is axes[0]:
                ax.set_yticks([])
                ax.set_ylabel(f"{len(q_order)} balanced questions")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     label=f"residual {outcome}")
        fig.suptitle(
            f"Q×C interaction diagnostic on `{outcome}` "
            "(n=1 per cell → descriptive only). "
            "Bright cells = where additive model fails; candidates for replication.",
            fontsize=11)
        fig.savefig(out_dir / f"q_by_c_interaction_residuals_{outcome}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        sub_out = sub[["checkpoint_label", "question_id", "cue_id",
                       outcome, pred_col, resid_col]].copy()
        sub_out["outcome"] = outcome
        sub_out = sub_out.rename(columns={
            outcome: "value", pred_col: "additive_pred", resid_col: "residual"})
        top_combined.append(sub_out.reindex(
            sub_out["residual"].abs().sort_values(ascending=False).index).head(200))

    top = pd.concat(top_combined, ignore_index=True)
    top.to_csv(out_dir / "q_by_c_top_residuals.csv", index=False)
    print(f"  q×c interaction (descriptive): "
          f"{len(balanced_qids)} balanced questions, top 200 residuals saved")
    return top


# ── README dump ───────────────────────────────────────────────────────────

def write_readme(out_dir, args, n_rows, n_q, n_c):
    txt = textwrap.dedent(f"""\
        v6.0 filtered-trial tier-1 analysis
        ===================================

        Input:  {args.input}
                ({n_rows} rows; {n_q} unique questions; {n_c} unique cues;
                 balanced across 4 checkpoints)

        Outcomes:
          flip       = 1 - t4_accuracy  (binary)
          shift      = prob_correct_shift_mean  (continuous, t4 - t2)
          shift_norm = bound-normalized shift, in [-1, 1]:
                         shift / (1 - t2_prob)   if shift >= 0
                         shift / t2_prob         if shift < 0
                       Removes the mechanical ceiling/floor artifact: a
                       high-confidence trial can't move up much; a
                       low-confidence trial can't move down much. Pair
                       with `shift` rather than replacing it.
          confidence = t2_prob_correct_mean (raw correct-letter prob;
                       NOT normalized over A/B/C/D — see goal-6 caveats)

        Files in this directory:
          coverage_summary.txt              — overall coverage stats
          coverage_per_question.csv         — cue count per (q, ckpt)
          coverage_by_domain.csv            — balanced-subset bias check
          balanced_subset_question_ids.csv  — ≥{args.balanced_min_cues}-cue questions
          coverage_heatmap.png

          variance_decomposition.csv/.png   — η² per factor per ckpt per outcome

          per_question_profile.csv          — (ckpt × q) summary
          per_question_resistance_ranking.png

          per_cue_profile.csv               — (ckpt × c) summary
          per_cue_effectiveness_ranking.png

          rank_stability.csv/.png           — Spearman ρ across ckpts

          confidence_vs_outcome.csv/.png    — trial + per-question correlations

          cue_clusters.csv / cue_clusters_shift_norm.csv
          cue_clusters_dendrogram.png / cue_clusters_dendrogram_shift_norm.png
          cue_by_checkpoint_heatmap.png / cue_by_checkpoint_heatmap_shift_norm.png
                                            — empirical cue groups
                                              (raw + bound-normalized)

          cue_strength_validation.csv/.png  — does hand-coded strength match
                                              observed effectiveness?

          q_by_c_interaction_residuals_shift.png
          q_by_c_interaction_residuals_shift_norm.png
                                            — residuals from additive model
                                              (raw + bound-normalized)
          q_by_c_top_residuals.csv          — biggest deviations (replicate?)

        Caveats:
          * Each (q, c, ckpt) cell has count=1. Interaction terms are
            unidentified from residual variance — Q×C is descriptive only.
          * `confidence` is the raw probability assigned to the correct
            letter token, not normalized over answer options. Calibration
            may differ across checkpoints; trust Spearman ρ over Pearson r
            when comparing across stages.
          * Variance-decomposition factors are estimated independently and
            overlap (questions and domains are nested; cue and cue_strength
            also overlap). Treat η² as "how much variation aligns with this
            factor," not as orthogonal partitions.
        """)
    (out_dir / "README.txt").write_text(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv")
    ap.add_argument("--output_dir", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/filtered_followup")
    ap.add_argument("--balanced_min_cues", type=int, default=18,
        help="Min cues per question (in EVERY checkpoint) to enter the "
             "balanced subset used for the Q×C interaction diagnostic.")
    ap.add_argument("--n_clusters", type=int, default=4)
    args = ap.parse_args()

    in_path = REPO_ROOT / args.input if not Path(args.input).is_absolute() \
              else Path(args.input)
    out_dir = REPO_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() \
              else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path}")
    df = load_cells(in_path)
    print(f"  {len(df)} rows / "
          f"{df['question_id'].nunique()} questions / "
          f"{df['cue_id'].nunique()} cues / "
          f"{df['checkpoint_label'].nunique()} checkpoints")

    print("\n=== 1. Coverage ===")
    balanced_qids = coverage(df, out_dir, min_cues=args.balanced_min_cues)

    print("\n=== 2. Variance decomposition ===")
    variance_decomposition(df, out_dir)

    print("\n=== 3. Per-question resistance profile ===")
    per_q = per_question_profile(df, out_dir)

    print("\n=== 4. Per-cue effectiveness profile ===")
    per_c = per_cue_profile(df, out_dir)

    print("\n=== 5. Rank stability ===")
    rank_stability(per_q, per_c, out_dir)

    print("\n=== 6. Confidence vs outcome ===")
    confidence_vs_outcome(df, per_q, out_dir)
    confidence_vs_outcome_per_cue(df, out_dir)

    print("\n=== 7. Cue clustering ===")
    cue_clustering(df, out_dir, n_clusters=args.n_clusters)

    print("\n=== 8. Cue-strength validation ===")
    cue_strength_validation(per_c, out_dir)

    print("\n=== 9. Q×C interaction (descriptive) ===")
    question_by_cue(df, balanced_qids, out_dir)

    write_readme(out_dir, args, len(df), df["question_id"].nunique(),
                 df["cue_id"].nunique())
    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
