#!/usr/bin/env python3
"""Top-K attributed SFT examples per group: composition by source_dataset.

Two metrics:
  --metric count → stacked-bar (raw counts of top-K examples by source).
  --metric lift  → heatmap (log2 lift over baseline subsample share).

Lift normalizes for source prevalence: lift_S = (count_S_in_topK / K) /
(count_S_in_subsample / N). Lift > 1 = attribution favors this source for
this group; lift < 1 = disfavors. Useful when raw counts at large K are
dominated by the subsample's source-share baseline.

The `domain` field is no longer plotted (removed 2026-06 to focus on
source_dataset as the canonical attribution category). If domain-level
analysis is needed in the future, the script can be generalized again.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot")
OUT_REPO_BASE = REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis"


def _auto_subdir(tag, flip_split):
    """Pick the analysis subfolder based on tag and flags."""
    if flip_split:
        return "top_k_composition_by_flip_outcome"
    if "lam1e6" in tag or "lam1e8" in tag:
        return "top_k_composition_lambda_robustness"
    return "top_k_composition_canonical"


def _ordered_groups(man_ord, group_by):
    """Stable group ordering.

      cue_rank → ["low", "med", "high"]
      cue_id   → ascending checkpoint-pooled mean flip rate (least → most
                 effective at flipping), pulled from
                 trial_manifest_all_meta.json's `cue_ranking_full` field so
                 the order matches per_cue_effectiveness_ranking.png.
                 Falls back to alphabetical if the meta file isn't found.
      other    → alphabetical.
    """
    vals = list(man_ord[group_by].unique())
    if group_by == "cue_rank":
        order = ["low", "med", "high"]
        return [g for g in order if g in vals]
    if group_by == "cue_id":
        import json
        meta_path = (REPO_ROOT
                     / "analysis_outputs/v6_0_olmo3_7b_pilot/trak"
                     / "trial_manifest_all_meta.json")
        if meta_path.exists():
            ranking = json.loads(meta_path.read_text()).get("cue_ranking_full")
            if ranking:
                ordered = [c for (_, c, _) in ranking if c in vals]
                ordered += [c for c in sorted(vals) if c not in ordered]
                return ordered
    return sorted(vals)


def lift_heatmap(lift_per_group, groups, group_by, cats_ordered,
                 title, out_path, top_k, vmax_log2=4.0):
    """Heatmap of log2(lift) — sources × groups (groups on x-axis, sources
    on y-axis). Symmetric diverging colormap centered at log2(1)=0.

    Cells with lift==0 (source absent from top-K) render as -inf → displayed
    as the deepest blue. Cells with abs(log2 lift) > vmax_log2 clamp to the
    colorbar extremes.
    """
    n_g, n_s = len(groups), len(cats_ordered)
    mat = np.zeros((n_s, n_g), dtype=float)
    for j, g in enumerate(groups):
        for i, c in enumerate(cats_ordered):
            mat[i, j] = lift_per_group[g].get(c, 0.0)
    log_lift = np.where(mat > 0, np.log2(np.where(mat > 0, mat, 1.0)),
                        -np.inf)
    finite_max = np.nanmax(np.abs(np.where(np.isfinite(log_lift),
                                            log_lift, np.nan)))
    vmax = min(vmax_log2, max(1.0, finite_max))

    fig_w = max(8, 0.5 * n_g + 4)
    fig_h = max(5, 0.32 * n_s + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.clip(log_lift, -vmax, vmax), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_g))
    ax.set_xticklabels(groups,
                       rotation=45 if n_g > 6 else 0,
                       ha="right" if n_g > 6 else "center", fontsize=9)
    ax.set_yticks(np.arange(n_s))
    ax.set_yticklabels(cats_ordered, fontsize=8)
    ax.set_xlabel(group_by)

    for i in range(n_s):
        for j in range(n_g):
            v = mat[i, j]
            ll = log_lift[i, j]
            if not np.isfinite(ll):
                txt = "·"
            elif v >= 10:
                txt = f"{v:.0f}"
            elif v >= 1:
                txt = f"{v:.1f}"
            elif v >= 0.1:
                txt = f"{v:.2f}"
            else:
                txt = "<.1"
            color = "white" if (np.isfinite(ll) and abs(ll) > vmax * 0.55) \
                    else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("log₂(lift)   [lift = obs share / baseline share]",
                   fontsize=8)
    log_ticks = [-vmax, -vmax / 2, 0, vmax / 2, vmax]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{2**t:.2g}×" for t in log_ticks])

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def lift_heatmap_pair(lift_per_behavior, n_trials_per_behavior, groups,
                      group_by, cats_ordered, suptitle, out_path, top_k,
                      vmax_log2=4.0):
    """Side-by-side heatmaps for two behavior classes (flipped vs not-flipped).

    Shared color scale across both panels (vmax = max |log2 lift| across both,
    capped at vmax_log2). Shared y-axis (categories) — labels only on left
    panel. Single colorbar to the right of both panels.
    """
    panels = [("flipped", "flipped trials (sft_flip=1)"),
              ("notflipped", "not-flipped trials (sft_flip=0)")]
    n_g, n_s = len(groups), len(cats_ordered)

    mats, log_lifts = {}, {}
    for key, _ in panels:
        mat = np.zeros((n_s, n_g), dtype=float)
        for j, g in enumerate(groups):
            for i, c in enumerate(cats_ordered):
                mat[i, j] = lift_per_behavior[key][g].get(c, 0.0)
        mats[key] = mat
        log_lifts[key] = np.where(mat > 0, np.log2(np.where(mat > 0, mat, 1.0)),
                                   -np.inf)

    finite_max = max(
        np.nanmax(np.abs(np.where(np.isfinite(ll), ll, np.nan)))
        for ll in log_lifts.values())
    vmax = min(vmax_log2, max(1.0, finite_max))

    fig_w = max(13, 0.5 * n_g * 2 + 6)
    fig_h = max(5, 0.32 * n_s + 2)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)

    im = None
    for ax, (key, label) in zip(axes, panels):
        im = ax.imshow(np.clip(log_lifts[key], -vmax, vmax), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_g))
        n_by_g = n_trials_per_behavior[key]
        xlabels = [f"{g}\n(n={n_by_g[g]})" for g in groups]
        ax.set_xticklabels(xlabels,
                           rotation=45 if n_g > 6 else 0,
                           ha="right" if n_g > 6 else "center", fontsize=9)
        ax.set_xlabel(group_by)
        ax.set_title(label, fontsize=10)

        for i in range(n_s):
            for j in range(n_g):
                v = mats[key][i, j]
                ll = log_lifts[key][i, j]
                if not np.isfinite(ll):
                    txt = "·"
                elif v >= 10:
                    txt = f"{v:.0f}"
                elif v >= 1:
                    txt = f"{v:.1f}"
                elif v >= 0.1:
                    txt = f"{v:.2f}"
                else:
                    txt = "<.1"
                color = ("white" if (np.isfinite(ll) and abs(ll) > vmax * 0.55)
                         else "black")
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color=color)

    axes[0].set_yticks(np.arange(n_s))
    axes[0].set_yticklabels(cats_ordered, fontsize=8)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("log₂(lift)   [lift = obs share / baseline share]",
                   fontsize=8)
    log_ticks = [-vmax, -vmax / 2, 0, vmax / 2, vmax]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{2**t:.2g}×" for t in log_ticks])

    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def _lift_per_group(counts_per_group, baseline_share, top_k):
    """counts_per_group: {group: pd.Series of source→count in top-K}.
    baseline_share: pd.Series of source→share-of-subsample.
    Returns {group: {source: lift}} for every source in baseline_share."""
    out = {}
    for g, counts in counts_per_group.items():
        out[g] = {}
        for c, share in baseline_share.items():
            obs_frac = counts.get(c, 0) / top_k
            out[g][c] = obs_frac / share if share > 0 else np.nan
    return out


def stacked_bar(counts_per_group, groups, group_by, title, out_path, top_k):
    all_cats = sorted({c for d in counts_per_group.values() for c in d.index})
    totals = {c: sum(counts_per_group[g].get(c, 0) for g in groups)
              for c in all_cats}
    all_cats = sorted(all_cats, key=lambda c: -totals[c])

    cmap = plt.get_cmap("tab20" if len(all_cats) > 10 else "tab10")
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(all_cats)}

    fig_w = max(7, 0.55 * len(groups) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))
    for cat in all_cats:
        heights = np.array([counts_per_group[g].get(cat, 0) for g in groups],
                           dtype=float)
        ax.bar(x, heights, bottom=bottoms, label=cat, color=colors[cat],
               edgecolor="white", linewidth=0.5)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(groups,
                       rotation=45 if len(groups) > 6 else 0,
                       ha="right" if len(groups) > 6 else "center",
                       fontsize=8 if len(groups) > 6 else 10)
    ax.set_xlabel(group_by)
    ax.set_ylabel(f"# of top-{top_k} attributed SFT examples")
    ax.set_title(title)
    ax.set_ylim(0, top_k)
    ax.legend(title="source_dataset", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def _bucket_membership_str(man_ord, groups, group_by, n_trials_by_group):
    """Build a one-line legend showing the cue_id membership of each group.

    Returns "" for non-cue_rank group_by (where the group IS the cue_id and
    the x-axis already shows it). For cue_rank, returns e.g.
    "low: c11,c20,c13,c03,c02,c01 (n=205) | med: ... | high: ...".
    """
    if group_by != "cue_rank":
        return ""
    parts = []
    for g in groups:
        cues = sorted(man_ord[man_ord["cue_rank"] == g]["cue_id"].unique())
        parts.append(f"{g}: {','.join(cues)} (n={n_trials_by_group[g]})")
    return "  |  ".join(parts)


def _compute_and_plot(args, man_ord, scores, sft_aligned, out_dir,
                      behavior_name=None):
    """One scoring + plotting pass for source_dataset. behavior_name in
    {None, 'flipped', 'notflipped'} controls trial filtering by sft_flip
    and output naming."""
    if behavior_name == "flipped":
        behavior_mask = (man_ord["sft_flip"] == 1.0).values
        behavior_label = "flipped trials only (sft_flip=1)"
        behavior_suffix = "_flipped"
    elif behavior_name == "notflipped":
        behavior_mask = (man_ord["sft_flip"] == 0.0).values
        behavior_label = "not-flipped trials only (sft_flip=0)"
        behavior_suffix = "_notflipped"
    else:
        behavior_mask = np.ones(len(man_ord), dtype=bool)
        behavior_label = "all trials"
        behavior_suffix = ""

    groups = _ordered_groups(man_ord, args.group_by)
    src_counts = {}
    n_trials_by_group = {}
    for g in groups:
        mask = (man_ord[args.group_by] == g).values & behavior_mask
        n = int(mask.sum())
        n_trials_by_group[g] = n
        if n == 0:
            src_counts[g] = pd.Series(dtype=int)
            print(f"  {args.group_by}={g} ({behavior_label}): "
                  f"n_trials=0, skipping top-K")
            continue
        mean_attr = scores[mask].mean(axis=0)
        top = np.argsort(mean_attr)[-args.top_k:][::-1]
        rows = sft_aligned.iloc[top]
        src_counts[g] = rows["source_dataset"].value_counts()
        print(f"  {args.group_by}={g} (n_trials={n}, {behavior_label}): "
              f"top-{args.top_k} sources -> "
              f"{src_counts[g].head(3).to_dict()}")

    bucket_str = _bucket_membership_str(man_ord, groups, args.group_by,
                                         n_trials_by_group)
    tag = f"_{args.tag}" if args.tag else ""
    by_suffix = f"_by_{args.group_by}" if args.group_by != "cue_rank" else ""
    title_suffix = f" — {behavior_label}" if behavior_name else ""

    def _full_title(base):
        t = base + title_suffix
        if bucket_str:
            t = t + "\n" + bucket_str
        return t

    if args.metric == "count":
        stacked_bar(src_counts, groups, args.group_by,
            _full_title(f"Top-{args.top_k} attributed SFT examples by source_dataset"),
            out_dir / (f"top{args.top_k}_source_stacked{behavior_suffix}"
                       f"{by_suffix}{tag}.png"),
            args.top_k)
    else:  # lift
        src_baseline = (sft_aligned["source_dataset"].value_counts()
                        / len(sft_aligned))
        src_lift = _lift_per_group(src_counts, src_baseline, args.top_k)
        src_mean = {s: float(np.nanmean([src_lift[g][s] for g in groups]))
                    for s in src_baseline.index}
        src_ordered = sorted(src_baseline.index, key=lambda s: -src_mean[s])
        lift_heatmap(src_lift, groups, args.group_by, src_ordered,
            _full_title(f"Top-{args.top_k} attributed SFT examples — "
                        f"log2(lift over baseline) by source_dataset"),
            out_dir / (f"top{args.top_k}_source_lift_heatmap{behavior_suffix}"
                       f"{by_suffix}{tag}.png"),
            args.top_k)


def _compute_and_plot_paired(args, man_ord, scores, sft_aligned, out_dir):
    """Paired side-by-side lift heatmap (source_dataset): flipped on left,
    not-flipped on right. Single figure with shared color scale.
    Only used when --flip_split AND --metric lift.
    """
    groups = _ordered_groups(man_ord, args.group_by)
    behaviors = [("flipped", 1.0), ("notflipped", 0.0)]

    src_counts_by_b = {b: {} for b, _ in behaviors}
    n_trials_by_b = {b: {} for b, _ in behaviors}
    for behavior, flip_val in behaviors:
        b_mask = (man_ord["sft_flip"] == flip_val).values
        for g in groups:
            mask = (man_ord[args.group_by] == g).values & b_mask
            n = int(mask.sum())
            n_trials_by_b[behavior][g] = n
            if n == 0:
                src_counts_by_b[behavior][g] = pd.Series(dtype=int)
                continue
            mean_attr = scores[mask].mean(axis=0)
            top = np.argsort(mean_attr)[-args.top_k:][::-1]
            rows = sft_aligned.iloc[top]
            src_counts_by_b[behavior][g] = rows["source_dataset"].value_counts()
        print(f"  {behavior}: n_trials per {args.group_by} = "
              f"{n_trials_by_b[behavior]}")

    src_baseline = sft_aligned["source_dataset"].value_counts() / len(sft_aligned)
    src_lift_by_b = {b: _lift_per_group(src_counts_by_b[b], src_baseline, args.top_k)
                     for b, _ in behaviors}

    # Ordering: pool across both behaviors AND all groups, sort by mean lift
    # desc (consistent y-axis across panels).
    means = {}
    for c in src_baseline.index:
        vals = [src_lift_by_b[b][g][c] for b, _ in behaviors for g in groups]
        means[c] = float(np.nanmean(vals)) if vals else 0.0
    src_ordered = sorted(src_baseline.index, key=lambda c: -means[c])

    if args.group_by == "cue_rank":
        bucket_parts = []
        for g in groups:
            cues = sorted(man_ord[man_ord["cue_rank"] == g]["cue_id"].unique())
            bucket_parts.append(f"{g}: {','.join(cues)}")
        bucket_str = "  |  ".join(bucket_parts)
    else:
        bucket_str = ""

    tag = f"_{args.tag}" if args.tag else ""
    by_suffix = f"_by_{args.group_by}" if args.group_by != "cue_rank" else ""

    suptitle = (f"Top-{args.top_k} attributed SFT examples — "
                f"log2(lift over baseline) by source_dataset\n"
                f"flipped vs not-flipped trials, "
                f"grouped by {args.group_by}")
    if bucket_str:
        suptitle = suptitle + "\n" + bucket_str

    lift_heatmap_pair(src_lift_by_b, n_trials_by_b, groups, args.group_by,
        src_ordered, suptitle,
        out_dir / (f"top{args.top_k}_source_lift_heatmap_by_flip"
                   f"{by_suffix}{tag}.png"),
        args.top_k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--scores_file", type=str,
                    default="attribution_scores_all_lam1e7.npy")
    ap.add_argument("--tag", type=str, default="all_lam1e7")
    ap.add_argument("--group_by", type=str, default="cue_id",
                    choices=["cue_id", "cue_rank"],
                    help="Manifest column to group trials by. cue_id = one "
                         "bar per cue (20 bars for v6.0); cue_rank = low/"
                         "med/high buckets (3 bars).")
    ap.add_argument("--metric", type=str, default="count",
                    choices=["count", "lift"],
                    help="count = stacked-bar of raw top-K counts; "
                         "lift = heatmap of log2(lift) over baseline "
                         "subsample share.")
    ap.add_argument("--flip_split", action="store_true",
                    help="Produce per-behavior heatmap (flipped vs "
                         "not-flipped). With --metric lift, renders as a "
                         "single paired figure (flipped | not-flipped); "
                         "with --metric count, two separate stacked bars. "
                         "Recommend pairing with --group_by cue_rank: "
                         "low-flip cues have few flipped trials and vice "
                         "versa, so per-cue × per-behavior cells can be "
                         "too sparse; bucketed thirds give >190 trials per "
                         "(bucket, behavior) cell.")
    ap.add_argument("--out_subdir", type=str, default=None,
                    help="Analysis subfolder under analysis/. Auto-picked "
                         "from --tag and --flip_split if not given: "
                         "top_k_composition_canonical for lam1e7, "
                         "top_k_composition_lambda_robustness for lam1e6/1e8, "
                         "top_k_composition_by_flip_outcome for --flip_split.")
    args = ap.parse_args()

    subdir = args.out_subdir or _auto_subdir(args.tag, args.flip_split)
    out_dir = OUT_REPO_BASE / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.flip_split and args.group_by == "cue_id":
        print("WARNING: --flip_split with --group_by cue_id is sparse "
              "(low-flip cues have <20 flipped trials each). Recommend "
              "--group_by cue_rank for stable per-bucket × per-behavior "
              "cells. Continuing anyway.")

    scores = np.load(SCRATCH / "scores" / args.scores_file)
    sft_idx_order = np.load(SCRATCH / "scores/sft_idx_order.npy")
    trial_id_order = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/scores/trial_id_order.csv"
    )["trial_id"].tolist()
    manifest = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv")
    sft = pd.read_parquet(SCRATCH / "sft_subsample.parquet").set_index("sft_idx")
    man_ord = manifest.set_index("trial_id").loc[trial_id_order].reset_index()
    assert len(man_ord) == scores.shape[0]
    sft_aligned = sft.loc[sft_idx_order]

    if args.flip_split and args.metric == "lift":
        _compute_and_plot_paired(args, man_ord, scores, sft_aligned, out_dir)
    elif args.flip_split:
        for behavior in ("flipped", "notflipped"):
            print(f"\n=== {behavior} ===")
            _compute_and_plot(args, man_ord, scores, sft_aligned, out_dir,
                              behavior_name=behavior)
    else:
        _compute_and_plot(args, man_ord, scores, sft_aligned, out_dir,
                          behavior_name=None)


if __name__ == "__main__":
    main()
