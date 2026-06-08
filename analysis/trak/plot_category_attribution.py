#!/usr/bin/env python3
"""Per-trial category attribution share, averaged within flipped / not-flipped trials.

For each trial:
    category_inf[c] = sum(attr scores for SFT examples in category c)
                      / sum(attr scores across all 50k examples)

Then average per-category attribution share across trials within each
(cue_rank, sft_flip) group. One figure per cue_rank, two panels (flipped
vs not-flipped), one bar per category with SEM error bars.

NOTE: Until the group-removal counterfactual validates causality, treat
these as descriptive attribution shares, not as causal "influence."
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot")
OUT_REPO = (REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis"
            / "category_attribution_per_cue")


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


def per_trial_category_share(scores, sft_aligned, field):
    """Return DataFrame [n_trials × n_categories] of per-trial category
    attribution shares."""
    cats = sft_aligned[field].astype("category")
    cat_codes = cats.cat.codes.values             # (50000,)
    cat_names = list(cats.cat.categories)
    n_cat = len(cat_names)
    n_trials = scores.shape[0]
    # Per-category sum of attribution scores per trial: (n_trials, n_cat)
    sums = np.zeros((n_trials, n_cat), dtype=np.float64)
    for c in range(n_cat):
        col_mask = cat_codes == c
        sums[:, c] = scores[:, col_mask].sum(axis=1)
    denom = sums.sum(axis=1, keepdims=True)
    # Denominator can be ~0 or negative; replace 0 with NaN to avoid div-by-0
    denom_safe = np.where(denom == 0, np.nan, denom)
    inf = sums / denom_safe
    return pd.DataFrame(inf, columns=cat_names), denom.squeeze()


def plot_one_group(group_by, group_val, inf_df, trial_rows, denom, field,
                   cats_ordered, out_path, bucket_member_str=""):
    flipped_idx = trial_rows.index[trial_rows["sft_flip"] == 1.0]
    notflip_idx = trial_rows.index[trial_rows["sft_flip"] == 0.0]
    panes = [("flipped (sft_flip=1)", flipped_idx),
             ("not flipped (sft_flip=0)", notflip_idx)]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.35 * len(cats_ordered)), 8),
                             sharex=True)
    x = np.arange(len(cats_ordered))

    for ax, (title, idx) in zip(axes, panes):
        n = len(idx)
        sub = inf_df.loc[idx, cats_ordered]
        means = sub.mean(axis=0).values
        sems = sub.sem(axis=0).values  # nan-safe via pandas
        ax.bar(x, means, yerr=sems, capsize=2, color="steelblue",
               edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(f"{title}  (n={n} trials, mean Σattr={denom[idx].mean():.2g})")
        ax.set_ylabel("category attribution share\n(Σattr in cat / Σattr all)")
        # Reference: uniform-share line = 1 / n_cat
        ax.axhline(1.0 / len(cats_ordered), color="red", linestyle="--",
                   linewidth=0.6, label=f"uniform = 1/{len(cats_ordered)}")
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cats_ordered, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel(field)
    suptitle = (f"{group_by} = {group_val} — per-trial {field} attribution "
                f"share, averaged within behavior group")
    if bucket_member_str:
        suptitle = suptitle + "\n" + bucket_member_str
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores_file", type=str,
                    default="attribution_scores_all_lam1e7.npy")
    ap.add_argument("--tag", type=str, default="all_lam1e7")
    # Field is hardcoded to source_dataset (domain branch was removed
    # 2026-06; only source_dataset is plotted across the pipeline).
    ap.add_argument("--group_by", type=str, default="cue_id",
                    choices=["cue_id", "cue_rank"],
                    help="Manifest column to group trials by. cue_id = one "
                         "plot per cue (20 plots for v6.0); cue_rank = low/"
                         "med/high buckets (3 plots).")
    args = ap.parse_args()
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

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

    field = "source_dataset"
    inf_df, denom = per_trial_category_share(scores, sft_aligned, field)
    print(f"Built per-trial attribution shares: {inf_df.shape}; "
          f"denominator stats: mean={np.nanmean(denom):.3g}, "
          f"#trials with denom<=0: {(denom <= 0).sum()}")

    # Order categories by overall absolute mean attribution share (across all trials)
    overall = inf_df.mean(axis=0).abs().sort_values(ascending=False)
    cats_ordered = overall.index.tolist()

    for group_val in _ordered_groups(man_ord, args.group_by):
        trial_rows = man_ord[man_ord[args.group_by] == group_val]
        if args.group_by == "cue_rank":
            cues_in_bucket = sorted(trial_rows["cue_id"].unique())
            bucket_member_str = (f"{group_val} bucket: {','.join(cues_in_bucket)} "
                                 f"(n={len(trial_rows)} trials)")
        else:
            bucket_member_str = ""
        out = OUT_REPO / (f"category_attribution_{field}_"
                          f"{group_val}{tag}.png")
        plot_one_group(args.group_by, group_val, inf_df, trial_rows, denom,
                       field, cats_ordered, out, bucket_member_str)


if __name__ == "__main__":
    main()
