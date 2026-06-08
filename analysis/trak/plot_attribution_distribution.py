#!/usr/bin/env python3
"""Distribution of per-cue mean attribution scores across the 50k SFT
examples, to inform which top-K is meaningful.

Two panels:

  Left  — sorted-by-rank curves, one per cue (colored by cue effectiveness
          rank, viridis = least → most effective). Reading: at rank K on
          the x-axis, the y-value is the K-th-highest mean attribution
          score for that cue. Where the curve plateaus is where the
          "tail" ends and the bulk begins — top-K is informative as long
          as it stays in the steep part of the curve. Reference vertical
          lines at K=100, 500, 5000.

  Right — pooled histogram of all per-cue mean attribution scores
          (n_cues × 50,000 ≈ 1M values). Shows the underlying
          distribution shape (long-tailed vs Gaussian-ish). The K=100/
          500/5000 reference lines mark the average score at that rank
          across cues — so you can see what attribution-score level you
          need to clear to be "in the top K."

Saved to analysis_outputs/.../analysis/attribution_distribution_by_cue_<tag>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot")
OUT_REPO = (REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis"
            / "score_distribution")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores_file", type=str,
                    default="attribution_scores_all_lam1e7.npy")
    ap.add_argument("--tag", type=str, default="all_lam1e7")
    ap.add_argument("--ks", type=int, nargs="+", default=[100, 500, 5000],
                    help="Reference K values to mark on the plot.")
    args = ap.parse_args()
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    scores = np.load(SCRATCH / "scores" / args.scores_file)
    trial_id_order = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/scores/trial_id_order.csv"
    )["trial_id"].tolist()
    manifest = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv")
    man_ord = manifest.set_index("trial_id").loc[trial_id_order].reset_index()
    assert len(man_ord) == scores.shape[0]

    ranking = json.loads(
        (REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak"
         / "trial_manifest_all_meta.json").read_text())["cue_ranking_full"]
    cues = [c for (_, c, _) in ranking
            if c in man_ord["cue_id"].unique()]
    n_cues = len(cues)
    n_sft = scores.shape[1]

    # Per-cue mean attribution, sorted descending.
    sorted_per_cue = np.zeros((n_cues, n_sft), dtype=np.float32)
    for i, c in enumerate(cues):
        mask = (man_ord["cue_id"] == c).values
        sorted_per_cue[i] = np.sort(scores[mask].mean(axis=0))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("viridis")
    ranks_x = np.arange(1, n_sft + 1)

    # ── Left: sorted curves, one per cue ──────────────────────────────
    for i in range(n_cues):
        ax1.plot(ranks_x, sorted_per_cue[i],
                 color=cmap(i / max(1, n_cues - 1)),
                 linewidth=1, alpha=0.85)
    ax1.axhline(0, color="black", linewidth=0.5)
    for k in args.ks:
        ax1.axvline(k, color="red", linestyle="--", linewidth=1, alpha=0.6)
    # Place K labels at the bottom so they don't overlap the curves
    y0, y1 = ax1.get_ylim()
    for k in args.ks:
        ax1.text(k, y0 + 0.02 * (y1 - y0), f" K={k}", color="red",
                 fontsize=9, ha="left", va="bottom")
    ax1.set_xscale("log")
    ax1.set_xlim(1, n_sft)
    ax1.set_xlabel("rank (1 = highest attribution per cue)")
    ax1.set_ylabel("mean attribution score (averaged across trials in cue)")
    ax1.set_title("Per-cue: sorted mean attribution score")
    ax1.grid(True, alpha=0.3, which="both")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels([f"{cues[0]} (least effective)",
                         f"{cues[n_cues // 2]} (middle)",
                         f"{cues[-1]} (most effective)"])
    cbar.ax.tick_params(labelsize=8)

    # ── Right: pooled histogram ───────────────────────────────────────
    pooled = sorted_per_cue.flatten()
    ax2.hist(pooled, bins=200, color="steelblue", edgecolor="white",
             linewidth=0.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("mean attribution score")
    ax2.set_ylabel("count (log scale)")

    # Mark mean score-at-rank for each K (across cues)
    y0_h, y1_h = ax2.get_ylim()
    for k in args.ks:
        score_at_k_per_cue = sorted_per_cue[:, k - 1]
        mean_score = float(score_at_k_per_cue.mean())
        ax2.axvline(mean_score, color="red", linestyle="--", linewidth=1,
                    alpha=0.6)
        ax2.text(mean_score, y1_h * 0.6,
                 f" K={k}\n  score≈{mean_score:.2g}",
                 color="red", fontsize=8, ha="left", va="top")

    # Show the median for context
    med = float(np.median(pooled))
    ax2.axvline(med, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.text(med, y1_h * 0.95, f" median", color="black", fontsize=8,
             ha="left", va="top")

    ax2.set_title(f"Pooled histogram ({n_cues}×{n_sft:,} ≈ "
                  f"{n_cues * n_sft / 1e6:.1f}M scores)")
    ax2.grid(True, alpha=0.3, which="major")

    # Per-cue quantile summary table on the side
    summary_lines = ["per-cue mean of score-at-rank K (across all cues):"]
    for k in args.ks:
        scores_at_k = sorted_per_cue[:, k - 1]
        summary_lines.append(
            f"  K={k:>5}: mean={scores_at_k.mean():.3g}  "
            f"min={scores_at_k.min():.3g}  max={scores_at_k.max():.3g}")
    summary_lines.append(f"  median across all scores: {med:.3g}")
    print("\n".join(summary_lines))

    fig.suptitle(
        f"Attribution score distribution per cue (file: {args.scores_file})",
        fontsize=12)
    fig.tight_layout()
    out = OUT_REPO / f"attribution_distribution_by_cue{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
