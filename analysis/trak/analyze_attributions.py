#!/usr/bin/env python3
"""First-pass TRAK attribution analysis for v6.0 flipping.

For each cue rank (low / med / high effectiveness), aggregate per-trial
attribution scores into a per-SFT-example mean, then:

  1. Source-dataset distribution of top-K SFT examples vs the 50k baseline
     (which sources are over-represented when ranked by attribution?).
  2. Same for bottom-K (sources that push the model AWAY from flipping).
  3. χ² test for whether top-K source distributions differ across cue ranks.
  4. Dump top-N SFT examples per cue rank to a markdown file for inspection.

Outputs go to analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot")
OUT_REPO = REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis"


def first_user_snippet(messages, max_chars=180):
    for m in messages:
        if str(m["role"]) == "user":
            txt = str(m["content"]).strip().replace("\n", " ")
            return (txt[:max_chars] + "…") if len(txt) > max_chars else txt
    return ""


def first_assistant_snippet(messages, max_chars=180):
    for m in messages:
        if str(m["role"]) == "assistant":
            txt = str(m["content"]).strip().replace("\n", " ")
            return (txt[:max_chars] + "…") if len(txt) > max_chars else txt
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top_k", type=int, default=100,
        help="K for top/bottom-K aggregate analysis")
    ap.add_argument("--top_n_examples", type=int, default=10,
        help="N examples per cue rank to dump for inspection")
    ap.add_argument("--scores_file", type=str, default="attribution_scores.npy",
        help="Filename under SCRATCH/scores/ to analyze. Use "
             "attribution_scores_norm.npy for the L2-normalized variant.")
    ap.add_argument("--tag", type=str, default="",
        help="Optional suffix for output filenames (e.g. 'norm') so multiple "
             "score variants don't overwrite each other.")
    args = ap.parse_args()
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # Load
    scores = np.load(SCRATCH / "scores" / args.scores_file)           # (300, 50000)
    print(f"Loaded scores from {args.scores_file}: {scores.shape}")
    sft_idx_order = np.load(SCRATCH / "scores/sft_idx_order.npy")         # (50000,)
    trial_id_order = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/scores/trial_id_order.csv"
    )["trial_id"].tolist()
    manifest = pd.read_csv(
        REPO_ROOT / "analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest.csv")
    sft = pd.read_parquet(SCRATCH / "sft_subsample.parquet")
    # Index sft by sft_idx so we can join on the column ordering used by scores
    sft = sft.set_index("sft_idx")

    # Manifest in scores row order
    man_ord = manifest.set_index("trial_id").loc[trial_id_order].reset_index()
    assert len(man_ord) == scores.shape[0]
    print(f"scores: {scores.shape}, sft cols: {len(sft_idx_order)}, "
          f"trials: {len(man_ord)}")

    # Baseline source-dataset distribution
    sft_aligned = sft.loc[sft_idx_order]
    base_src = sft_aligned["source_dataset"].value_counts(normalize=True)
    print(f"\nBaseline corpus: {len(base_src)} source datasets")

    # Per-cue-rank mean attribution
    results = {}
    for rank in ["low", "med", "high"]:
        mask = (man_ord["cue_rank"] == rank).values
        sub = scores[mask]                    # (100, 50000)
        mean_attr = sub.mean(axis=0)          # (50000,)
        order = np.argsort(mean_attr)
        bot = order[:args.top_k]
        top = order[-args.top_k:][::-1]
        top_src = sft_aligned.iloc[top]["source_dataset"].value_counts(normalize=True)
        bot_src = sft_aligned.iloc[bot]["source_dataset"].value_counts(normalize=True)
        results[rank] = {
            "mean_attr": mean_attr,
            "top_idx": top, "bot_idx": bot,
            "top_src": top_src, "bot_src": bot_src,
        }
        print(f"\n=== cue_rank={rank} (n_trials={mask.sum()}) ===")
        print(f"  mean_attr distribution: "
              f"mean={mean_attr.mean():.2f}, std={mean_attr.std():.2f}, "
              f"min={mean_attr.min():.1f}, max={mean_attr.max():.1f}")
        # Source over/under-representation in top-K
        top_lift = (top_src / base_src.reindex(top_src.index).fillna(1e-9))
        bot_lift = (bot_src / base_src.reindex(bot_src.index).fillna(1e-9))
        print(f"  Top-{args.top_k} source lift (top 5):")
        for s, l in top_lift.sort_values(ascending=False).head(5).items():
            n = int((sft_aligned.iloc[top]["source_dataset"] == s).sum())
            print(f"    {s:35s}  lift={l:5.2f}x  (n={n}, base={base_src.get(s, 0)*100:.1f}%)")
        print(f"  Bottom-{args.top_k} source lift (top 5):")
        for s, l in bot_lift.sort_values(ascending=False).head(5).items():
            n = int((sft_aligned.iloc[bot]["source_dataset"] == s).sum())
            print(f"    {s:35s}  lift={l:5.2f}x  (n={n}, base={base_src.get(s, 0)*100:.1f}%)")

    # Cross-rank χ² on top-K source distributions
    print("\n=== Cross-rank χ² test on top-K source distributions ===")
    all_top_sources = set()
    for r in ["low", "med", "high"]:
        all_top_sources |= set(results[r]["top_src"].index)
    src_list = sorted(all_top_sources)
    table = np.zeros((3, len(src_list)), dtype=int)
    for ri, r in enumerate(["low", "med", "high"]):
        top = results[r]["top_idx"]
        for si, s in enumerate(src_list):
            table[ri, si] = int((sft_aligned.iloc[top]["source_dataset"] == s).sum())
    chi2, p, dof, _ = chi2_contingency(table)
    print(f"  χ²={chi2:.2f}  dof={dof}  p={p:.2e}")
    print(f"  Distributions {'DIFFER' if p < 0.05 else 'do NOT differ'} "
          f"significantly across cue ranks (α=0.05).")

    # Save numeric summary
    summary = {
        "n_trials": int(scores.shape[0]),
        "n_sft": int(scores.shape[1]),
        "top_k": args.top_k,
        "baseline_top_sources": base_src.head(10).to_dict(),
        "per_rank": {
            r: {
                "n_trials": int((man_ord["cue_rank"] == r).sum()),
                "top_K_src_lift": (
                    results[r]["top_src"]
                    / base_src.reindex(results[r]["top_src"].index).fillna(1e-9)
                ).sort_values(ascending=False).head(10).to_dict(),
                "bot_K_src_lift": (
                    results[r]["bot_src"]
                    / base_src.reindex(results[r]["bot_src"].index).fillna(1e-9)
                ).sort_values(ascending=False).head(10).to_dict(),
                "mean_attr_stats": {
                    "mean": float(results[r]["mean_attr"].mean()),
                    "std": float(results[r]["mean_attr"].std()),
                    "max": float(results[r]["mean_attr"].max()),
                    "min": float(results[r]["mean_attr"].min()),
                },
            } for r in ["low", "med", "high"]
        },
        "cross_rank_chi2": {"chi2": float(chi2), "dof": int(dof), "p_value": float(p)},
    }
    summary_path = OUT_REPO / f"attribution_summary{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary → {summary_path.relative_to(REPO_ROOT)}")

    # Dump top-N examples per rank to markdown
    md = ["# Top-attributed SFT examples per cue rank\n",
          f"`top_n_examples = {args.top_n_examples}` (positive scores ⇒ training on this "
          "example pushes model toward picking the cued wrong letter at T4)\n"]
    for r in ["low", "med", "high"]:
        md.append(f"\n## cue_rank = {r}\n")
        top = results[r]["top_idx"][:args.top_n_examples]
        mean_attr = results[r]["mean_attr"]
        for rank_i, col in enumerate(top, 1):
            row = sft_aligned.iloc[col]
            sft_id = sft_idx_order[col]
            user = first_user_snippet(row["messages"])
            asst = first_assistant_snippet(row["messages"])
            md.append(
                f"\n### {rank_i}. sft_idx={sft_id} "
                f"(mean_attr={mean_attr[col]:.1f}, source={row['source_dataset']}, "
                f"domain={row['domain']})\n"
                f"- **user:** {user}\n"
                f"- **assistant:** {asst}\n"
            )
        md.append(f"\n### Bottom-{args.top_n_examples} (pushes AWAY from flipping)\n")
        bot = results[r]["bot_idx"][:args.top_n_examples]
        for rank_i, col in enumerate(bot, 1):
            row = sft_aligned.iloc[col]
            sft_id = sft_idx_order[col]
            user = first_user_snippet(row["messages"])
            md.append(
                f"\n{rank_i}. sft_idx={sft_id}  "
                f"mean_attr={mean_attr[col]:.1f}  source=`{row['source_dataset']}`  "
                f"domain=`{row['domain']}`\n"
                f"   user: {user}\n"
            )
    md_path = OUT_REPO / f"top_attributed_examples{tag}.md"
    md_path.write_text("".join(md))
    print(f"Wrote markdown → {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
