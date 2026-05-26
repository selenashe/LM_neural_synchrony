#!/usr/bin/env python3
"""Sample 300 trials (100 questions × 3 cues, paired) for the TRAK SFT
attribution study.

Design:
  * 3 cues at rank 1 / 10 / 19 in `per_cue_effectiveness_ranking.png` order
    (ascending checkpoint-pooled mean_flip): least / median / most effective.
  * 100 questions, stratified into 4 quartiles of SFT-checkpoint
    `t2_prob_correct_mean` (25 per quartile, with a fixed seed).
  * Every selected question is paired with all 3 cues → 300 trials.

Constraints:
  * Question must survive in all 4 checkpoints for all 3 selected cues
    (intersection of (q, cue) presence across checkpoints).
  * Trial must be in the filtered cell_summary (T2 correct, T4 parseable,
    pair surviving all 4 checkpoints).

Output:
  trial_manifest.csv  — 300 rows, schema:
      trial_id, question_id, cue_id, cue_rank (low|med|high), env_id,
      question_domain, cue_strength, cue_strength_label, correct_letter,
      cued_wrong_letter, t2_conf_sft, t2_conf_quartile,
      sft_flip, sft_shift, sft_shift_norm,
      base_flip, base_shift, base_shift_norm,
      gain_flip (sft - base), gain_shift_norm (sft - base)

Run:
  python analysis/trak/sample_trials.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell_summary", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv")
    ap.add_argument("--raw_results_dir", type=str,
        default="sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot",
        help="Dir holding results_<checkpoint>_seed0.csv (needed to recover "
             "env_id and the exact cued_wrong_letter per trial).")
    ap.add_argument("--out_dir", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak")
    ap.add_argument("--n_questions", type=int, default=100)
    ap.add_argument("--n_quartiles", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cs_path = REPO_ROOT / args.cell_summary

    print(f"Loading {cs_path}")
    cs = pd.read_csv(cs_path)
    assert {"checkpoint_label", "question_id", "cue_id", "t2_prob_correct_mean",
            "prob_correct_shift_mean", "t4_accuracy", "cue_strength",
            "cue_strength_label", "question_domain"}.issubset(cs.columns)

    # --- 1) Pick 3 cues by checkpoint-pooled mean flip (low / med / high) ---
    cue_rank = (cs.groupby(["checkpoint_label", "cue_id"])
                  ["t4_accuracy"].mean().groupby("cue_id")
                  .apply(lambda s: 1.0 - s.mean())  # mean flip across ckpts
                  .sort_values())
    cues_sorted = cue_rank.index.tolist()
    n_cues = len(cues_sorted)
    idx_low = 0
    idx_high = n_cues - 1
    idx_med = n_cues // 2
    selected_cues = {
        "low":  cues_sorted[idx_low],
        "med":  cues_sorted[idx_med],
        "high": cues_sorted[idx_high],
    }
    print(f"\n  cue ranking ({n_cues} cues, ascending checkpoint-pooled flip):")
    for r, c in enumerate(cues_sorted):
        marker = ""
        for label, cid in selected_cues.items():
            if cid == c:
                marker = f"  <-- {label}"
        print(f"    rank {r:>2}  {c}  flip={cue_rank[c]:.3f}{marker}")

    # --- 2) Find questions where all 3 cues survived in all 4 ckpts ---
    ckpts = sorted(cs["checkpoint_label"].unique())  # Base DPO RLVR SFT
    pair_sets_per_ckpt = [
        set(map(tuple, cs[cs["checkpoint_label"] == ck]
                       [["question_id", "cue_id"]].itertuples(index=False, name=None)))
        for ck in ckpts
    ]
    # Per-cue: questions present in every checkpoint for that cue
    qids_per_cue = []
    for cid in selected_cues.values():
        q_in_all_ck = set.intersection(
            *[{q for (q, c) in ps if c == cid} for ps in pair_sets_per_ckpt])
        qids_per_cue.append(q_in_all_ck)
    candidate_qids = sorted(set.intersection(*qids_per_cue))
    print(f"\n  candidate questions surviving all 3 cues × all 4 ckpts: "
          f"{len(candidate_qids)}")

    # --- 3) Per-question SFT T2 confidence (for stratification) ---
    sft = cs[cs["checkpoint_label"] == "SFT"]
    q_conf = (sft[sft["question_id"].isin(candidate_qids)]
                .groupby("question_id")["t2_prob_correct_mean"].mean()
                .reindex(candidate_qids))
    if q_conf.isna().any():
        miss = q_conf[q_conf.isna()].index.tolist()
        print(f"  WARNING: dropping {len(miss)} questions with no SFT t2 conf")
        q_conf = q_conf.dropna()

    # Quartile labels 0..n_quartiles-1
    quartile = pd.qcut(q_conf, args.n_quartiles, labels=False, duplicates="drop")
    rng = np.random.default_rng(args.seed)
    per_quartile = args.n_questions // args.n_quartiles
    leftover = args.n_questions - per_quartile * args.n_quartiles

    chosen_qids = []
    chosen_quartile = []
    for q in sorted(quartile.unique()):
        bucket = q_conf.index[quartile == q].tolist()
        take = per_quartile + (1 if leftover > 0 else 0)
        leftover = max(0, leftover - 1)
        if len(bucket) < take:
            print(f"  WARNING: quartile {q} has only {len(bucket)} questions "
                  f"(need {take}); taking all")
            take = len(bucket)
        picks = rng.choice(bucket, size=take, replace=False)
        chosen_qids.extend(picks.tolist())
        chosen_quartile.extend([int(q)] * len(picks))

    print(f"\n  chose {len(chosen_qids)} questions across "
          f"{args.n_quartiles} SFT-T2-confidence quartiles")
    quartile_summary = pd.Series(chosen_quartile).value_counts().sort_index()
    print(f"  per-quartile counts: {quartile_summary.to_dict()}")
    print(f"  T2 conf range: [{q_conf[chosen_qids].min():.3f}, "
          f"{q_conf[chosen_qids].max():.3f}], "
          f"mean={q_conf[chosen_qids].mean():.3f}")

    # --- 4) Build 300-row trial manifest ---
    # Need env_id and (per-trial) cued_wrong_letter from raw results CSV
    raw_sft = pd.read_csv(REPO_ROOT / args.raw_results_dir
                          / "results_OLMo-3-7B-Instruct-SFT_seed0.csv")
    raw_base = pd.read_csv(REPO_ROOT / args.raw_results_dir
                           / "results_OLMo-3-1025-7B_seed0.csv")

    # Cell-summary rows for the chosen (q, cue, ckpt) cells
    rows = []
    for q, qrt in zip(chosen_qids, chosen_quartile):
        for cue_rank_label, cid in selected_cues.items():
            sft_row = cs[(cs["checkpoint_label"] == "SFT")
                         & (cs["question_id"] == q)
                         & (cs["cue_id"] == cid)]
            base_row = cs[(cs["checkpoint_label"] == "Base")
                          & (cs["question_id"] == q)
                          & (cs["cue_id"] == cid)]
            if sft_row.empty or base_row.empty:
                print(f"  WARNING: missing cell for q={q}, cue={cid} "
                      f"(sft empty={sft_row.empty} base empty={base_row.empty})")
                continue
            sft_row = sft_row.iloc[0]; base_row = base_row.iloc[0]

            # Recover env_id + cued letter from raw SFT result
            raw = raw_sft[(raw_sft["question_id"] == q) & (raw_sft["cue_id"] == cid)]
            if raw.empty:
                print(f"  WARNING: raw SFT missing q={q} cue={cid}; skipping")
                continue
            raw = raw.iloc[0]
            rows.append({
                "trial_id":            f"t{len(rows):04d}",
                "question_id":         q,
                "cue_id":              cid,
                "cue_rank":            cue_rank_label,
                "env_id":              raw["env_id"],
                "question_domain":     raw["question_domain"],
                "cue_strength":        int(raw["cue_strength"]),
                "cue_strength_label":  raw["cue_strength_label"],
                "correct_letter":      raw["correct_position"],
                "cued_wrong_letter":   raw["wrong_position"],
                "t2_conf_sft":         float(q_conf[q]),
                "t2_conf_quartile":    int(qrt),
                "sft_flip":            1.0 - float(sft_row["t4_accuracy"]),
                "sft_shift":           float(sft_row["prob_correct_shift_mean"]),
                "base_flip":           1.0 - float(base_row["t4_accuracy"]),
                "base_shift":          float(base_row["prob_correct_shift_mean"]),
            })

    man = pd.DataFrame(rows)

    # Bound-normalized shifts (same formula as filtered followup)
    def _norm(shift, conf):
        max_up = 1.0 - conf
        max_down = conf
        denom = np.where(shift >= 0, max_up, max_down)
        denom = np.where(denom > 1e-9, denom, np.nan)
        return np.clip(shift / denom, -1.0, 1.0)

    # For norm we need per-trial t2_prob_correct_mean from each ckpt (cell value).
    # Pull from cs in one shot
    sft_lookup = cs[cs["checkpoint_label"] == "SFT"].set_index(
        ["question_id", "cue_id"])["t2_prob_correct_mean"]
    base_lookup = cs[cs["checkpoint_label"] == "Base"].set_index(
        ["question_id", "cue_id"])["t2_prob_correct_mean"]
    keys = list(zip(man["question_id"], man["cue_id"]))
    sft_conf_v = np.array([sft_lookup.get(k, np.nan) for k in keys])
    base_conf_v = np.array([base_lookup.get(k, np.nan) for k in keys])
    man["sft_shift_norm"] = _norm(man["sft_shift"].values, sft_conf_v)
    man["base_shift_norm"] = _norm(man["base_shift"].values, base_conf_v)
    man["gain_flip"] = man["sft_flip"] - man["base_flip"]
    man["gain_shift_norm"] = man["sft_shift_norm"] - man["base_shift_norm"]

    out_csv = out_dir / "trial_manifest.csv"
    man.to_csv(out_csv, index=False)
    print(f"\nWrote {len(man)} trials to {out_csv.relative_to(REPO_ROOT)}")
    print(f"  cue counts:        {man['cue_id'].value_counts().to_dict()}")
    print(f"  quartile counts:   {man['t2_conf_quartile'].value_counts().sort_index().to_dict()}")
    print(f"\nSFT vs Base flip rate by cue (sanity):")
    print(man.groupby("cue_rank")[["base_flip", "sft_flip",
                                   "gain_flip", "gain_shift_norm"]].mean().round(3))

    # Also write a tiny machine-readable selection record
    sel = {
        "cues": selected_cues,
        "cue_ranking_full": [(int(r), c, float(cue_rank[c]))
                             for r, c in enumerate(cues_sorted)],
        "n_questions": len(chosen_qids),
        "n_quartiles": args.n_quartiles,
        "seed": args.seed,
        "sft_t2_conf_quartile_edges": q_conf.quantile(
            np.linspace(0, 1, args.n_quartiles + 1)).round(4).to_dict(),
    }
    import json
    (out_dir / "trial_manifest_meta.json").write_text(
        json.dumps(sel, indent=2, default=str))
    print(f"\nWrote selection metadata to "
          f"{(out_dir / 'trial_manifest_meta.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
