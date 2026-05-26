#!/usr/bin/env python3
"""Sample 50k examples from dolci-instruct-sft (the SFT corpus for
OLMo-3-7B-Instruct-SFT), stratified by `source_dataset`.

Strategy:
  * Allocate per-source quotas proportional to corpus prevalence (capped so
    no single source >25% of the subsample, to preserve tail coverage).
  * Sample without replacement within each source using a fixed seed.

Output:
  sft_subsample.parquet  — schema:
    sft_idx (0..n-1), shard_path (basename), shard_row_idx,
    source_dataset, domain, id, messages (list of {role, content})

  sft_subsample_meta.json  — summary

Run:
  python analysis/trak/sample_sft_subset.py --n 50000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SFT_DIR = Path("/juice6/scr6/nlp/interp-models/OLMo-3-7B/datasets/sft/dolci-instruct-sft/data")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out_dir", type=str,
        default="/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot",
        help="Where the (sizable) parquet lands. Default is scratch.")
    ap.add_argument("--meta_dir", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak",
        help="Where the small meta json lands (under REPO_ROOT).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source_cap_pct", type=float, default=0.25,
        help="Max share of any single source_dataset in the subsample")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = REPO_ROOT / args.meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(SFT_DIR.glob("train-*.parquet"))
    print(f"Scanning {len(files)} shards in {SFT_DIR}")

    # Pass 1: enumerate source_dataset per row across all shards.
    # Materializes ~2M rows × 3 small columns; fits easily in RAM.
    catalog = []
    for shard_idx, fp in enumerate(files):
        t = pq.read_table(fp, columns=["source_dataset", "domain", "id"])
        df = t.to_pandas()
        df["shard_path"] = fp.name
        df["shard_row_idx"] = np.arange(len(df), dtype=np.int64)
        catalog.append(df)
        print(f"  shard {shard_idx:>2}/{len(files)-1}: {len(df):,} rows ({fp.name})")
    cat = pd.concat(catalog, ignore_index=True)
    print(f"Total corpus rows: {len(cat):,}")
    src_counts = cat["source_dataset"].value_counts()
    print(f"Total source datasets: {len(src_counts)}")

    # Allocate per-source quotas
    cap = int(args.n * args.source_cap_pct)
    src_share = (src_counts / src_counts.sum())
    quota = (src_share * args.n).round().astype(int).clip(upper=cap)
    # Bump up under-target total by redistributing leftover to underfilled sources
    deficit = args.n - quota.sum()
    if deficit > 0:
        # Pour deficit proportionally into sources that still have capacity
        room = (src_counts - quota).clip(lower=0)
        room_share = (room / room.sum())
        bonus = (room_share * deficit).round().astype(int)
        quota = (quota + bonus).clip(upper=src_counts)
    if quota.sum() > args.n:
        # Trim from largest sources
        over = quota.sum() - args.n
        for s in quota.sort_values(ascending=False).index:
            take = min(over, quota[s] - 1)
            quota[s] -= take; over -= take
            if over == 0: break
    assert quota.sum() == args.n, f"quota={quota.sum()} != n={args.n}"

    print(f"\nQuota allocation (top 15 of {len(quota)}):")
    print(quota.sort_values(ascending=False).head(15).to_string())

    # Sample within each source
    rng = np.random.default_rng(args.seed)
    sampled_rows = []
    for src, k in quota.items():
        if k <= 0: continue
        pool = cat.index[cat["source_dataset"] == src].to_numpy()
        if len(pool) < k:
            print(f"  WARNING: source {src!r} pool={len(pool)} < quota={k}, taking all")
            k = len(pool)
        pick = rng.choice(pool, size=k, replace=False)
        sampled_rows.append(pick)
    pick_idx = np.concatenate(sampled_rows)
    rng.shuffle(pick_idx)  # global shuffle so SLURM shards aren't source-clumped

    subsample_idx = cat.loc[pick_idx, ["shard_path", "shard_row_idx",
                                       "source_dataset", "domain", "id"]
                            ].reset_index(drop=True)
    subsample_idx["sft_idx"] = np.arange(len(subsample_idx), dtype=np.int64)
    subsample_idx = subsample_idx[["sft_idx", "shard_path", "shard_row_idx",
                                   "source_dataset", "domain", "id"]]

    # Pass 2: pull `messages` field for picked rows shard-by-shard
    print(f"\nPulling messages field for {len(subsample_idx):,} rows ...")
    by_shard = subsample_idx.groupby("shard_path")
    messages_col = [None] * len(subsample_idx)
    for shard_name, grp in by_shard:
        fp = SFT_DIR / shard_name
        t = pq.read_table(fp, columns=["messages"])
        msg_arr = t["messages"].to_pylist()
        for _, row in grp.iterrows():
            messages_col[row["sft_idx"]] = msg_arr[int(row["shard_row_idx"])]
        print(f"  pulled from {shard_name}: {len(grp)} rows")
    subsample_idx["messages"] = messages_col

    out_path = out_dir / "sft_subsample.parquet"
    # Write parquet (messages is a nested list of dicts)
    pa_tbl = pd.DataFrame(subsample_idx)
    pa_tbl.to_parquet(out_path, index=False)
    print(f"\nWrote {len(pa_tbl):,} rows to {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "n_sampled": len(pa_tbl),
        "n_corpus": int(len(cat)),
        "n_sources": int(len(quota[quota > 0])),
        "source_cap_pct": args.source_cap_pct,
        "seed": args.seed,
        "parquet_path": str(out_path),
        "source_dataset_counts": quota.sort_values(ascending=False).to_dict(),
    }
    (meta_dir / "sft_subsample_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote metadata to "
          f"{(meta_dir / 'sft_subsample_meta.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
