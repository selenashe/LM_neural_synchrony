#!/usr/bin/env python3
"""Merge per-batch base re-run CSVs into the canonical v6.0 results dir.

After `bash/run_olmo3_7b_v6_0_pilot_base_rerun.sh` finishes all 12 array tasks,
each task has written its slice to a separate output dir:
  sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot_base_batch{NN}/
    results_OLMo-3-1025-7B_seed0.csv
    prompt_records/OLMo-3-1025-7B_seed0/episode_*.txt

This script concatenates the 12 batch CSVs into the canonical location:
  sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot/
    results_OLMo-3-1025-7B_seed0.csv

so the analyzer (analysis/analyze_v6_0.py) picks it up alongside the SFT/DPO/
RLVR CSVs without changes. episode_idx is reassigned to a unique 0..N-1
sequence in env_id order so the merged file looks identical to a single-shot
run. prompt_records are left in their batch dirs (the analyzer doesn't read
them; they're available for debugging).

Usage:
  python simulation/merge_v6_0_base_batches.py
  python simulation/merge_v6_0_base_batches.py --dry_run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = "OLMo-3-1025-7B"
SEED = 0
NUM_BATCHES = 12
CANONICAL_DIR = REPO_ROOT / "sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot"
BATCH_DIR_FMT = "sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot_base_batch{:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be merged without writing.")
    args = parser.parse_args()

    csv_name = f"results_{CHECKPOINT}_seed{SEED}.csv"
    canonical_csv = CANONICAL_DIR / csv_name

    frames = []
    missing = []
    for i in range(NUM_BATCHES):
        batch_csv = REPO_ROOT / BATCH_DIR_FMT.format(i) / csv_name
        if not batch_csv.exists():
            missing.append(batch_csv)
            continue
        df = pd.read_csv(batch_csv)
        print(f"  batch {i:02d}: {len(df):>5} rows  ({batch_csv.relative_to(REPO_ROOT)})")
        frames.append(df)

    if missing:
        print(f"\nERROR: {len(missing)} batch CSVs missing:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nTotal merged rows: {len(merged)}")

    # Reassign episode_idx 0..N-1. Sort by env_id first so the order is
    # deterministic (env_ids are V60_00001..V60_09500, lex order = numeric order).
    merged = merged.sort_values("env_id").reset_index(drop=True)
    merged["episode_idx"] = merged.index

    n_dup_env = merged["env_id"].duplicated().sum()
    if n_dup_env:
        print(f"WARNING: {n_dup_env} duplicate env_ids in the merged set — "
              f"batches overlap. Inspect before using.", file=sys.stderr)

    if canonical_csv.exists():
        print(f"\nERROR: canonical CSV already exists at {canonical_csv}. "
              f"Delete or move it before merging to avoid clobbering.",
              file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"\n[dry run] would write {len(merged)} rows to {canonical_csv}")
        return

    merged.to_csv(canonical_csv, index=False)
    print(f"\nWrote {canonical_csv.relative_to(REPO_ROOT)} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
