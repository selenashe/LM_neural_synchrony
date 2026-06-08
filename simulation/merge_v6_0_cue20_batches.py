#!/usr/bin/env python3
"""Append cue c20 batch CSVs to the canonical v6.0 results CSVs.

After `bash/run_olmo3_7b_v6_0_pilot_cue20.sh` finishes all 20 array tasks
(4 checkpoints × 5 batches of 100), each task has written 100 rows to its
own per-checkpoint, per-batch directory:

  sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot_cue20_batch{0..4}/
    results_<CHECKPOINT>_seed0.csv

This script, for each of the 4 checkpoints:
  1. Concatenates the 5 batch CSVs (500 c20 rows total).
  2. Re-assigns `episode_idx` to extend the canonical CSV's existing
     idx range (purely cosmetic — the analyzer doesn't use it).
  3. Appends to the canonical CSV at
       sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot/
         results_<CHECKPOINT>_seed0.csv

Refuses to merge if the canonical CSV already has any c20 rows (i.e. the
merge has been done) — re-run safely after deleting them if you need to.

Usage:
  python simulation/merge_v6_0_cue20_batches.py --dry_run
  python simulation/merge_v6_0_cue20_batches.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINTS = [
    "OLMo-3-1025-7B",
    "OLMo-3-7B-Instruct-SFT",
    "OLMo-3-7B-Instruct-DPO",
    "OLMo-3-7B-Instruct-RLVR-step400",
]
SEED = 0
N_BATCHES = 5
CANONICAL_DIR = REPO_ROOT / "sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot"
BATCH_DIR_FMT = "sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot_cue20_batch{}"


def merge_one(checkpoint, dry_run):
    csv_name = f"results_{checkpoint}_seed{SEED}.csv"
    canonical_csv = CANONICAL_DIR / csv_name

    if not canonical_csv.exists():
        print(f"  [{checkpoint}] ERROR: canonical CSV missing at "
              f"{canonical_csv.relative_to(REPO_ROOT)}", file=sys.stderr)
        return False

    # Load batches.
    frames = []
    missing = []
    for b in range(N_BATCHES):
        batch_csv = REPO_ROOT / BATCH_DIR_FMT.format(b) / csv_name
        if not batch_csv.exists():
            missing.append(batch_csv)
            continue
        df = pd.read_csv(batch_csv)
        print(f"  [{checkpoint}] batch {b}: {len(df):>4} rows  "
              f"({batch_csv.relative_to(REPO_ROOT)})")
        frames.append(df)

    if missing:
        print(f"  [{checkpoint}] ERROR: {len(missing)} batch CSVs missing:",
              file=sys.stderr)
        for p in missing:
            print(f"    {p}", file=sys.stderr)
        return False

    new_df = pd.concat(frames, ignore_index=True)

    # Sanity: every appended row must be a c20 trial.
    cue_set = set(new_df["cue_id"].unique())
    if cue_set != {"c20"}:
        print(f"  [{checkpoint}] ERROR: appended rows contain cues "
              f"{sorted(cue_set)} (expected only c20). Aborting.",
              file=sys.stderr)
        return False

    # Check canonical doesn't already have c20.
    canon_df = pd.read_csv(canonical_csv)
    if (canon_df["cue_id"] == "c20").any():
        print(f"  [{checkpoint}] canonical already has "
              f"{(canon_df['cue_id'] == 'c20').sum()} c20 rows — skipping "
              f"(delete them and re-run if you need to re-merge).")
        return True  # not an error, just already done

    # Re-number episode_idx to extend the existing range.
    next_idx = int(canon_df["episode_idx"].max()) + 1 if len(canon_df) else 0
    new_df = new_df.sort_values("env_id").reset_index(drop=True)
    new_df["episode_idx"] = range(next_idx, next_idx + len(new_df))

    # Re-order columns to match the canonical schema exactly.
    new_df = new_df[canon_df.columns.tolist()]

    print(f"  [{checkpoint}] canonical has {len(canon_df)} rows; "
          f"appending {len(new_df)} c20 rows → {len(canon_df) + len(new_df)} total")

    if dry_run:
        print(f"  [{checkpoint}] [dry run] no write")
        return True

    # Append without header.
    new_df.to_csv(canonical_csv, mode="a", header=False, index=False)
    print(f"  [{checkpoint}] wrote → "
          f"{canonical_csv.relative_to(REPO_ROOT)}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be merged without writing.")
    args = parser.parse_args()

    print(f"Canonical dir: {CANONICAL_DIR.relative_to(REPO_ROOT)}")
    print(f"Checkpoints:   {len(CHECKPOINTS)}; batches per ckpt: {N_BATCHES}")
    print()

    all_ok = True
    for ckpt in CHECKPOINTS:
        print(f"=== {ckpt} ===")
        ok = merge_one(ckpt, args.dry_run)
        all_ok = all_ok and ok
        print()

    if not all_ok:
        sys.exit("Some checkpoints failed — see errors above.")
    print("Done.")


if __name__ == "__main__":
    main()
