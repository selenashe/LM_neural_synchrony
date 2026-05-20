#!/usr/bin/env python3
"""Tier 1 structural analysis of OLMo 3.1 32B post-training datasets."""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = "/juice6/scr6/nlp/interp-models/OLMo-3.1-32B/datasets"


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_counter(counter, label="", top_n=None, total=None):
    if total is None:
        total = sum(counter.values())
    items = counter.most_common(top_n)
    max_key_len = max(len(str(k)) for k, _ in items) if items else 0
    for k, v in items:
        print(f"  {str(k):<{max_key_len}}  {v:>10,}  ({100*v/total:5.1f}%)")
    if top_n and len(counter) > top_n:
        other = sum(v for k, v in counter.most_common()[top_n:])
        print(f"  {'(other)':<{max_key_len}}  {other:>10,}  ({100*other/total:5.1f}%)")
    if label:
        print(f"  {label}: {total:,} total")


def print_length_stats(lengths, label=""):
    arr = np.array(lengths)
    if len(arr) == 0:
        print(f"  {label}: (empty)")
        return
    print(f"  {label} (n={len(arr):,}):")
    print(f"    mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
          f"p5={np.percentile(arr,5):.0f}  p25={np.percentile(arr,25):.0f}  "
          f"p75={np.percentile(arr,75):.0f}  p95={np.percentile(arr,95):.0f}  "
          f"max={arr.max():.0f}")


def get_parquet_files(subpath):
    data_dir = os.path.join(BASE, subpath, "data")
    return sorted(glob.glob(os.path.join(data_dir, "*.parquet")))


# ─────────────────────────────────────────────────────────────────────────────
# SFT analysis (works for both Instruct and Thinking)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sft(name, subpath, source_col, has_domain=False):
    print_header(f"SFT: {name}")
    files = get_parquet_files(subpath)

    total = 0
    msg_count_dist = Counter()      # len(messages) -> count
    user_turn_dist = Counter()      # num user turns -> count
    source_single = Counter()
    source_multi = Counter()
    domain_single = Counter() if has_domain else None
    domain_multi = Counter() if has_domain else None

    user_lens_single = []
    asst_lens_single = []
    user_lens_multi = []
    asst_lens_multi = []

    for fi, f in enumerate(files):
        df = pd.read_parquet(f)
        total += len(df)

        for _, row in df.iterrows():
            msgs = row["messages"]
            n_msgs = len(msgs)
            msg_count_dist[n_msgs] += 1

            n_user = sum(1 for m in msgs if m["role"] == "user")
            user_turn_dist[n_user] += 1

            is_multi = n_user >= 2
            src = row.get(source_col, "unknown")

            if is_multi:
                source_multi[src] += 1
            else:
                source_single[src] += 1

            if has_domain:
                dom = row.get("domain", "unknown")
                if is_multi:
                    domain_multi[dom] += 1
                else:
                    domain_single[dom] += 1

            for m in msgs:
                c = len(m.get("content", "") or "")
                if m["role"] == "user":
                    (user_lens_multi if is_multi else user_lens_single).append(c)
                elif m["role"] == "assistant":
                    (asst_lens_multi if is_multi else asst_lens_single).append(c)

        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            print(f"  ... processed {fi+1}/{len(files)} shards ({total:,} rows)")

    n_single = sum(v for k, v in user_turn_dist.items() if k <= 1)
    n_multi = total - n_single

    print(f"\n  Total examples: {total:,}")
    print(f"  Single-turn (1 user msg):  {n_single:>10,}  ({100*n_single/total:.1f}%)")
    print(f"  Multi-turn  (2+ user msg): {n_multi:>10,}  ({100*n_multi/total:.1f}%)")

    print(f"\n  --- Message count distribution (top 15) ---")
    print_counter(msg_count_dist, top_n=15)

    print(f"\n  --- User turns per conversation ---")
    print_counter(user_turn_dist, top_n=15)

    print(f"\n  --- Turn length stats (chars) ---")
    print_length_stats(user_lens_single, "User turns (single-turn convos)")
    print_length_stats(asst_lens_single, "Assistant turns (single-turn convos)")
    print_length_stats(user_lens_multi, "User turns (multi-turn convos)")
    print_length_stats(asst_lens_multi, "Assistant turns (multi-turn convos)")

    print(f"\n  --- Source dataset (single-turn, top 15) ---")
    print_counter(source_single, top_n=15)
    print(f"\n  --- Source dataset (multi-turn, top 15) ---")
    print_counter(source_multi, top_n=15)

    if has_domain:
        print(f"\n  --- Domain (single-turn) ---")
        print_counter(domain_single)
        print(f"\n  --- Domain (multi-turn) ---")
        print_counter(domain_multi)


# ─────────────────────────────────────────────────────────────────────────────
# DPO analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dpo(name, subpath, has_dataset_source=False):
    print_header(f"DPO: {name}")
    files = get_parquet_files(subpath)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    total = len(df)
    print(f"  Total examples: {total:,}")

    # Turn count distribution (based on chosen)
    chosen_lens = df["chosen"].apply(len)
    msg_count_dist = Counter(chosen_lens)

    # User turn count
    def count_user_turns(msgs):
        return sum(1 for m in msgs if m.get("role") == "user")

    user_turns = df["chosen"].apply(count_user_turns)
    user_turn_dist = Counter(user_turns)

    n_single = int((user_turns <= 1).sum())
    n_multi = total - n_single

    print(f"  Single-turn (1 user msg):  {n_single:>10,}  ({100*n_single/total:.1f}%)")
    print(f"  Multi-turn  (2+ user msg): {n_multi:>10,}  ({100*n_multi/total:.1f}%)")

    print(f"\n  --- Chosen message count distribution (top 15) ---")
    print_counter(msg_count_dist, top_n=15)

    print(f"\n  --- User turns per conversation ---")
    print_counter(user_turn_dist, top_n=15)

    # Chosen vs rejected final assistant turn length
    def last_assistant_len(msgs):
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                return len(m.get("content", "") or "")
        return 0

    chosen_final = df["chosen"].apply(last_assistant_len)
    rejected_final = df["rejected"].apply(last_assistant_len)
    diff = chosen_final - rejected_final

    print(f"\n  --- Final assistant turn length (chars) ---")
    print_length_stats(chosen_final.tolist(), "Chosen")
    print_length_stats(rejected_final.tolist(), "Rejected")
    print(f"  Length diff (chosen - rejected): mean={diff.mean():.0f}  median={diff.median():.0f}")

    # Preference type
    print(f"\n  --- Preference type ---")
    print_counter(Counter(df["preference_type"]))

    # Model breakdown
    print(f"\n  --- Chosen model (top 10) ---")
    print_counter(Counter(df["chosen_model"]), top_n=10)
    print(f"\n  --- Rejected model (top 10) ---")
    print_counter(Counter(df["rejected_model"]), top_n=10)

    # Source breakdown
    if has_dataset_source:
        src_col = "dataset_source"
        print(f"\n  --- Dataset source ---")
        print_counter(Counter(df[src_col]))

    # Turn length stats
    user_lens = []
    asst_lens_chosen = []
    asst_lens_rejected = []
    for _, row in df.iterrows():
        for m in row["chosen"]:
            c = len(m.get("content", "") or "")
            if m.get("role") == "user":
                user_lens.append(c)
            elif m.get("role") == "assistant":
                asst_lens_chosen.append(c)
        for m in row["rejected"]:
            if m.get("role") == "assistant":
                asst_lens_rejected.append(len(m.get("content", "") or ""))

    print(f"\n  --- Turn length stats (chars) ---")
    print_length_stats(user_lens, "User turns")
    print_length_stats(asst_lens_chosen, "Assistant turns (chosen)")
    print_length_stats(asst_lens_rejected, "Assistant turns (rejected)")


# ─────────────────────────────────────────────────────────────────────────────
# RLVR analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_rlvr(name, subpath, has_ability=False):
    print_header(f"RLVR: {name}")
    files = get_parquet_files(subpath)
    cols_to_read = ["prompt", "passrate", "dataset_source"]
    if has_ability:
        cols_to_read.extend(["ability", "constraint_type"])
    df = pd.concat([pd.read_parquet(f, columns=cols_to_read) for f in files], ignore_index=True)
    total = len(df)
    print(f"  Total examples: {total:,}")

    # Prompt length
    prompt_lens = df["prompt"].str.len()
    print_length_stats(prompt_lens.tolist(), "Prompt length (chars)")

    # Passrate distribution
    pr = df["passrate"]
    bins = [
        ("= 0", (pr == 0).sum()),
        ("(0, 0.25]", ((pr > 0) & (pr <= 0.25)).sum()),
        ("(0.25, 0.5]", ((pr > 0.25) & (pr <= 0.5)).sum()),
        ("(0.5, 0.75]", ((pr > 0.5) & (pr <= 0.75)).sum()),
        ("(0.75, 1.0)", ((pr > 0.75) & (pr < 1.0)).sum()),
        ("= 1.0", (pr == 1.0).sum()),
    ]
    print(f"\n  --- Passrate distribution ---")
    for label, count in bins:
        print(f"  {label:<15}  {count:>10,}  ({100*count/total:5.1f}%)")

    # Dataset source
    print(f"\n  --- Dataset source ---")
    print_counter(Counter(df["dataset_source"]))

    if has_ability:
        print(f"\n  --- Ability ---")
        print_counter(Counter(df["ability"].dropna()))
        ct = df["constraint_type"].dropna()
        if len(ct) > 0:
            print(f"\n  --- Constraint type (top 10) ---")
            print_counter(Counter(ct), top_n=10)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write results to this file instead of stdout")
    args = parser.parse_args()

    if args.output:
        outfile = open(args.output, "w")
        sys.stdout = outfile
        sys.stderr.write(f"Writing results to {args.output}\n")

    print("Tier 1 Structural Analysis — OLMo 3.1 32B Post-Training Datasets")
    print(f"Data root: {BASE}")

    analyze_sft("Instruct (dolci-instruct-sft)", "sft/dolci-instruct-sft",
                source_col="source_dataset", has_domain=True)

    analyze_sft("Thinking (dolci-thinking-sft)", "sft/dolci-thinking-sft",
                source_col="dataset_source", has_domain=False)

    analyze_dpo("Instruct (dolci-instruct-dpo)", "dpo/dolci-instruct-dpo",
                has_dataset_source=False)

    analyze_dpo("Thinking (dolci-thinking-dpo)", "dpo/dolci-thinking-dpo",
                has_dataset_source=True)

    analyze_rlvr("Instruct (dolci-instruct-rl)", "rlvr/dolci-instruct-rl",
                 has_ability=True)

    analyze_rlvr("Think (dolci-think-rl)", "rlvr/dolci-think-rl",
                 has_ability=False)

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")

    if args.output:
        sys.stdout = sys.__stdout__
        outfile.close()
        sys.stderr.write(f"Finished. Results written to {args.output}\n")
