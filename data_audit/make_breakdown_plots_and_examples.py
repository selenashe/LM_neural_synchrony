#!/usr/bin/env python3
"""Generate single-vs-multi-turn breakdown plots and sample examples for SFT/DPO phases."""

import gc
import glob
import json
import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

MAX_CONTENT_CHARS = 20000  # truncate per-message content when storing examples

BASE = "/juice6/scr6/nlp/interp-models/OLMo-3.1-32B/datasets"
OUT = "/juice6/u/jshe/nlp/LM_neural_synchrony/data_audit"
EX_DIR = os.path.join(OUT, "data_examples")
PLOT_DIR = os.path.join(OUT, "plots")
N_EXAMPLES = 5
TOP_N_SOURCES = 15
RNG = random.Random(0)

DATASETS = [
    # (label,        phase, subpath,                  msgs_col, source_col,        is_dpo)
    ("sft_instruct", "SFT", "sft/dolci-instruct-sft", "messages", "source_dataset",  False),
    ("sft_thinking", "SFT", "sft/dolci-thinking-sft", "messages", "dataset_source",  False),
    ("dpo_instruct", "DPO", "dpo/dolci-instruct-dpo", "chosen",   "preference_type", True),
    ("dpo_thinking", "DPO", "dpo/dolci-thinking-dpo", "chosen",   "dataset_source",  True),
]


def n_user_turns(msgs):
    return sum(1 for m in msgs if m.get("role") == "user")


def get_shards(subpath):
    return sorted(glob.glob(os.path.join(BASE, subpath, "data", "*.parquet")))


def _truncate(s):
    s = s or ""
    if len(s) <= MAX_CONTENT_CHARS:
        return s
    return s[:MAX_CONTENT_CHARS] + f"\n...[truncated {len(s) - MAX_CONTENT_CHARS} chars]"


def _compact_msgs(msgs):
    out = []
    for m in msgs:
        out.append({"role": m.get("role", "?"),
                    "content": _truncate(m.get("content", ""))})
    return out


def collect_stats_and_examples(label, subpath, msgs_col, source_col, is_dpo):
    """Stream parquet batches: count single/multi per source, reservoir-sample N_EXAMPLES per (turn_type, source)."""
    files = get_shards(subpath)
    counts = defaultdict(lambda: Counter())
    samples = defaultdict(list)
    seen = defaultdict(int)

    cols_needed = [msgs_col, source_col]
    if is_dpo:
        cols_needed.append("rejected")

    for fi, f in enumerate(files, 1):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2000, columns=cols_needed):
            msgs_col_arr = batch.column(msgs_col).to_pylist()
            src_col_arr = batch.column(source_col).to_pylist()
            rej_col_arr = batch.column("rejected").to_pylist() if is_dpo else None
            for i, msgs in enumerate(msgs_col_arr):
                src = src_col_arr[i] or "unknown"
                n_u = sum(1 for m in msgs if m.get("role") == "user")
                turn_type = "single" if n_u <= 1 else "multi"
                counts[turn_type][src] += 1

                key = (turn_type, src)
                seen[key] += 1
                store = len(samples[key]) < N_EXAMPLES
                if not store:
                    j = RNG.randint(0, seen[key] - 1)
                    if j < N_EXAMPLES:
                        store = True
                        idx = j
                if store:
                    ex = {"messages": _compact_msgs(msgs), "source": src, "turn_type": turn_type}
                    if is_dpo:
                        ex["rejected"] = _compact_msgs(rej_col_arr[i])
                    if len(samples[key]) < N_EXAMPLES:
                        samples[key].append(ex)
                    else:
                        samples[key][idx] = ex
            del msgs_col_arr, src_col_arr, rej_col_arr
        del pf
        gc.collect()
        if fi % 10 == 0 or fi == len(files):
            print(f"  [{label}] shard {fi}/{len(files)}", flush=True)

    return counts, samples


def write_examples(label, samples):
    base = os.path.join(EX_DIR, label)
    for (turn_type, src), exs in samples.items():
        sub = os.path.join(base, f"{turn_type}_turn")
        os.makedirs(sub, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in str(src))[:120]
        path = os.path.join(sub, f"{safe}.txt")
        with open(path, "w") as fh:
            fh.write(f"# Dataset: {label}\n# Turn type: {turn_type}\n# Source: {src}\n# Examples: {len(exs)}\n\n")
            for i, ex in enumerate(exs, 1):
                fh.write("=" * 80 + f"\n EXAMPLE {i}\n" + "=" * 80 + "\n")
                for m in ex["messages"]:
                    role = m.get("role", "?")
                    content = (m.get("content") or "")
                    fh.write(f"\n--- [{role}] ---\n{content}\n")
                if "rejected" in ex:
                    fh.write("\n" + "-" * 40 + " REJECTED " + "-" * 40 + "\n")
                    for m in ex["rejected"]:
                        role = m.get("role", "?")
                        content = (m.get("content") or "")
                        fh.write(f"\n--- [{role}] ---\n{content}\n")
                fh.write("\n")


def plot_single_vs_multi(stats_by_label):
    """One figure per phase: grouped bars showing % single vs % multi per dataset."""
    by_phase = defaultdict(list)
    for label, (counts, _phase) in stats_by_label.items():
        single = sum(counts["single"].values())
        multi = sum(counts["multi"].values())
        total = single + multi
        by_phase[_phase].append((label, 100.0 * single / total, 100.0 * multi / total, total))

    fig, axes = plt.subplots(1, len(by_phase), figsize=(6 * len(by_phase), 5), squeeze=False)
    for ax, (phase, rows) in zip(axes[0], by_phase.items()):
        labels = [r[0] for r in rows]
        single_pct = [r[1] for r in rows]
        multi_pct = [r[2] for r in rows]
        totals = [r[3] for r in rows]
        x = np.arange(len(labels))
        w = 0.38
        b1 = ax.bar(x - w/2, single_pct, w, label="Single-turn", color="#4C78A8")
        b2 = ax.bar(x + w/2, multi_pct,  w, label="Multi-turn",  color="#F58518")
        for bar, pct in zip(b1, single_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", fontsize=9)
        for bar, pct in zip(b2, multi_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={t:,})" for l, t in zip(labels, totals)])
        ax.set_ylabel("Percentage of dataset (%)")
        ax.set_title(f"{phase} phase: Single vs Multi-turn")
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "single_vs_multi_by_phase.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_source_breakdown(label, counts):
    """One figure per dataset: 2 subplots (single, multi) with horizontal source bars (top N)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, turn_type in zip(axes, ["single", "multi"]):
        c = counts[turn_type]
        total = sum(c.values())
        if total == 0:
            ax.text(0.5, 0.5, "(no rows)", ha="center", va="center")
            ax.set_title(f"{label} — {turn_type}-turn (total=0)")
            ax.axis("off")
            continue
        items = c.most_common(TOP_N_SOURCES)
        names = [str(k) for k, _ in items]
        vals = [100.0 * v / total for _, v in items]
        raw = [v for _, v in items]
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color="#4C78A8" if turn_type == "single" else "#F58518")
        ax.set_yticks(y)
        # Truncate long source names for display
        disp = [n if len(n) <= 55 else n[:52] + "..." for n in names]
        ax.set_yticklabels(disp, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of " + turn_type + "-turn examples")
        ax.set_title(f"{label} — {turn_type}-turn (n={total:,}, top {len(items)})")
        for bar, pct, r in zip(bars, vals, raw):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{pct:.1f}% ({r:,})", va="center", fontsize=8)
        ax.set_xlim(0, max(vals) * 1.25 + 1)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Source breakdown — {label}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"source_breakdown_{label}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(EX_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    stats_by_label = {}
    for label, phase, subpath, msgs_col, source_col, is_dpo in DATASETS:
        print(f"\n=== {label} ({subpath}) ===")
        counts, samples = collect_stats_and_examples(label, subpath, msgs_col, source_col, is_dpo)
        stats_by_label[label] = (counts, phase)
        write_examples(label, samples)
        plot_source_breakdown(label, counts)

        # Persist raw counts as JSON
        with open(os.path.join(PLOT_DIR, f"counts_{label}.json"), "w") as fh:
            json.dump({tt: dict(c) for tt, c in counts.items()}, fh, indent=2)

    plot_single_vs_multi(stats_by_label)
    print("\nDone.")


if __name__ == "__main__":
    main()
