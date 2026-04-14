#!/usr/bin/env python3
"""
Batch behavioral metrics over all Sotopia episodes for one model pair.

Use this to see whether failures like allocation / point miscounting cluster in
task domains (from build_labels_v2.task_structure) vs generic numeric-dialog
patterns.

Data source
-----------
Agent *transcripts* live in::

    sotopia_results/dialogs/<pair_name>/<episode>_temp<temp>_seed<seed>.csv

with columns Intro, Dialog, Intended Dialog.

The parallel ``prompt_records/*.csv`` files only store per-turn *prompts* (no model
replies). They cannot be used to detect miscounting in dialog; use the dialog CSVs.

Metrics
-------
1. Episode metadata: codename, source, task_structure (from env_agent_combos + envs).

2. Deal-style grounding (only when the intro matches ``logit_lens_mistral`` heuristics):
   - scenario_inventory parsed (books/hats/ball counts)
   - per-agent point tables parsed
   - allocation_sums_to_inventory (regex allocation vs inventory)
   - optional: claimed point totals vs computed private utility when allocation valid

3. Cross-domain surface stats on the Dialog text (all episodes):
   - numbers per 1k characters
   - counts of $ amounts, and math-ish cue words (total, each, percent, split, ...)

Outputs JSON (per-episode list + aggregate by task_structure) and prints a summary.

Usage:
  python analyze_transcript_domain_metrics.py \\
    --pair-name Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0 \\
    --temp 0.7 --seed 0 \\
    --output transcript_domain_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Imports torch/matplotlib as side effect; acceptable for one-shot batch analysis.
import logit_lens_mistral as llm  # noqa: E402

from build_labels_v2 import classify_task_structure  # noqa: E402
from experiment_config import COMBOS_PATH, EPISODES_NUM  # noqa: E402

ENVS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils", "sotopia_data", "envs.json")

# Loose "math language" cues in dialog (domain-agnostic surface signal).
_MATH_CUE_RE = re.compile(
    r"(?i)\b(total of|in total|adds? up|each|per |split |half of|"
    r"percent|percentage|dollar|\$\d|multiply|divide|average|mean |"
    r"sum |equals?|\+|=\s*\d)\b"
)


def load_episode_metadata() -> Tuple[Dict[int, Dict[str, Any]], int]:
    with open(ENVS_PATH) as f:
        envs = json.load(f)
    with open(COMBOS_PATH) as f:
        combos = json.load(f)
    meta: Dict[int, Dict[str, Any]] = {}
    for idx, combo in enumerate(combos):
        env_id = combo["env_id"]
        ev = envs.get(env_id, {})
        codename = ev.get("codename", "?")
        source = ev.get("source", "?")
        meta[idx] = {
            "env_id": env_id,
            "codename": codename,
            "source": source,
            "task_structure": classify_task_structure(codename, source),
        }
    return meta, len(combos)


def dialog_surface_stats(dialog_blob: str) -> Dict[str, Any]:
    if not dialog_blob:
        return {
            "chars": 0,
            "numbers_per_1k_chars": 0.0,
            "number_tokens": 0,
            "dollar_mentions": 0,
            "math_cue_hits": 0,
        }
    chars = max(len(dialog_blob), 1)
    numbers = re.findall(r"\b\d+\b", dialog_blob)
    dollars = len(re.findall(r"\$[\d,]+(?:\.\d+)?", dialog_blob))
    cues = len(_MATH_CUE_RE.findall(dialog_blob))
    return {
        "chars": len(dialog_blob),
        "numbers_per_1k_chars": 1000.0 * len(numbers) / chars,
        "number_tokens": len(numbers),
        "dollar_mentions": dollars,
        "math_cue_hits": cues,
    }


def claimed_vs_computed(
    summary: Dict[str, Any],
    agent1: str,
    agent2: str,
) -> Dict[str, Any]:
    """When allocation validated and we have claimed totals, compare."""
    alloc_ok = summary.get("allocation_sums_to_inventory")
    claimed = summary.get("final_claimed_rewards")
    computed = summary.get("outcome_rewards_private_utility")
    out: Dict[str, Any] = {
        "claimed_recorded": claimed is not None,
        "computed_recorded": computed is not None,
        "claimed_matches_computed": None,
    }
    if not alloc_ok or not claimed or not computed:
        return out
    try:
        match = (
            claimed.get(agent1) == computed.get(agent1)
            and claimed.get(agent2) == computed.get(agent2)
        )
        out["claimed_matches_computed"] = bool(match)
    except Exception:
        out["claimed_matches_computed"] = None
    return out


def analyze_pair(
    pair_name: str,
    temp: float,
    seed: int,
    n_episodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta_map, ncombo = load_episode_metadata()
    if n_episodes > ncombo:
        raise ValueError(f"--episodes max is {ncombo} (env combos length)")

    rows: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        path = os.path.join(
            llm.REPO_ROOT,
            "sotopia_results",
            "dialogs",
            pair_name,
            f"{ep}_temp{temp}_seed{seed}.csv",
        )
        if not os.path.isfile(path):
            rows.append(
                {
                    "episode": ep,
                    "error": "missing_dialog_csv",
                    "path": path,
                }
            )
            continue

        with open(path, newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        if not csv_rows:
            rows.append({"episode": ep, "error": "empty_csv", "path": path})
            continue

        record = csv_rows[0]
        intro = record.get("Intro", "") or ""
        dialog = record.get("Dialog", "") or ""
        em = meta_map.get(ep, {})
        agent1, agent2 = llm.infer_agent_names(record)

        summary = llm.build_behavioral_summary(record, agent1, agent2)
        inv = summary.get("scenario_inventory") or {}
        pv = summary.get("point_values") or {}
        surface = dialog_surface_stats(dialog)

        deal_inventory_parsed = bool(inv)
        deal_points_parsed = len(pv) == 2
        allocation_ok = bool(summary.get("allocation_sums_to_inventory"))
        claim_cmp = claimed_vs_computed(summary, agent1, agent2)

        rows.append(
            {
                "episode": ep,
                "codename": em.get("codename"),
                "source": em.get("source"),
                "task_structure": em.get("task_structure"),
                "deal_inventory_parsed": deal_inventory_parsed,
                "deal_points_parsed": deal_points_parsed,
                "deal_strict_applicable": deal_inventory_parsed and deal_points_parsed,
                "allocation_sums_to_inventory": allocation_ok,
                "deal_reached": summary.get("deal_reached"),
                "final_claimed_rewards": summary.get("final_claimed_rewards"),
                "outcome_rewards_private_utility": summary.get("outcome_rewards_private_utility"),
                "claimed_vs_computed": claim_cmp,
                "dialog_surface": surface,
            }
        )

    # --- aggregates by task_structure ---
    by_ts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_ts[r["task_structure"]].append(r)

    aggregate: Dict[str, Any] = {}
    for ts, lst in sorted(by_ts.items()):
        n = len(lst)
        strict = [x for x in lst if x.get("deal_strict_applicable")]
        alloc_fail = [x for x in strict if not x.get("allocation_sums_to_inventory")]
        mismatch = [
            x
            for x in strict
            if x.get("claimed_vs_computed", {}).get("claimed_matches_computed") is False
        ]
        surf_nums = [x["dialog_surface"]["numbers_per_1k_chars"] for x in lst]
        mean_nums = sum(surf_nums) / len(surf_nums) if surf_nums else 0.0

        pct_dollar = sum(1 for x in lst if x["dialog_surface"]["dollar_mentions"] > 0) / n if n else 0.0
        aggregate[ts] = {
            "n_episodes": n,
            "deal_strict_applicable_n": len(strict),
            "allocation_fail_among_strict": len(alloc_fail),
            "allocation_fail_rate_among_strict": (
                len(alloc_fail) / len(strict) if strict else None
            ),
            "claimed_mismatch_among_strict": len(mismatch),
            "mean_numbers_per_1k_chars": mean_nums,
            "mean_math_cue_hits": (
                sum(x["dialog_surface"]["math_cue_hits"] for x in lst) / n if n else 0.0
            ),
            "mean_dollar_mentions": (
                sum(x["dialog_surface"]["dollar_mentions"] for x in lst) / n if n else 0.0
            ),
            "pct_episodes_with_dollar_mention": pct_dollar,
        }

    return rows, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-name", required=True, help="Folder under sotopia_results/dialogs/")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_NUM,
        help="How many episode indices 0..N-1 to scan (default: length of env agent combo list).",
    )
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "transcript_domain_metrics.json"))
    args = parser.parse_args()

    per_episode, aggregate = analyze_pair(args.pair_name, args.temp, args.seed, args.episodes)

    out_doc = {
        "pair_name": args.pair_name,
        "temp": args.temp,
        "seed": args.seed,
        "note": (
            "allocation_* metrics apply only when deal_strict_applicable (books/hats/ball "
            "inventory + both agents' point rows parsed). Other scenarios need different checks."
        ),
        "per_episode": per_episode,
        "aggregate_by_task_structure": aggregate,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)

    print(f"Wrote {args.output}")
    print("\n=== Aggregate by task_structure ===")
    for ts in sorted(aggregate.keys()):
        a = aggregate[ts]
        rate = a["allocation_fail_rate_among_strict"]
        rate_s = f"{rate:.1%}" if rate is not None else "n/a"
        pct = a.get("pct_episodes_with_dollar_mention", 0.0)
        print(
            f"{ts:32s}  n={a['n_episodes']:3d}  "
            f"deal_strict={a['deal_strict_applicable_n']:3d}  "
            f"alloc_fail(strict)={rate_s}  "
            f"nums/1k≈{a['mean_numbers_per_1k_chars']:.1f}  "
            f"pct$_dlg≈{pct:.0%}"
        )


if __name__ == "__main__":
    main()
