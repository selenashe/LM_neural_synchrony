#!/usr/bin/env python3
"""
Analyze v5.1 — Sycophancy False-Belief pilot.

Thin wrapper around `analyze_v5_0.py`. The v5.1 CSV schema mirrors v5.0
exactly except for the question-identity columns (`scenario_id` / `item`
instead of `question_id` / `question_domain`) and the v5.1-only
`scenario_prior` column. We patch the v5.0 module's constants and
`load_results` so that all of its plotting functions (which key on
`question_id`) operate on v5.1 data unchanged.

Usage:
  python analysis/analyze_v5_1.py --model_family olmo31-32b-instruct
  python analysis/analyze_v5_1.py --model_family olmo3-7b-instruct
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import analyze_v5_0 as base  # noqa: E402

# ── v5.1 overrides ─────────────────────────────────────────────────────────

V51_MODEL_FAMILIES = {
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct (v5.1 False-Belief Sycophancy)",
        "checkpoints": [
            "Olmo-3-1125-32B",
            "Olmo-3.1-32B-Instruct-SFT",
            "Olmo-3.1-32B-Instruct-DPO",
            "Olmo-3.1-32B-Instruct",
        ],
        "stage_labels": {
            "Olmo-3-1125-32B": "Base",
            "Olmo-3.1-32B-Instruct-SFT": "SFT",
            "Olmo-3.1-32B-Instruct-DPO": "DPO",
            "Olmo-3.1-32B-Instruct": "Instruct",
        },
        "results_family": "olmo31_32b",
        "default_output_dir": "analysis_outputs/v5_1_olmo31_32b_pilot",
    },
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v5.1 False-Belief Sycophancy)",
        "checkpoints": [
            "OLMo-3-1025-7B",
            "OLMo-3-7B-Instruct-SFT",
            "OLMo-3-7B-Instruct-DPO",
            "OLMo-3-7B-Instruct-RLVR-step400",
        ],
        "stage_labels": {
            "OLMo-3-1025-7B": "Base",
            "OLMo-3-7B-Instruct-SFT": "SFT",
            "OLMo-3-7B-Instruct-DPO": "DPO",
            "OLMo-3-7B-Instruct-RLVR-step400": "RLVR",
        },
        "results_family": "olmo3_instruct",
        "default_output_dir": "analysis_outputs/v5_1_olmo3_7b_pilot",
    },
}

# Full 8-cell factorial on disk; uo dropped for analysis (does not differ
# meaningfully from sys). Mirrors the v5.0 collapse.
ALL_V51_CONDITIONS = [
    "v5_1_single_sys_quote",
    "v5_1_single_sys_report",
    "v5_1_single_uo_quote",
    "v5_1_single_uo_report",
    "v5_1_multi_sys_quote",
    "v5_1_multi_sys_report",
    "v5_1_multi_uo_quote",
    "v5_1_multi_uo_report",
]

V51_CONDITIONS = [
    "v5_1_single_sys_quote",
    "v5_1_single_sys_report",
    "v5_1_multi_sys_quote",
    "v5_1_multi_sys_report",
]

V51_COND_SHORT = {
    "v5_1_single_sys_quote":  "S·quote",
    "v5_1_single_sys_report": "S·report",
    "v5_1_multi_sys_quote":   "M·quote",
    "v5_1_multi_sys_report":  "M·report",
}


def _v51_condition_dir(family_results, condition, run_label=""):
    suffix = condition[len("v5_1_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v5_1_{suffix}_pilot{label_suffix}"


def _v51_load_results(family, seed=0, run_label=""):
    """Load v5.1 CSVs and alias scenario_id→question_id, item→question_domain
    so v5.0 plot code (which keys on `question_id`) just works."""
    frames = []
    for cond in V51_CONDITIONS:
        results_dir = REPO_ROOT / _v51_condition_dir(
            family["results_family"], cond, run_label,
        )
        for ckpt in family["checkpoints"]:
            csv_path = results_dir / f"results_{ckpt}_seed{seed}.csv"
            if not csv_path.exists():
                print(f"  WARNING: missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No v5.1 result CSVs found")
    df = pd.concat(frames, ignore_index=True)

    # Alias v5.1 columns to v5.0 names so downstream plots key consistently.
    if "scenario_id" in df.columns and "question_id" not in df.columns:
        df["question_id"] = df["scenario_id"]
    if "item" in df.columns and "question_domain" not in df.columns:
        df["question_domain"] = df["item"]

    for n in base.CASCADE_TURNS:
        for letter in ["A", "B", "C", "D"]:
            col = f"turn{n}_prob_{letter}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cue_strength"] = pd.to_numeric(df["cue_strength"], errors="coerce").astype(int)

    for n in base.CASCADE_TURNS:
        if f"turn{n}_prob_A" not in df.columns:
            continue
        df[f"turn{n}_prob_correct"] = df.apply(
            lambda r, n=n: r[f"turn{n}_prob_{r['correct_position']}"]
            if pd.notna(r.get(f"turn{n}_prob_A")) else np.nan, axis=1)
        df[f"turn{n}_prob_wrong"] = df.apply(
            lambda r, n=n: r[f"turn{n}_prob_{r['wrong_position']}"]
            if pd.notna(r.get(f"turn{n}_prob_A")) else np.nan, axis=1)
        df[f"turn{n}_is_correct"] = df[f"turn{n}_correct"] == "correct"

    df["turn4_is_suggested_wrong"] = df["turn4_parsed_letter"] == df["wrong_position"]
    df["prob_correct_shift"] = df["turn4_prob_correct"] - df["turn2_prob_correct"]
    return df


def _patch_base():
    """Swap v5.0 module-level constants/functions for v5.1 equivalents.

    `base.load_results` and several plotting functions reference these names
    through the module's globals dict at runtime, so patching them here
    flows through transparently."""
    base.MODEL_FAMILIES = V51_MODEL_FAMILIES
    base.CONDITIONS = V51_CONDITIONS
    base.COND_SHORT = V51_COND_SHORT
    base._condition_dir = _v51_condition_dir
    base.load_results = _v51_load_results


def main():
    _patch_base()

    parser = argparse.ArgumentParser(
        description="Analyze v5.1 sycophancy false-belief pilot."
    )
    parser.add_argument("--model_family", type=str, default="olmo31-32b-instruct",
                        choices=list(V51_MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation.")
    args = parser.parse_args()

    family = V51_MODEL_FAMILIES[args.model_family]
    checkpoints = family["checkpoints"]
    stage_labels = family["stage_labels"]
    family_title = family["title"]

    default_out = family["default_output_dir"]
    if args.run_label:
        default_out = f"{default_out}_{args.run_label}"
    out_dir = REPO_ROOT / (args.output_dir or default_out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model family: {family_title}")
    if args.run_label:
        print(f"Run label: {args.run_label}")
    df = _v51_load_results(family, seed=args.seed, run_label=args.run_label)
    print(f"  Loaded {len(df)} rows across {df['checkpoint'].nunique()} checkpoints "
          f"and {df['condition'].nunique()} conditions")

    base.report_data_quality(df, out_dir, checkpoints, stage_labels)
    base.report_baseline_accuracy(df, out_dir, checkpoints, stage_labels)
    base.plot_headline_22(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_rate_by_cue_strength(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_rate_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_to_suggested(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_marginals(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_multi_metrics(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_training_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_per_question(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_probs(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_vs_generated(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_interactions(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_per_condition_breakdowns(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_cascade_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_cascade_by_cue_strength(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_question_cascade_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
