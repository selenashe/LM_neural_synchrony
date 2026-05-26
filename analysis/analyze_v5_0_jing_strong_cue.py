#!/usr/bin/env python3
"""
Analyze v5_0 Jing-strong-cue variant.

Thin wrapper around `analyze_v5_0.py`. CSV schema is identical to v5.0 — only
the condition prefix differs (`v5_0_jing_strong_cue_*`), and there is a
single cue (so per-cue-strength plots collapse to one point per checkpoint).
We patch the v5.0 module's CONDITIONS, _condition_dir, COND_SHORT, and
default output dirs so all downstream plots run unchanged.

Usage:
  python analysis/analyze_v5_0_jing_strong_cue.py --model_family olmo3-7b-instruct
  python analysis/analyze_v5_0_jing_strong_cue.py --model_family olmo31-32b-instruct
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import analyze_v5_0 as base  # noqa: E402

# ── Jing-strong-cue overrides ──────────────────────────────────────────────

JSC_MODEL_FAMILIES = {
    "olmo31-32b-instruct": {
        "title": "OLMo-3.1 32B Instruct (v5.0 Jing strong-cue)",
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
        "default_output_dir": "analysis_outputs/v5_0_jing_strong_cue_olmo31_32b_pilot",
    },
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v5.0 Jing strong-cue)",
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
        "default_output_dir": "analysis_outputs/v5_0_jing_strong_cue_olmo3_7b_pilot",
    },
}

# Mirror the v5.0 post-hoc collapse: keep sys-only, drop uo.
JSC_CONDITIONS = [
    "v5_0_jing_strong_cue_single_sys_quote",
    "v5_0_jing_strong_cue_single_sys_report",
    "v5_0_jing_strong_cue_multi_sys_quote",
    "v5_0_jing_strong_cue_multi_sys_report",
]

JSC_COND_SHORT = {
    "v5_0_jing_strong_cue_single_sys_quote":  "S·quote",
    "v5_0_jing_strong_cue_single_sys_report": "S·report",
    "v5_0_jing_strong_cue_multi_sys_quote":   "M·quote",
    "v5_0_jing_strong_cue_multi_sys_report":  "M·report",
}


def _jsc_condition_dir(family_results, condition, run_label=""):
    """`condition[len("v5_0_"):]` strips the shared `v5_0_` prefix, leaving
    `jing_strong_cue_<turns>_<role>_<framing>` as the suffix — which matches
    the dirs the sim script produces via its own `_results_dir`."""
    suffix = condition[len("v5_0_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v5_0_{suffix}_pilot{label_suffix}"


def _patch_base():
    base.MODEL_FAMILIES = JSC_MODEL_FAMILIES
    base.CONDITIONS = JSC_CONDITIONS
    base.COND_SHORT = JSC_COND_SHORT
    base._condition_dir = _jsc_condition_dir


def main():
    _patch_base()

    parser = argparse.ArgumentParser(
        description="Analyze v5.0 jing-strong-cue variant."
    )
    parser.add_argument("--model_family", type=str, default="olmo3-7b-instruct",
                        choices=list(JSC_MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation.")
    args = parser.parse_args()

    family = JSC_MODEL_FAMILIES[args.model_family]
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
    df = base.load_results(family, seed=args.seed, run_label=args.run_label)
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
