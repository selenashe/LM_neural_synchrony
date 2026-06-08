#!/usr/bin/env python3
"""
Analyze v5.0 Jing-non-option-suggested-cue pilot.

Thin wrapper around `analyze_v5_0.py` (with v6.0-style monkey-patches for
multi-cue support). CSV schema is identical to v5.0 — only the condition
prefix differs (`v5_0_jing_nonoption_suggested_cue_*`) and the design is
collapsed to a single cell:
  * 1 condition: v5_0_jing_nonoption_suggested_cue_multi_sys_report
  * 20 questions × 38 cues = 760 multi-turn trials per checkpoint.

Patches v5.0 module constants (MODEL_FAMILIES, CONDITIONS, COND_SHORT,
_condition_dir, STRENGTHS, CUE_LABELS) so all strength-axis plots extend
to 38 cues, then runs only the v5.0 plot functions that remain meaningful
under a single-condition design (skipping headline-2×2, marginals,
multi-metrics, condition interactions, and per-condition breakdowns).

Caveats baked into the design:
  * Cues c20-c38 (and c11/c30, rebuttal_only) have NO designated wrong
    option — flip_to_suggested values for these cues are undefined.
    The plot still draws bars for them; readers should ignore those bars.
  * Strength axis above s=10 is categorical (pressure-type index), not an
    ordinal authority intensity. Above s=19 it is a second "no-option"
    pass over the same categorical types. Read strength-axis plots with
    the CUE_LABELS legend for s > 10.

Usage:
  python analysis/analyze_v5_0_jing_nonoption_suggested_cue.py --model_family olmo3-7b-instruct
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_v5_0 as base  # noqa: E402
# Reuse v6.0's plot replacements and exclusion logic — same single-condition
# design, larger N, larger cue count.
import analyze_v6_0 as v60  # noqa: E402

# ── v5.0 jing-non-option-suggested-cue overrides ───────────────────────────

JNOS_MODEL_FAMILIES = {
    "olmo3-7b-instruct": {
        "title": "OLMo-3 7B Instruct (v5.0 Jing non-option-suggested cue)",
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
        "default_output_dir": "analysis_outputs/v5_0_jing_nonoption_suggested_cue_olmo3_7b_pilot",
    },
}

JNOS_CONDITIONS = [
    "v5_0_jing_nonoption_suggested_cue_multi_sys_report",
]

JNOS_COND_SHORT = {
    "v5_0_jing_nonoption_suggested_cue_multi_sys_report": "M·report",
}

# Strengths 1-19 inherit v5.0 labels (1-10) + v6.0 labels (11-19);
# strengths 20-38 are the non-option-suggesting adaptations of 1-19.
# Short " (NO)" suffix = "no-option" tag, kept terse so tick labels render.
JNOS_CUE_LABELS_EXTRA = {
    # 11-19 from v6.0
    11: "Rebuttal\nonly",
    12: "Biographical\nauthority",
    13: "Self-doubt\nmimicry",
    14: "Citation\n(Wikipedia)",
    15: "Emotional\nappeal",
    16: "Feedback\nsycophancy",
    17: "Leading\npremise",
    18: "Reward\nframing",
    19: "Textbook\ncitation",
    # 20-38 non-option-suggesting
    20: "Gentle\ndoubt (NO)",
    21: "Mild\nsuggestion (NO)",
    22: "Social\nproof (NO)",
    23: "Polite\ndisagree (NO)",
    24: "Confident\nassertion (NO)",
    25: "Authority\nappeal (NO)",
    26: "Direct\ncorrection (NO)",
    27: "Firm\ncorrection (NO)",
    28: "Harsh\ncorrection (NO)",
    29: "Expert\ndemand (NO)",
    30: "Rebuttal\nonly (NO)",
    31: "Biographical\nauthority (NO)",
    32: "Self-doubt\nmimicry (NO)",
    33: "Citation\n(Wikipedia) (NO)",
    34: "Emotional\nappeal (NO)",
    35: "Feedback\nsycophancy (NO)",
    36: "Leading\npremise (NO)",
    37: "Reward\nframing (NO)",
    38: "Textbook\ncitation (NO)",
}

JNOS_STRENGTHS = list(range(1, 39))  # 1..38


def _jnos_condition_dir(family_results, condition, run_label=""):
    """Output dirs are `sotopia_results_<family>_v5_0_jing_nonoption_suggested_cue_<suffix>_pilot`.
    Same `v5_0_` strip as v5.0/jing_strong_cue."""
    suffix = condition[len("v5_0_"):]
    label_suffix = f"_{run_label}" if run_label else ""
    return f"sotopia_results_{family_results}_v5_0_{suffix}_pilot{label_suffix}"


def _patch_base():
    base.MODEL_FAMILIES = JNOS_MODEL_FAMILIES
    base.CONDITIONS = JNOS_CONDITIONS
    base.COND_SHORT = JNOS_COND_SHORT
    base._condition_dir = _jnos_condition_dir
    base.STRENGTHS = JNOS_STRENGTHS
    base.CUE_LABELS = {**base.CUE_LABELS, **JNOS_CUE_LABELS_EXTRA}


def main():
    _patch_base()

    parser = argparse.ArgumentParser(
        description="Analyze v5.0 jing-non-option-suggested-cue pilot."
    )
    parser.add_argument("--model_family", type=str, default="olmo3-7b-instruct",
                        choices=list(JNOS_MODEL_FAMILIES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_label", type=str, default="",
                        help="Match the --label used in simulation.")
    args = parser.parse_args()

    family = JNOS_MODEL_FAMILIES[args.model_family]
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

    # Skipped (require multiple conditions / both turn-types):
    #   plot_headline_22, plot_marginals, plot_multi_metrics,
    #   plot_interactions, plot_per_condition_breakdowns
    base.report_data_quality(df, out_dir, checkpoints, stage_labels)
    df = v60.drop_null_response_trials(df, out_dir, checkpoints, stage_labels)
    base.report_baseline_accuracy(df, out_dir, checkpoints, stage_labels)
    v60.plot_headline_accuracy_by_stage(df, out_dir, checkpoints, stage_labels, family_title)
    v60.plot_flip_rate_by_cue_strength_bars(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_rate_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_flip_to_suggested(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_training_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    # 20 questions is small enough for the v5.0 per-question plot to remain
    # readable — keep it instead of v6.0's per-domain replacement.
    base.plot_per_question(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_probs(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_logit_vs_generated(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_cascade_dynamics(df, out_dir, checkpoints, stage_labels, family_title)
    v60.plot_cascade_by_cue_strength_t6(df, out_dir, checkpoints, stage_labels, family_title)
    base.plot_question_cascade_heatmap(df, out_dir, checkpoints, stage_labels, family_title)
    base.save_cell_summary(df, out_dir, stage_labels)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
