#!/usr/bin/env python3
"""
V5.0 Jing-non-option-suggested-cue variant.

Same prompt scaffolding as v5.0's `multi_sys_report` cell (system-prompted
multi-turn: pre-cue T2 + post-cue T4 + cascade T6/8/10/12/14, third-person
report framing). Stimuli are 20 v5.0 MMLU questions × 38 cues:
  * c01-c19: v6.0 unified cue set (v5.0 ladder c01-c10 + 9 new pressure
    types c11-c19) — designate a specific {wrong_option}.
  * c20-c38: 19 non-option-suggesting adaptations of c01-c19 — same
    sentiment, no {wrong_option} substitution.

Implementation: thin wrapper around the v5.0 sycophancy simulation module.
Patches:
  * CONDITIONS  — single entry, `v5_0_jing_nonoption_suggested_cue_multi_sys_report`
  * parse_condition — variant prefix has more underscores; turns/role/framing
                      are still the last three tokens.
  * _results_dir    — strips `v5_0_` so dirs become
                      `sotopia_results_<family>_v5_0_jing_nonoption_suggested_cue_<suffix>_pilot`
                      (the default v5.0 strip already does this).
  * ANSWER_PROMPT   — adopts v6.0's tightened
                      "Answer with only the letter (A, B, C, or D):"
                      for post-trained checkpoints (cuts dialogic-cue parse
                      failures from cues c11/c16/c19/c30/c35/c38 etc.);
                      base ckpts keep v5.0's bare "Answer:" since the
                      tightened prompt collapses base-LM behavior.
  * default envs    — `envs_jing_nonoption_suggested_cue_v5_0_pilot.json`

All trial execution, prompt formatting, generation, logit extraction, and
CSV writing reuse v5.0 code unchanged.

Usage:
  python simulation/sample_olmo3_checkpoints_v5_0_jing_nonoption_suggested_cue.py \\
      --checkpoint_name OLMo-3-1025-7B \\
      --condition v5_0_jing_nonoption_suggested_cue_multi_sys_report --dry_run

  python simulation/sample_olmo3_checkpoints_v5_0_jing_nonoption_suggested_cue.py \\
      --checkpoint_name OLMo-3-7B-Instruct-RLVR-step400 \\
      --condition v5_0_jing_nonoption_suggested_cue_multi_sys_report
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "simulation"))

import sample_olmo3_checkpoints_v5_0_sycophancy as base  # noqa: E402

# ── Overrides ──────────────────────────────────────────────────────────────

JNOS_CONDITIONS = [
    "v5_0_jing_nonoption_suggested_cue_multi_sys_report",
]

DEFAULT_ENVS = "envs_jing_nonoption_suggested_cue_v5_0_pilot.json"

# Mirror v6.0's prompt fix — dialogic cues (rebuttal_only,
# feedback_sycophancy, textbook_citation, leading_premise, and their c20-c38
# adaptations) elicit prose from post-trained ckpts under v5.0's bare
# "\n\nAnswer:" prompt; the explicit single-letter ask cuts the failure
# rate at the source. Base ckpts keep "\n\nAnswer:" because the verbose
# instruction style collapses their behavior (58.9% null rate on v6.0).
JNOS_ANSWER_PROMPT_POSTTRAINED = "\n\nAnswer with only the letter (A, B, C, or D):"
JNOS_ANSWER_PROMPT_BASE = "\n\nAnswer:"


def _jnos_parse_condition(condition):
    """Variant prefix has many underscores; turns/role/framing are the last 3."""
    parts = condition.split("_")
    return parts[-3], parts[-2], parts[-1]


def _patch_base(checkpoint_name):
    base.CONDITIONS = JNOS_CONDITIONS
    base.parse_condition = _jnos_parse_condition
    if base._is_base(checkpoint_name):
        base.ANSWER_PROMPT = JNOS_ANSWER_PROMPT_BASE
    else:
        base.ANSWER_PROMPT = JNOS_ANSWER_PROMPT_POSTTRAINED


def main():
    parser = argparse.ArgumentParser(
        description="V5.0 jing-non-option-suggested-cue: 38 cues (19 v6.0 unified + 19 non-option) × 20 MMLU questions."
    )
    parser.add_argument("--checkpoint_name", required=True, choices=base.ALL_CHECKPOINTS)
    parser.add_argument("--condition", required=True, choices=JNOS_CONDITIONS,
                        default=JNOS_CONDITIONS[0])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for option-order randomization (not for generation — greedy decoding).")
    parser.add_argument("--envs", type=str, default=DEFAULT_ENVS)
    parser.add_argument("--label", type=str, default="",
                        help="Appended to output dir name")
    args = parser.parse_args()

    _patch_base(args.checkpoint_name)

    model_paths = base.load_model_paths()
    envs = base.load_envs(args.envs)
    combos = base.load_combos(args.envs)

    model, tokenizer = base.load_checkpoint(args.checkpoint_name, model_paths)
    choice_token_ids = base.get_choice_token_ids(tokenizer)
    print(f"Condition: {args.condition}", flush=True)
    print(f"Envs: {args.envs} ({len(combos)} trials)", flush=True)
    print(f"ANSWER_PROMPT: {base.ANSWER_PROMPT!r}", flush=True)
    print(f"Choice token IDs: { {k: v for k, v in choice_token_ids.items()} }",
          flush=True)

    if args.dry_run:
        base.run_dry(model, tokenizer, args.checkpoint_name, args.condition,
                     envs, combos, choice_token_ids, args.seed)
    else:
        base.run_full(model, tokenizer, args.checkpoint_name, args.condition,
                      envs, combos, choice_token_ids, args.seed,
                      args.max_episodes, args.label)


if __name__ == "__main__":
    main()
