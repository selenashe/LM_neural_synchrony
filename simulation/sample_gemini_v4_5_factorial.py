#!/usr/bin/env python3
"""
V4.5 — 2×2×2 factorial (turns × role × framing) — Gemini / Vertex AI.

Same 8 conditions and CSV schema as sample_olmo3_checkpoints_v4_5_factorial.py,
but uses litellm → Vertex AI instead of local HF weights.

Prompt structure mirrors the OLMo script exactly:
  sys conditions  → system message (identity) + user message (contradiction + question)
  uo conditions   → single user message (identity + contradiction + question, no system)
  multi-turn      → two API calls: pre-cue (T2) then post-cue (T4)
  sycophancy locus → framing axis is degenerate; same content for quote/report

Logprobs: requests top_logprobs=20 from the Vertex API and extracts P(A), P(B), P(C)
from the first generated token. Falls back to NaN if the API doesn't return them.

Requires Vertex AI auth (same as sample_gemini_false_belief.py):
  gcloud auth application-default login
  gcloud config set project hs-soil-gemini

Usage:
  # Dry run (print one prompt per unique condition, no API calls)
  python simulation/sample_gemini_v4_5_factorial.py \
      --condition v4_5_single_sys_quote --dry_run

  # Pilot (10-item)
  python simulation/sample_gemini_v4_5_factorial.py \
      --condition v4_5_single_sys_quote \
      --envs envs_false_belief_v4_1_pilot.json --label pilot

  # Full (100-item)
  python simulation/sample_gemini_v4_5_factorial.py \
      --condition v4_5_multi_uo_report \
      --envs envs_false_belief_v4_1.json --label full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from litellm import completion
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SOTOPIA_DATA = REPO_ROOT / "sotopia_utils" / "sotopia_data"

# ── Vertex AI defaults (same as gemini_sot_env.py) ─────────────────────────

DEFAULT_VERTEX_MODEL    = "vertex_ai/gemini-2.5-pro-preview-05-06"
DEFAULT_VERTEX_PROJECT  = "hs-soil-gemini"
DEFAULT_VERTEX_LOCATION = "global"
DEFAULT_MAX_ATTEMPTS    = 5

# ── Conditions ──────────────────────────────────────────────────────────────

CONDITIONS = [
    "v4_5_single_sys_quote",
    "v4_5_single_sys_report",
    "v4_5_single_uo_quote",
    "v4_5_single_uo_report",
    "v4_5_multi_sys_quote",
    "v4_5_multi_sys_report",
    "v4_5_multi_uo_quote",
    "v4_5_multi_uo_report",
]

# ── CSV schema (identical to OLMo script) ───────────────────────────────────

CSV_HEADER = [
    "episode_idx", "env_id", "scenario_id", "scenario_type", "cell_id",
    "slice", "locus", "specificity", "dosage", "prior", "goal_condition",
    "item", "location_truth", "location_other",
    "checkpoint", "seed", "condition", "turns", "role", "framing",
    "truth_position", "option_a", "option_b",
    "turn3_contradiction",
    "turn2_raw_response", "turn2_parsed_letter", "turn2_choice",
    "turn2_prob_A", "turn2_prob_B", "turn2_prob_C",
    "turn4_raw_response", "turn4_parsed_letter", "turn4_choice",
    "turn4_prob_A", "turn4_prob_B", "turn4_prob_C",
]


# ── Utilities (copied from OLMo script; no torch dependency) ────────────────

def parse_condition(condition):
    """Returns (turns, role, framing) — e.g. ('single', 'sys', 'quote')."""
    parts = condition.split("_")  # v4_5_{turns}_{role}_{framing}
    return parts[2], parts[3], parts[4]


def load_envs(envs_basename="envs_false_belief_v4_1_pilot.json"):
    with open(SOTOPIA_DATA / envs_basename) as f:
        return json.load(f)


def load_combos(envs_basename="envs_false_belief_v4_1_pilot.json"):
    combos_basename = envs_basename.replace("envs_", "env_agent_combos_")
    with open(SOTOPIA_DATA / combos_basename) as f:
        return json.load(f)


def build_option_order(meta, episode_idx, seed):
    rng = random.Random(seed * 10000 + episode_idx)
    loc_truth = meta["location_truth"]
    loc_other  = meta["location_other"]
    truth_first = rng.random() < 0.5
    if truth_first:
        option_a, option_b = loc_truth, loc_other
        truth_position = "A"
    else:
        option_a, option_b = loc_other, loc_truth
        truth_position = "B"
    return truth_position, option_a.replace("_", " "), option_b.replace("_", " ")


def build_choice_block(question, option_a, option_b):
    return (
        f"{question}\n\n"
        f"Respond with exactly one of:\n"
        f"(A) {option_a}\n"
        f"(B) {option_b}\n"
        f"(C) Unsure\n\n"
        f"Answer:"
    )


def _identity_text(meta):
    return f"You are {meta['ava_name']}. {meta['prior_text']}"


def _contradiction_text(meta, framing):
    if meta["locus"] == "sycophancy":
        return meta["turn3_premise"]
    cue = meta.get("cue_text", "").strip()
    if framing == "quote":
        mia = f'Mia said: "{meta["mia_message"]}"'
    elif framing == "report":
        mia = f"Mia told me the {meta['item']} is {meta['location_other_phrase']}."
    else:
        raise ValueError(f"unknown framing: {framing!r}")
    return f"{mia} {cue}" if cue else mia


def _question_block(meta, opt_a, opt_b):
    question = f"Where do you think the {meta['item']} is?"
    return build_choice_block(question, opt_a, opt_b)


def parse_3choice(text, truth_position):
    match = re.search(r'\b([A-C])\b', text)
    if not match:
        return None, ""
    letter = match.group(1)
    if letter == "C":
        return letter, "unsure"
    if letter == truth_position:
        return letter, "truth"
    return letter, "other"


# ── Message builders (Gemini native messages format, no ChatML) ──────────────

def _build_single_messages(meta, opt_a, opt_b, condition):
    """Build the messages list for a single-turn trial."""
    _, role, framing = parse_condition(condition)
    identity     = _identity_text(meta)
    contradiction = _contradiction_text(meta, framing)
    qblock        = _question_block(meta, opt_a, opt_b)

    if role == "sys":
        return [
            {"role": "system", "content": identity},
            {"role": "user",   "content": f"{contradiction}\n\n{qblock}"},
        ]
    # uo: everything in user, no system message
    return [
        {"role": "user", "content": f"{identity}\n\n{contradiction}\n\n{qblock}"},
    ]


def _build_multi_pre_messages(meta, opt_a, opt_b, condition):
    """Messages for the pre-cue turn (T1 user → T2 assistant)."""
    _, role, _ = parse_condition(condition)
    identity = _identity_text(meta)
    qblock   = _question_block(meta, opt_a, opt_b)

    if role == "sys":
        return [
            {"role": "system", "content": identity},
            {"role": "user",   "content": qblock},
        ]
    return [
        {"role": "user", "content": f"{identity}\n\n{qblock}"},
    ]


def _build_multi_post_messages(pre_messages, pre_response, meta, opt_a, opt_b, condition):
    """Extend pre-cue messages with the assistant reply + T3 user message."""
    _, _, framing = parse_condition(condition)
    contradiction = _contradiction_text(meta, framing)
    qblock        = _question_block(meta, opt_a, opt_b)
    return pre_messages + [
        {"role": "assistant", "content": pre_response},
        {"role": "user",      "content": f"{contradiction}\n\n{qblock}"},
    ]


# ── Vertex AI generation ─────────────────────────────────────────────────────

def _vertex_call(
    messages: list,
    *,
    vertex_model: str,
    vertex_project: str,
    vertex_location: str,
    temperature: float,
    max_tokens: int,
    want_logprobs: bool,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> tuple[str, dict]:
    """Call Vertex AI via litellm.  Returns (raw_text, probs) where probs is
    {'A': float, 'B': float, 'C': float} (NaN if not available)."""
    nan = float("nan")
    probs = {"A": nan, "B": nan, "C": nan}
    last_err: Optional[BaseException] = None

    for attempt in range(max_attempts):
        try:
            kwargs: dict = dict(
                model=vertex_model,
                messages=messages,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
                num_retries=2,
            )
            if want_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 20

            resp = completion(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()

            # Extract first-token logprobs for A / B / C
            if want_logprobs:
                try:
                    top = resp.choices[0].logprobs.content[0].top_logprobs
                    lp_map = {entry.token.strip(): entry.logprob for entry in top}
                    for letter in ["A", "B", "C"]:
                        if letter in lp_map:
                            probs[letter] = math.exp(lp_map[letter])
                except Exception:
                    pass  # logprobs unavailable; probs stay NaN

            return raw, probs

        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  Vertex error (attempt {attempt + 1}/{max_attempts}): {e!r}; "
                  f"sleeping {wait}s", flush=True)
            time.sleep(wait)

    assert last_err is not None
    raise last_err


# ── Per-trial execution ──────────────────────────────────────────────────────

def _empty_turn():
    nan = float("nan")
    return {"raw": "", "letter": "", "choice": "",
            "probs": {"A": nan, "B": nan, "C": nan}}


def run_trial_gemini(meta, opt_a, opt_b, truth_pos, condition,
                     vertex_model, vertex_project, vertex_location,
                     temperature, want_logprobs=True):
    """Returns dict with keys turn2 and turn4, same structure as OLMo script.
    Single-turn conditions leave turn2 blank."""
    turns, _, _ = parse_condition(condition)
    out = {"turn2": _empty_turn(), "turn4": _empty_turn()}

    if turns == "single":
        msgs = _build_single_messages(meta, opt_a, opt_b, condition)
        raw, probs = _vertex_call(
            msgs, vertex_model=vertex_model, vertex_project=vertex_project,
            vertex_location=vertex_location, temperature=temperature,
            max_tokens=50, want_logprobs=want_logprobs,
        )
        letter, choice = parse_3choice(raw, truth_pos)
        out["turn4"] = {"raw": raw, "letter": letter or "", "choice": choice, "probs": probs}
        return out, msgs

    # multi-turn: T2 (pre-cue) then T4 (post-cue)
    pre_msgs = _build_multi_pre_messages(meta, opt_a, opt_b, condition)
    pre_raw, pre_probs = _vertex_call(
        pre_msgs, vertex_model=vertex_model, vertex_project=vertex_project,
        vertex_location=vertex_location, temperature=temperature,
        max_tokens=50, want_logprobs=want_logprobs,
    )
    pre_letter, pre_choice = parse_3choice(pre_raw, truth_pos)
    out["turn2"] = {"raw": pre_raw, "letter": pre_letter or "", "choice": pre_choice,
                    "probs": pre_probs}

    post_msgs = _build_multi_post_messages(pre_msgs, pre_raw, meta, opt_a, opt_b, condition)
    post_raw, post_probs = _vertex_call(
        post_msgs, vertex_model=vertex_model, vertex_project=vertex_project,
        vertex_location=vertex_location, temperature=temperature,
        max_tokens=50, want_logprobs=want_logprobs,
    )
    post_letter, post_choice = parse_3choice(post_raw, truth_pos)
    out["turn4"] = {"raw": post_raw, "letter": post_letter or "", "choice": post_choice,
                    "probs": post_probs}
    return out, post_msgs


# ── Dry run ──────────────────────────────────────────────────────────────────

def run_dry(condition, envs, combos):
    _, role, framing = parse_condition(condition)
    seen, count = set(), 0

    for ep_idx, combo in enumerate(combos):
        env  = envs[combo["env_id"]]
        meta = env["_meta"]
        key  = (meta["slice"], meta["locus"])
        if key in seen:
            continue
        seen.add(key)
        count += 1

        truth_pos, opt_a, opt_b = build_option_order(meta, ep_idx, seed=0)
        turns, _, _ = parse_condition(condition)

        if turns == "single":
            msgs = _build_single_messages(meta, opt_a, opt_b, condition)
            header = "=== single-turn ==="
        else:
            pre_msgs  = _build_multi_pre_messages(meta, opt_a, opt_b, condition)
            post_msgs = _build_multi_post_messages(
                pre_msgs, "{pre_cue_response}", meta, opt_a, opt_b, condition)
            msgs = post_msgs
            header = "=== multi-turn (post-cue messages) ==="

        print(f"\n{'='*60}")
        print(f"Condition: {condition} | slice={meta['slice']} locus={meta['locus']}")
        print(header)
        for m in msgs:
            role_tag = m["role"].upper()
            print(f"[{role_tag}] {m['content']}")

    print(f"\nDry run complete: {count} unique (slice, locus) cells shown.")


# ── Full run ─────────────────────────────────────────────────────────────────

def _results_dir(model_key, condition, label=""):
    suffix      = condition[len("v4_5_"):]
    label_suffix = f"_{label}" if label else ""
    return REPO_ROOT / f"sotopia_results_{model_key}_v4_5_{suffix}{label_suffix}"


def run_full(condition, envs, combos, model_key,
             vertex_model, vertex_project, vertex_location,
             temperature, seed, max_episodes=None, label="",
             want_logprobs=True):
    turns, role, framing = parse_condition(condition)

    results_dir = _results_dir(model_key, condition, label)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / f"results_{model_key}_seed{seed}.csv"

    total = min(len(combos), max_episodes) if max_episodes else len(combos)

    # Resume support
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(int(row["episode_idx"]))
        print(f"Resuming: {len(done)} episodes already done", flush=True)

    write_header = not csv_path.exists() or len(done) == 0
    remaining = total - len(done)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()

        pbar = tqdm(total=remaining, desc=condition, unit="ep")
        for ep_idx in range(total):
            if ep_idx in done:
                continue
            env  = envs[combos[ep_idx]["env_id"]]
            meta = env["_meta"]
            truth_pos, opt_a, opt_b = build_option_order(meta, ep_idx, seed)

            result, _ = run_trial_gemini(
                meta, opt_a, opt_b, truth_pos, condition,
                vertex_model=vertex_model,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                temperature=temperature,
                want_logprobs=want_logprobs,
            )

            t2 = result["turn2"]
            t4 = result["turn4"]
            contradiction = _contradiction_text(meta, framing)

            writer.writerow({
                "episode_idx":      ep_idx,
                "env_id":           combos[ep_idx]["env_id"],
                "scenario_id":      meta.get("scenario_id", ""),
                "scenario_type":    meta.get("scenario_type", ""),
                "cell_id":          meta.get("cell_id", ""),
                "slice":            meta.get("slice", ""),
                "locus":            meta.get("locus", ""),
                "specificity":      meta.get("specificity", ""),
                "dosage":           meta.get("dosage", ""),
                "prior":            meta.get("prior", ""),
                "goal_condition":   meta.get("goal_condition", ""),
                "item":             meta.get("item", ""),
                "location_truth":   meta.get("location_truth", ""),
                "location_other":   meta.get("location_other", ""),
                "checkpoint":       model_key,
                "seed":             seed,
                "condition":        condition,
                "turns":            turns,
                "role":             role,
                "framing":          framing,
                "truth_position":   truth_pos,
                "option_a":         opt_a,
                "option_b":         opt_b,
                "turn3_contradiction": contradiction,
                "turn2_raw_response":  t2["raw"],
                "turn2_parsed_letter": t2["letter"],
                "turn2_choice":        t2["choice"],
                "turn2_prob_A":        t2["probs"]["A"],
                "turn2_prob_B":        t2["probs"]["B"],
                "turn2_prob_C":        t2["probs"]["C"],
                "turn4_raw_response":  t4["raw"],
                "turn4_parsed_letter": t4["letter"],
                "turn4_choice":        t4["choice"],
                "turn4_prob_A":        t4["probs"]["A"],
                "turn4_prob_B":        t4["probs"]["B"],
                "turn4_prob_C":        t4["probs"]["C"],
            })
            f.flush()
            pbar.update(1)

        pbar.close()
    print(f"Done: {csv_path}", flush=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4.5 2×2×2 factorial for Gemini via Vertex AI."
    )
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--envs", type=str, default="envs_false_belief_v4_1_pilot.json",
                        help="Env JSON filename (in sotopia_utils/sotopia_data/).")
    parser.add_argument("--label", type=str, default="",
                        help="Appended to output dir: sotopia_results_{model_key}_v4_5_{cond}_{label}/")
    parser.add_argument("--model_key", type=str, default="gemini",
                        help="Short name used in results dir and 'checkpoint' column.")
    parser.add_argument("--vertex_model", type=str, default=DEFAULT_VERTEX_MODEL,
                        help="LiteLLM model id (default: vertex_ai/gemini-2.5-pro-preview-05-06).")
    parser.add_argument("--vertex_project",  type=str, default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--vertex_location", type=str, default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--max_episodes", type=int,   default=None)
    parser.add_argument("--no_logprobs",  action="store_true",
                        help="Skip logprob extraction (faster; probs columns will be NaN).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print one prompt per unique (slice, locus) cell; no API calls.")
    args = parser.parse_args()

    envs   = load_envs(args.envs)
    combos = load_combos(args.envs)

    print(f"Condition:    {args.condition}", flush=True)
    print(f"Envs:         {args.envs}  ({len(combos)} combos)", flush=True)
    print(f"Vertex model: {args.vertex_model}", flush=True)
    print(f"Model key:    {args.model_key}", flush=True)
    if args.label:
        print(f"Label:        {args.label}", flush=True)

    if args.dry_run:
        run_dry(args.condition, envs, combos)
        return

    run_full(
        condition=args.condition,
        envs=envs,
        combos=combos,
        model_key=args.model_key,
        vertex_model=args.vertex_model,
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        temperature=args.temperature,
        seed=args.seed,
        max_episodes=args.max_episodes,
        label=args.label,
        want_logprobs=not args.no_logprobs,
    )


if __name__ == "__main__":
    main()
