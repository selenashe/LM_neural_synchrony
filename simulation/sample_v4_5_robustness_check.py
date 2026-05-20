#!/usr/bin/env python3
"""
V4.5 robustness check — run the same 2×2×2 factorial on non-OLMo models.

Uses tokenizer.apply_chat_template() for model-native prompt formatting,
handling system-prompt support and multi-turn conventions automatically.

Models:
  Qwen3-30B-A3B             (MoE, thinking model)
  Qwen3-8B                  (thinking model)
  Mistral-7B-Instruct-v0.3  (no system prompt; merged into user)
  DeepSeek-R1-Distill-Llama-8B  (thinking model)

Usage:
  python simulation/sample_v4_5_robustness_check.py \
      --model_name Qwen3-8B \
      --condition v4_5_single_sys_quote --dry_run

  python simulation/sample_v4_5_robustness_check.py \
      --model_name DeepSeek-R1-Distill-Llama-8B \
      --condition v4_5_multi_uo_report
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SOTOPIA_DATA = REPO_ROOT / "sotopia_utils" / "sotopia_data"

MODEL_CONFIGS = {
    "Qwen3-30B-A3B": {
        "path": "/juice6/u/jshe/nlp/hf_models/Qwen3-30B-A3B",
        "is_think": True,
    },
    "Qwen3-8B": {
        "path": "/juice6/u/jshe/nlp/hf_models/Qwen3-8B",
        "is_think": True,
    },
    "Mistral-7B-Instruct-v0.3": {
        "path": "/juice6/u/jshe/nlp/hf_models/Mistral-7B-Instruct-v0.3",
        "is_think": False,
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "path": "/juice6/u/jshe/nlp/hf_models/DeepSeek-R1-Distill-Llama-8B",
        "is_think": True,
    },
}

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


def parse_condition(condition):
    parts = condition.split("_")
    return parts[2], parts[3], parts[4]


# ── Data loading ───────────────────────────────────────────────────────────

def load_envs(envs_basename="envs_false_belief_v4_1_pilot.json"):
    with open(SOTOPIA_DATA / envs_basename) as f:
        return json.load(f)


def load_combos(envs_basename="envs_false_belief_v4_1_pilot.json"):
    combos_basename = envs_basename.replace("envs_", "env_agent_combos_")
    with open(SOTOPIA_DATA / combos_basename) as f:
        return json.load(f)


# ── Model ──────────────────────────────────────────────────────────────────

def load_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    path = cfg["path"]
    print(f"Loading tokenizer from {path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    print(f"Loading model from {path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    model.to("cuda").eval()
    print(f"Model loaded: {model.config.architectures}, dtype={model.dtype}", flush=True)
    return model, tokenizer


def generate_response(model, tokenizer, formatted_prompt, max_new_tokens=50,
                      temperature=0.7, seed=0):
    torch.manual_seed(seed)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def strip_think_blocks(text):
    closing = text.find("</think>")
    if closing >= 0:
        return text[closing + len("</think>"):].strip()
    return text.strip()


def get_choice_token_ids(tokenizer):
    candidates = {}
    for letter in ["A", "B", "C"]:
        ids = []
        for variant in [letter, f" {letter}", f"Ġ{letter}"]:
            ids.extend(tokenizer.encode(variant, add_special_tokens=False))
        context_ids = tokenizer.encode(f"Answer: {letter}", add_special_tokens=False)
        if context_ids:
            ids.append(context_ids[-1])
        candidates[letter] = list(set(ids))
    return candidates


def extract_logit_probs(model, tokenizer, formatted_prompt, choice_token_ids):
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    return {
        letter: max(probs[tid].item() for tid in tids) if tids else 0.0
        for letter, tids in choice_token_ids.items()
    }


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


# ── Prompt building ────────────────────────────────────────────────────────

def build_choice_block(question, option_a, option_b):
    return (
        f"{question}\n\n"
        f"Respond with exactly one of:\n"
        f"(A) {option_a}\n"
        f"(B) {option_b}\n"
        f"(C) Unsure\n\n"
        f"Answer:"
    )


def build_option_order(meta, episode_idx, seed):
    rng = random.Random(seed * 10000 + episode_idx)
    loc_truth = meta["location_truth"]
    loc_other = meta["location_other"]
    truth_first = rng.random() < 0.5
    if truth_first:
        option_a, option_b = loc_truth, loc_other
        truth_position = "A"
    else:
        option_a, option_b = loc_other, loc_truth
        truth_position = "B"
    return truth_position, option_a.replace("_", " "), option_b.replace("_", " ")


def _identity_text(meta):
    return f"You are {meta['ava_name']}. {meta['prior_text']}"


def _contradiction_text(meta, framing):
    if meta["locus"] == "sycophancy":
        return meta["turn3_premise"]
    cue = meta.get("cue_text", "").strip()
    if framing == "quote":
        mia = f'Mia said: "{meta["mia_message"]}"'
    elif framing == "report":
        loc = (meta["location_truth_phrase"]
               if meta.get("goal_condition") == "cooperate"
               else meta["location_other_phrase"])
        mia = f"Mia told me the {meta['item']} is {loc}."
    else:
        raise ValueError(f"unknown framing: {framing}")
    if cue:
        return f"{mia} {cue}"
    return mia


def _question_block(meta, opt_a, opt_b):
    question = f"Where do you think the {meta['item']} is?"
    return build_choice_block(question, opt_a, opt_b)


# ── Prompt assembly via apply_chat_template ───────────────────────────────

def _apply_template(tokenizer, messages):
    """Apply the tokenizer's chat template and return the formatted string."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def format_single_prompt(tokenizer, meta, opt_a, opt_b, condition):
    _, role, framing = parse_condition(condition)
    identity = _identity_text(meta)
    contradiction = _contradiction_text(meta, framing)
    qblock = _question_block(meta, opt_a, opt_b)

    if role == "sys":
        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": f"{contradiction}\n\n{qblock}"},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{identity}\n\n{contradiction}\n\n{qblock}"},
        ]
    return _apply_template(tokenizer, messages)


def format_multi_pre_prompt(tokenizer, meta, opt_a, opt_b, condition):
    _, role, _ = parse_condition(condition)
    identity = _identity_text(meta)
    qblock = _question_block(meta, opt_a, opt_b)

    if role == "sys":
        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": qblock},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{identity}\n\n{qblock}"},
        ]
    return _apply_template(tokenizer, messages)


def format_multi_post_prompt(tokenizer, pre_response, meta, opt_a, opt_b,
                             condition):
    _, role, framing = parse_condition(condition)
    identity = _identity_text(meta)
    contradiction = _contradiction_text(meta, framing)
    qblock = _question_block(meta, opt_a, opt_b)
    user_body = f"{contradiction}\n\n{qblock}"

    if role == "sys":
        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": _question_block(meta, opt_a, opt_b)},
            {"role": "assistant", "content": pre_response},
            {"role": "user", "content": user_body},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{identity}\n\n{_question_block(meta, opt_a, opt_b)}"},
            {"role": "assistant", "content": pre_response},
            {"role": "user", "content": user_body},
        ]
    return _apply_template(tokenizer, messages)


# ── Per-trial execution ────────────────────────────────────────────────────

def _empty_turn_record():
    return {
        "raw": "", "letter": "", "choice": "",
        "probs": {"A": float("nan"), "B": float("nan"), "C": float("nan")},
    }


def run_trial(model, tokenizer, meta, opt_a, opt_b, truth_pos, condition,
              model_name, choice_token_ids, temperature, seed, max_tokens):
    is_think = MODEL_CONFIGS[model_name]["is_think"]
    turns, _, _ = parse_condition(condition)
    out = {"turn2": _empty_turn_record(), "turn4": _empty_turn_record()}

    if turns == "single":
        prompt = format_single_prompt(tokenizer, meta, opt_a, opt_b, condition)
        probs = extract_logit_probs(model, tokenizer, prompt, choice_token_ids)
        raw = generate_response(model, tokenizer, prompt, max_new_tokens=max_tokens,
                                temperature=temperature, seed=seed)
        resp = strip_think_blocks(raw) if is_think else raw
        letter, choice = parse_3choice(resp, truth_pos)
        out["turn4"] = {"raw": raw, "letter": letter or "", "choice": choice, "probs": probs}
        return out, prompt

    # multi-turn
    pre_prompt = format_multi_pre_prompt(tokenizer, meta, opt_a, opt_b, condition)
    pre_probs = extract_logit_probs(model, tokenizer, pre_prompt, choice_token_ids)
    pre_raw = generate_response(model, tokenizer, pre_prompt, max_new_tokens=max_tokens,
                                temperature=temperature, seed=seed)
    pre_resp = strip_think_blocks(pre_raw) if is_think else pre_raw
    pre_letter, pre_choice = parse_3choice(pre_resp, truth_pos)
    out["turn2"] = {"raw": pre_raw, "letter": pre_letter or "", "choice": pre_choice,
                    "probs": pre_probs}

    post_prompt = format_multi_post_prompt(
        tokenizer, pre_raw, meta, opt_a, opt_b, condition,
    )
    post_probs = extract_logit_probs(model, tokenizer, post_prompt, choice_token_ids)
    post_raw = generate_response(model, tokenizer, post_prompt, max_new_tokens=max_tokens,
                                 temperature=temperature, seed=seed)
    post_resp = strip_think_blocks(post_raw) if is_think else post_raw
    post_letter, post_choice = parse_3choice(post_resp, truth_pos)
    out["turn4"] = {"raw": post_raw, "letter": post_letter or "", "choice": post_choice,
                    "probs": post_probs}
    return out, post_prompt


# ── Dry run ───────────────────────────────────────────────────────────────

def run_dry(model, tokenizer, model_name, condition, envs, combos,
            choice_token_ids, temperature, seed):
    is_think = MODEL_CONFIGS[model_name]["is_think"]
    max_tokens = 2000 if is_think else 50
    seen, count = set(), 0

    for ep_idx, combo in enumerate(combos):
        env = envs[combo["env_id"]]
        meta = env["_meta"]
        key = (meta["slice"], meta["locus"])
        if key in seen:
            continue
        seen.add(key)
        count += 1

        truth_pos, opt_a, opt_b = build_option_order(meta, ep_idx, seed)
        result, final_prompt = run_trial(
            model, tokenizer, meta, opt_a, opt_b, truth_pos, condition,
            model_name, choice_token_ids, temperature, seed, max_tokens,
        )

        _, _, framing = parse_condition(condition)
        contradiction = _contradiction_text(meta, framing)

        print(f"\n{'='*80}")
        print(f"[{count}] {meta['cell_id']}  slice={meta['slice']}  "
              f"locus={meta['locus']}  goal={meta['goal_condition']}")
        print(f"  truth_pos={truth_pos}  A={opt_a}  B={opt_b}")
        print(f"  contradiction text: {contradiction!r}")
        print(f"FINAL PROMPT:\n{final_prompt}")
        print(f"{'─'*80}")
        if result["turn2"]["choice"]:
            t2 = result["turn2"]
            print(f"T2 RESPONSE: {t2['raw'][:200]}")
            print(f"  parsed={t2['letter']} ({t2['choice']})  "
                  f"P(A)={t2['probs']['A']:.4f} P(B)={t2['probs']['B']:.4f} P(C)={t2['probs']['C']:.4f}")
        t4 = result["turn4"]
        print(f"T4 RESPONSE: {t4['raw'][:200]}")
        print(f"  parsed={t4['letter']} ({t4['choice']})  "
              f"P(A)={t4['probs']['A']:.4f} P(B)={t4['probs']['B']:.4f} P(C)={t4['probs']['C']:.4f}")
        if count >= 6:
            break

    print(f"\n{'='*80}\nDry run complete: {count} episodes.")


# ── Full run ──────────────────────────────────────────────────────────────

def _results_dir(model_name, condition):
    suffix = condition[len("v4_5_"):]
    return f"sotopia_results_robustness_v4_5_{suffix}"


def run_full(model, tokenizer, model_name, condition, envs, combos,
             choice_token_ids, temperature, seed, max_episodes=None):
    is_think = MODEL_CONFIGS[model_name]["is_think"]
    max_tokens = 2000 if is_think else 50
    turns, role, framing = parse_condition(condition)

    results_dir = REPO_ROOT / _results_dir(model_name, condition)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / f"results_{model_name}_seed{seed}.csv"

    total = min(len(combos), max_episodes) if max_episodes else len(combos)

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add(int(row["episode_idx"]))
        print(f"Resuming: {len(done)} episodes already done", flush=True)

    write_header = not csv_path.exists() or len(done) == 0
    t0 = time.time()
    completed = 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()

        for ep_idx in range(total):
            if ep_idx in done:
                continue
            env = envs[combos[ep_idx]["env_id"]]
            meta = env["_meta"]
            truth_pos, opt_a, opt_b = build_option_order(meta, ep_idx, seed)

            result, _ = run_trial(
                model, tokenizer, meta, opt_a, opt_b, truth_pos, condition,
                model_name, choice_token_ids, temperature, seed, max_tokens,
            )

            t2, t4 = result["turn2"], result["turn4"]
            row = {
                "episode_idx": ep_idx,
                "env_id": combos[ep_idx]["env_id"],
                "scenario_id": meta["scenario_id"],
                "scenario_type": meta["scenario_type"],
                "cell_id": meta["cell_id"],
                "slice": meta["slice"],
                "locus": meta["locus"],
                "specificity": meta["specificity"],
                "dosage": meta["dosage"],
                "prior": meta["prior"],
                "goal_condition": meta["goal_condition"],
                "item": meta["item"],
                "location_truth": meta["location_truth"],
                "location_other": meta["location_other"],
                "checkpoint": model_name,
                "seed": seed,
                "condition": condition,
                "turns": turns,
                "role": role,
                "framing": framing,
                "truth_position": truth_pos,
                "option_a": opt_a,
                "option_b": opt_b,
                "turn3_contradiction": _contradiction_text(meta, framing),
                "turn2_raw_response": t2["raw"],
                "turn2_parsed_letter": t2["letter"],
                "turn2_choice": t2["choice"],
                "turn2_prob_A": "" if t2["choice"] == "" else f"{t2['probs']['A']:.6f}",
                "turn2_prob_B": "" if t2["choice"] == "" else f"{t2['probs']['B']:.6f}",
                "turn2_prob_C": "" if t2["choice"] == "" else f"{t2['probs']['C']:.6f}",
                "turn4_raw_response": t4["raw"],
                "turn4_parsed_letter": t4["letter"],
                "turn4_choice": t4["choice"],
                "turn4_prob_A": f"{t4['probs']['A']:.6f}",
                "turn4_prob_B": f"{t4['probs']['B']:.6f}",
                "turn4_prob_C": f"{t4['probs']['C']:.6f}",
            }
            writer.writerow(row)
            f.flush()
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (total - ep_idx - 1) / rate if rate > 0 else 0
                print(f"  [{ep_idx+1}/{total}] completed={completed} "
                      f"rate={rate:.1f} ep/s  ETA={remaining/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {completed} completed, {len(done)} previously done "
          f"in {elapsed:.1f}s", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V4.5 robustness check: non-OLMo models, 2x2x2 factorial."
    )
    parser.add_argument("--model_name", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=str, default="envs_false_belief_v4_1_pilot.json")
    args = parser.parse_args()

    envs = load_envs(args.envs)
    combos = load_combos(args.envs)

    model, tokenizer = load_model(args.model_name)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Condition: {args.condition}", flush=True)
    print(f"Envs: {args.envs}", flush=True)
    print(f"Is think model: {MODEL_CONFIGS[args.model_name]['is_think']}", flush=True)
    print(f"Choice token IDs: { {k: v for k, v in choice_token_ids.items()} }",
          flush=True)

    if args.dry_run:
        run_dry(model, tokenizer, args.model_name, args.condition,
                envs, combos, choice_token_ids, args.temperature, args.seed)
    else:
        run_full(model, tokenizer, args.model_name, args.condition,
                 envs, combos, choice_token_ids, args.temperature, args.seed,
                 args.max_episodes)


if __name__ == "__main__":
    main()
