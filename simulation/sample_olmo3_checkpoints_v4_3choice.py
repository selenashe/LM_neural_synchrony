#!/usr/bin/env python3
"""
3-choice forced-choice evaluation of OLMo checkpoints on false-belief v4
pilot scenarios.

Each trial presents Ava with three options — two locations (randomized order)
and Unsure — then records the parsed choice, raw response, and logit
probabilities for each option token.

Usage:
  # Dry run:
  python simulation/sample_olmo3_checkpoints_v4_3choice.py \
      --checkpoint_name OLMo-3-7B-Instruct-SFT --dry_run

  # Full run:
  python simulation/sample_olmo3_checkpoints_v4_3choice.py \
      --checkpoint_name Olmo-3.1-32B-Instruct
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

# ── Checkpoint definitions ─────────────────────────────────────────────────

INSTRUCT_7B_CHECKPOINTS = [
    "OLMo-3-1025-7B",
    "OLMo-3-7B-Instruct-SFT",
    "OLMo-3-7B-Instruct-DPO",
    "OLMo-3-7B-Instruct-RLVR-step400",
]

THINK_7B_CHECKPOINTS = [
    "OLMo-3-7B-Think-SFT",
    "OLMo-3-7B-Think-DPO",
    "OLMo-3-7B-Think-RLVR-step1375",
]

INSTRUCT_32B_CHECKPOINTS = [
    "Olmo-3-1125-32B",
    "Olmo-3.1-32B-Instruct-SFT",
    "Olmo-3.1-32B-Instruct-DPO",
    "Olmo-3.1-32B-Instruct",
]

ALL_CHECKPOINTS = INSTRUCT_7B_CHECKPOINTS + THINK_7B_CHECKPOINTS + INSTRUCT_32B_CHECKPOINTS
BASE_CHECKPOINTS = {"OLMo-3-1025-7B", "Olmo-3-1125-32B"}

CSV_HEADER = [
    "episode_idx", "env_id", "scenario_id", "scenario_type", "cell_id",
    "slice", "locus", "specificity", "dosage", "prior", "goal_condition",
    "item", "location_truth", "location_other", "mia_message",
    "checkpoint", "seed",
    "truth_position", "option_a", "option_b",
    "raw_response", "parsed_letter",
    "choice", "prob_A", "prob_B", "prob_C",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_think(name):
    return name in THINK_7B_CHECKPOINTS


def _is_base(name):
    return name in BASE_CHECKPOINTS


def _is_32b(name):
    return name in INSTRUCT_32B_CHECKPOINTS


def _results_dir(name, suffix=""):
    if _is_32b(name):
        base = "sotopia_results_olmo31_32b_v4_3choice"
    elif _is_think(name):
        base = "sotopia_results_olmo3_think_v4_3choice"
    else:
        base = "sotopia_results_olmo3_instruct_v4_3choice"
    return base + suffix


# ── Data loading ───────────────────────────────────────────────────────────

def load_envs(envs_basename="envs_false_belief_v4_pilot.json"):
    with open(SOTOPIA_DATA / envs_basename) as f:
        return json.load(f)


def load_combos(envs_basename="envs_false_belief_v4_pilot.json"):
    combos_basename = envs_basename.replace("envs_", "env_agent_combos_")
    with open(SOTOPIA_DATA / combos_basename) as f:
        return json.load(f)


def load_model_paths():
    with open(REPO_ROOT / "model_paths.json") as f:
        return json.load(f)


# ── Model ──────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_name, model_paths):
    path = model_paths[checkpoint_name]
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

    print(f"Model loaded: {model.config.architectures}, dtype={model.dtype}",
          flush=True)
    return model, tokenizer


def format_prompt(raw_prompt, checkpoint_name):
    if _is_base(checkpoint_name):
        return raw_prompt
    if _is_think(checkpoint_name):
        return (
            f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )
    return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(model, tokenizer, formatted_prompt, max_new_tokens=50,
                      temperature=0.7, seed=0):
    torch.manual_seed(seed)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
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
    """Find token IDs for A, B, C (with and without space prefix)."""
    candidates = {}
    for letter in ["A", "B", "C"]:
        ids = []
        for variant in [letter, f" {letter}", f"Ġ{letter}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            ids.extend(encoded)
        # Also try encoding just the letter after "Answer: "
        context_ids = tokenizer.encode(f"Answer: {letter}", add_special_tokens=False)
        if context_ids:
            ids.append(context_ids[-1])
        candidates[letter] = list(set(ids))
    return candidates


def extract_logit_probs(model, tokenizer, formatted_prompt, choice_token_ids):
    """Forward pass to get softmax probabilities for A, B, C tokens."""
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_logits = outputs.logits[0, -1, :]

    all_ids = []
    id_to_letter = {}
    for letter, ids in choice_token_ids.items():
        for tid in ids:
            all_ids.append(tid)
            id_to_letter[tid] = letter

    probs = F.softmax(last_logits, dim=-1)

    letter_probs = {}
    for letter in ["A", "B", "C"]:
        token_ids = choice_token_ids[letter]
        if token_ids:
            letter_probs[letter] = max(probs[tid].item() for tid in token_ids)
        else:
            letter_probs[letter] = 0.0

    return letter_probs


# ── Prompt building ────────────────────────────────────────────────────────

def build_3choice_prompt(meta, episode_idx, seed):
    """Build prompt with randomized option ordering."""
    rng = random.Random(seed * 10000 + episode_idx)

    loc_truth = meta["location_truth"]
    loc_other = meta["location_other"]
    item = meta["item"]

    truth_first = rng.random() < 0.5
    if truth_first:
        option_a, option_b = loc_truth, loc_other
        truth_position = "A"
    else:
        option_a, option_b = loc_other, loc_truth
        truth_position = "B"

    # Capitalize location names for display
    option_a_display = option_a.replace("_", " ")
    option_b_display = option_b.replace("_", " ")

    prior_text = meta["prior_text"]
    cue_text = meta["cue_text"]
    mia_message = meta["mia_message"]
    ava_name = meta.get("ava_name", "Ava")
    mia_name = meta.get("mia_name", "Mia")

    prompt = (
        f"You are {ava_name}. {prior_text} {cue_text}\n\n"
        f"{mia_name} says: \"{mia_message}\"\n\n"
        f"Where do you think the {item} is? Respond with exactly one of:\n"
        f"(A) {option_a_display}\n"
        f"(B) {option_b_display}\n"
        f"(C) Unsure\n\n"
        f"Answer:"
    )

    return prompt, truth_position, option_a_display, option_b_display


def parse_3choice(text, truth_position):
    """Parse A/B/C from response and map to truth/other/unsure."""
    match = re.search(r'\b([A-C])\b', text)
    if not match:
        return None, "", ""
    letter = match.group(1)
    if letter == "C":
        return letter, "unsure", ""
    if letter == truth_position:
        return letter, "truth", ""
    return letter, "other", ""


# ── Dry run ────────────────────────────────────────────────────────────────

def run_dry(model, tokenizer, checkpoint_name, envs, combos,
            choice_token_ids, temperature, seed):
    is_think = _is_think(checkpoint_name)
    max_tokens = 2000 if is_think else 50

    seen_slices = set()
    count = 0

    for ep_idx, combo in enumerate(combos):
        env = envs[combo["env_id"]]
        meta = env["_meta"]
        sl = meta["slice"]
        if sl in seen_slices:
            continue
        seen_slices.add(sl)
        count += 1

        raw_prompt, truth_pos, opt_a, opt_b = build_3choice_prompt(
            meta, ep_idx, seed)
        formatted = format_prompt(raw_prompt, checkpoint_name)

        # Logit probs
        letter_probs = extract_logit_probs(
            model, tokenizer, formatted, choice_token_ids)

        # Generation
        raw_resp = generate_response(
            model, tokenizer, formatted, max_new_tokens=max_tokens,
            temperature=temperature, seed=seed,
        )
        if is_think:
            resp = strip_think_blocks(raw_resp)
        else:
            resp = raw_resp

        letter, choice, _ = parse_3choice(resp, truth_pos)

        print(f"\n{'='*80}")
        print(f"[{count}] {meta['cell_id']}  slice={sl}  "
              f"truth={meta['location_truth']}  goal={meta['goal_condition']}")
        print(f"  truth_position={truth_pos}  A={opt_a}  B={opt_b}")
        print(f"{'─'*80}")
        print(f"PROMPT:\n{raw_prompt}")
        print(f"{'─'*80}")
        print(f"RAW RESPONSE: {raw_resp}")
        print(f"PARSED: letter={letter}  choice={choice}")
        print(f"LOGITS: P(A)={letter_probs['A']:.4f}  "
              f"P(B)={letter_probs['B']:.4f}  P(C)={letter_probs['C']:.4f}")

        if len(seen_slices) >= 4:
            break

    print(f"\n{'='*80}")
    print(f"Dry run complete: {count} episodes across slices {sorted(seen_slices)}.")


# ── Full run ───────────────────────────────────────────────────────────────

def run_full(model, tokenizer, checkpoint_name, envs, combos,
             choice_token_ids, temperature, seed, max_episodes=None,
             results_suffix=""):
    is_think = _is_think(checkpoint_name)
    max_tokens = 2000 if is_think else 50

    results_dir = REPO_ROOT / _results_dir(checkpoint_name, results_suffix)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / f"results_{checkpoint_name}_seed{seed}.csv"

    total = len(combos)
    if max_episodes is not None:
        total = min(total, max_episodes)

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(int(row["episode_idx"]))
        print(f"Resuming: {len(done)} episodes already done", flush=True)

    print(f"Running {total} episodes, saving to {csv_path}", flush=True)

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

            raw_prompt, truth_pos, opt_a, opt_b = build_3choice_prompt(
                meta, ep_idx, seed)
            formatted = format_prompt(raw_prompt, checkpoint_name)

            # Logit probabilities
            letter_probs = extract_logit_probs(
                model, tokenizer, formatted, choice_token_ids)

            # Generation
            raw_resp = generate_response(
                model, tokenizer, formatted, max_new_tokens=max_tokens,
                temperature=temperature, seed=seed,
            )
            if is_think:
                resp = strip_think_blocks(raw_resp)
            else:
                resp = raw_resp

            letter, choice, _ = parse_3choice(resp, truth_pos)

            writer.writerow({
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
                "mia_message": meta["mia_message"],
                "checkpoint": checkpoint_name,
                "seed": seed,
                "truth_position": truth_pos,
                "option_a": opt_a,
                "option_b": opt_b,
                "raw_response": raw_resp,
                "parsed_letter": letter or "",
                "choice": choice,
                "prob_A": f"{letter_probs['A']:.6f}",
                "prob_B": f"{letter_probs['B']:.6f}",
                "prob_C": f"{letter_probs['C']:.6f}",
            })
            f.flush()

            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (total - ep_idx - 1) / rate if rate > 0 else 0
                print(
                    f"  [{ep_idx+1}/{total}] completed={completed} "
                    f"rate={rate:.1f} ep/s  ETA={remaining/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nDone: {completed} completed, {len(done)} previously done "
          f"in {elapsed:.1f}s", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-choice forced-choice evaluation on false-belief v4 pilot."
    )
    parser.add_argument(
        "--checkpoint_name", required=True, choices=ALL_CHECKPOINTS,
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--envs", type=str, default="envs_false_belief_v4_pilot.json",
        help="Envs JSON basename in sotopia_data/ "
             "(default: envs_false_belief_v4_pilot.json)",
    )
    parser.add_argument(
        "--results_suffix", type=str, default=None,
        help="Suffix appended to the results directory name "
             "(auto-detected from --envs if not set)",
    )
    args = parser.parse_args()

    if args.results_suffix is None:
        if "v4_pilot" in args.envs:
            args.results_suffix = ""
        elif "v4.json" in args.envs:
            args.results_suffix = "_full"
        else:
            stem = args.envs.replace("envs_false_belief_", "").replace(".json", "")
            args.results_suffix = f"_{stem}"

    model_paths = load_model_paths()
    envs = load_envs(args.envs)
    combos = load_combos(args.envs)

    model, tokenizer = load_checkpoint(args.checkpoint_name, model_paths)
    choice_token_ids = get_choice_token_ids(tokenizer)

    print(f"Choice token IDs: { {k: v for k, v in choice_token_ids.items()} }",
          flush=True)

    if args.dry_run:
        run_dry(model, tokenizer, args.checkpoint_name, envs, combos,
                choice_token_ids, args.temperature, args.seed)
    else:
        run_full(model, tokenizer, args.checkpoint_name, envs, combos,
                 choice_token_ids, args.temperature, args.seed,
                 args.max_episodes, args.results_suffix)


if __name__ == "__main__":
    main()
