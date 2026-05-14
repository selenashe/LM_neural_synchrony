#!/usr/bin/env python3
"""
Single-prompt completion evaluation of OLMo-2-1124-7B checkpoints on
false-belief v3 scenarios.

Instead of two-agent roleplay, Mia's opening is fixed to a deterministic
template ("The {item} is {prep} {location}.") and the model completes only
Ava's response.  This isolates comprehension / belief-vigilance ability
from turn-dynamics failures.

Usage:
  # Dry run (24 episodes, one per condition, prints prompt + response):
  python simulation/sample_olmo2_checkpoints.py \
      --checkpoint_name OLMo-2-1124-7B-Instruct --dry_run

  # Full run:
  python simulation/sample_olmo2_checkpoints.py \
      --checkpoint_name OLMo-2-1124-7B-Instruct
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SOTOPIA_DATA = REPO_ROOT / "sotopia_utils" / "sotopia_data"

CHECKPOINT_NAMES = [
    "OLMo-2-1124-7B-Base",
    "OLMo-2-1124-7B-SFT",
    "OLMo-2-1124-7B-DPO",
    "OLMo-2-1124-7B-Instruct",
    "OLMo-2-1124-7B-RLVR-step60",
    "OLMo-2-1124-7B-RLVR-step120",
    "OLMo-2-1124-7B-RLVR-step180",
    "OLMo-2-1124-7B-RLVR-step240",
    "OLMo-2-1124-7B-RLVR-step300",
    "OLMo-2-1124-7B-RLVR-step360",
]

BASE_CHECKPOINT = "OLMo-2-1124-7B-Base"

RESULTS_DIR_NAME = "sotopia_results_olmo2_checkpoints_v3"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_model_paths():
    with open(REPO_ROOT / "model_paths.json") as f:
        return json.load(f)


def load_envs():
    with open(SOTOPIA_DATA / "envs_false_belief_v3.json") as f:
        return json.load(f)


def load_combos():
    with open(
        SOTOPIA_DATA / "env_agent_combos_false_belief_v3_fixed_two_agents.json"
    ) as f:
        return json.load(f)


def load_scenario_vocab():
    with open(SOTOPIA_DATA / "scenario_vocab_false_belief_v2.json") as f:
        return json.load(f)


# ── Prompt construction ──────────────────────────────────────────────────────

def get_mia_opening(env, scenario_vocab):
    """Build Mia's fixed opening: 'The {item} is {prep} {location}.'"""
    meta = env["_meta"]
    vocab = scenario_vocab[meta["scenario_type"]]
    item = vocab["item"]
    prep = vocab["prep"]
    locations = vocab["locations"]
    true_location = meta["true_location"].replace("_", " ")

    if meta["a_gold"] == "say_false":
        stated_location = [loc for loc in locations if loc != true_location][0]
    else:
        stated_location = true_location

    return f"The {item} is {prep} {stated_location}."


def build_raw_prompt(env, mia_opening):
    """Build the raw prompt text (before chat-template wrapping)."""
    ava_goal = env["agent_goals"][1]
    scenario = env["scenario"]

    return (
        f"You are Ava Thompson. Respond only with short, direct dialogue.\n"
        f"Scenario: {scenario}\n"
        f"Participants: Mia Sanders and Ava Thompson\n"
        f"Ava Thompson's goal: {ava_goal}\n"
        f"Mia Sanders's goal: Unknown\n"
        f"Respond only with dialogue. Do not narrate actions or describe "
        f"thoughts. Once you have decided which location to check, you should "
        f"state your final choice and leave the conversation by saying LEAVE. "
        f"Keep your responses concise.\n"
        f"Turn 0: Mia Sanders said:{mia_opening}\n"
        f"Turn 1: Ava Thompson said:"
    )


def build_intro(env):
    """Build the Intro field for the output CSV (matches existing format)."""
    scenario = env["scenario"]
    mia_goal = env["agent_goals"][0]
    ava_goal = env["agent_goals"][1]
    return (
        f"Here is the context of this interaction:\n"
        f"Scenario: {scenario}\n"
        f"Participants: Mia Sanders and Ava Thompson\n"
        f"Mia Sanders's goal: {mia_goal}\n"
        f"Ava Thompson's goal: {ava_goal}"
    )


def format_prompt(raw_prompt, tokenizer, is_base_model):
    """Wrap the raw prompt with chat template if applicable."""
    if is_base_model:
        return raw_prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Model loading ────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_name, model_paths):
    path = model_paths[checkpoint_name]
    print(f"Loading tokenizer from {path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    print(f"Loading model from {path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = True
    model.to("cuda").eval()

    print(f"Model loaded: {model.config.architectures}, dtype={model.dtype}",
          flush=True)
    return model, tokenizer


# ── Generation ───────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, formatted_prompt, max_new_tokens=500,
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
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def parse_ava_line(raw_response):
    """Extract first meaningful line from Ava's raw generation."""
    first_line = raw_response.split("\n")[0].strip()

    # Strip echoed turn prefix if the model repeats it
    for prefix in [
        "Turn 1: Ava Thompson said:",
        "Ava Thompson said:",
        "Ava:",
    ]:
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix):].strip()

    return first_line


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_dialog_csv(path, intro, mia_opening, ava_response):
    dialog = [
        f"Turn 0: Mia Sanders said:{mia_opening}",
        f"Turn 1: Ava Thompson said:{ava_response}",
    ]
    dialog_str = repr(dialog)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Intro", "Dialog", "Intended Dialog"])
        writer.writerow([intro, dialog_str, dialog_str])


def save_prompt_record(path, formatted_prompt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["save_states"])
        writer.writerow([formatted_prompt])


def episode_exists(path):
    if not os.path.exists(path):
        return False
    try:
        return os.path.getsize(path) > 10
    except OSError:
        return False


# ── Dry run ──────────────────────────────────────────────────────────────────

def run_dry(model, tokenizer, envs, combos, scenario_vocab, is_base_model,
            temperature, seed):
    """Run one episode per condition (24 total), print prompt + response."""
    seen_conditions = set()
    count = 0

    for combo in combos:
        env_id = combo["env_id"]
        env = envs[env_id]
        condition = env["_meta"]["condition"]
        if condition in seen_conditions:
            continue
        seen_conditions.add(condition)
        count += 1

        mia_opening = get_mia_opening(env, scenario_vocab)
        raw_prompt = build_raw_prompt(env, mia_opening)
        formatted = format_prompt(raw_prompt, tokenizer, is_base_model)

        print(f"\n{'='*80}")
        print(f"[{count}/24] Condition: {condition}")
        print(f"  true_location: {env['_meta']['true_location']}")
        print(f"  a_gold: {env['_meta']['a_gold']}")
        print(f"  Mia opening: {mia_opening}")
        print(f"{'─'*80}")
        print(f"FORMATTED PROMPT:\n{formatted}")
        print(f"{'─'*80}")

        raw_response = generate_response(
            model, tokenizer, formatted, temperature=temperature, seed=seed,
        )
        ava_line = parse_ava_line(raw_response)

        print(f"RAW RESPONSE:\n{raw_response}")
        print(f"{'─'*80}")
        print(f"PARSED AVA LINE: {ava_line}")

        if len(seen_conditions) >= 24:
            break

    print(f"\n{'='*80}")
    print(f"Dry run complete: {count} episodes across {len(seen_conditions)} conditions.")


# ── Main loop ────────────────────────────────────────────────────────────────

def run_full(model, tokenizer, checkpoint_name, envs, combos, scenario_vocab,
             is_base_model, temperature, seed, max_episodes=None):
    results_dir = REPO_ROOT / RESULTS_DIR_NAME
    pair_name = f"{checkpoint_name}_None_0_{checkpoint_name}_None_0_false_belief_v3"
    dialog_dir = results_dir / "dialogs" / pair_name
    prompt_dir = dialog_dir / "prompt_records"
    os.makedirs(prompt_dir, exist_ok=True)

    total = len(combos)
    if max_episodes is not None:
        total = min(total, max_episodes)

    print(f"Running {total} episodes, saving to {dialog_dir}", flush=True)

    t0 = time.time()
    skipped = 0
    completed = 0

    for ep_idx in range(total):
        combo = combos[ep_idx]
        env_id = combo["env_id"]
        env = envs[env_id]

        csv_name = f"{ep_idx}_temp{temperature}_seed{seed}.csv"
        dialog_path = dialog_dir / csv_name
        prompt_path = prompt_dir / csv_name

        if episode_exists(str(dialog_path)):
            skipped += 1
            continue

        mia_opening = get_mia_opening(env, scenario_vocab)
        raw_prompt = build_raw_prompt(env, mia_opening)
        formatted = format_prompt(raw_prompt, tokenizer, is_base_model)

        raw_response = generate_response(
            model, tokenizer, formatted, temperature=temperature, seed=seed,
        )
        ava_line = parse_ava_line(raw_response)

        intro = build_intro(env)
        save_dialog_csv(str(dialog_path), intro, mia_opening, ava_line)
        save_prompt_record(str(prompt_path), formatted)

        completed += 1
        if completed % 100 == 0:
            elapsed = time.time() - t0
            rate = completed / elapsed
            remaining = (total - ep_idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{ep_idx+1}/{total}] completed={completed} skipped={skipped} "
                f"rate={rate:.1f} ep/s  ETA={remaining/60:.1f} min",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\nDone: {completed} completed, {skipped} skipped in {elapsed:.1f}s",
        flush=True,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo-2 checkpoints on false-belief v3 scenarios."
    )
    parser.add_argument(
        "--checkpoint_name", required=True, choices=CHECKPOINT_NAMES,
        help="Which OLMo-2 checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run 24 episodes (one per condition), print prompts + responses.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Cap total episodes (for testing).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_paths = load_model_paths()
    envs = load_envs()
    combos = load_combos()
    scenario_vocab = load_scenario_vocab()

    is_base_model = args.checkpoint_name == BASE_CHECKPOINT

    model, tokenizer = load_checkpoint(args.checkpoint_name, model_paths)

    if args.dry_run:
        run_dry(
            model, tokenizer, envs, combos, scenario_vocab, is_base_model,
            args.temperature, args.seed,
        )
    else:
        run_full(
            model, tokenizer, args.checkpoint_name, envs, combos,
            scenario_vocab, is_base_model, args.temperature, args.seed,
            args.max_episodes,
        )


if __name__ == "__main__":
    main()
