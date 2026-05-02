#!/usr/bin/env python3
"""
Logit-lens analysis for Mistral-7B dialog agents in false-belief scenarios.

Adapted from logit_lens_mistral.py for the 600-episode false-belief task
(100 scenario types x 3 belief conditions x 2 goal conditions).

For each model pair and a chosen episode, this script:
  1. Loads both agent models.
  2. Reads the saved per-turn prompts from the prompt_records CSV.
  3. Loads scenario metadata (item, locations, belief condition, goal) from
     envs_false_belief.json to parameterize probes.
  4. For continuation stems about location belief, intent, deception, and
     decision, applies the layerwise logit lens and saves heatmaps.
  5. For contrastive completion classes (two locations, honest/deceptive, etc.),
     computes average conditional token log-prob and saves plots.

Usage:
    python logit_lens_false_belief.py \\
        --model_1 Mistral-7B-Instruct-v0.3 \\
        --model_2 Mistral-7B-Instruct-v0.2 \\
        --episode 5 \\
        --seed 0 \\
        --temp 0.7
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.prompt_formatting import format_sotopia_prompt
from core.model_loader import load_model, resolve_model_path, _detect_arch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
SOTOPIA_DATA_DIR = os.path.join(REPO_ROOT, "sotopia_utils", "sotopia_data")


def resolve_hf_cache_dir(cli_cache: str | None) -> str | None:
    if cli_cache:
        return cli_cache
    for key in ("LM_HF_CACHE", "HF_HUB_CACHE"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def load_any_model(short_name: str, device: str = "cuda", cache_dir: str | None = None):
    """Load any supported model. Returns (tokenizer, model, arch_tag)."""
    tok, model = load_model(short_name, device=device, cache_dir=cache_dir)
    path = resolve_model_path(short_name)
    arch = _detect_arch(path)
    return tok, model, arch


# ────────────────────────────────────────────────────────────
# RMSNorm + unembedding  (same as logit_lens_mistral.py)
# ────────────────────────────────────────────────────────────

def get_unembedding_params(model, arch: str = "standard") -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    """Return (W_U, rms_weight_or_None, eps).

    For models using RMSNorm (Mistral, Llama, Gemma, Qwen), rms_weight is the
    learnable scale vector. For OLMo (unweighted LayerNorm), rms_weight is None.
    """
    if hasattr(model, "lm_head") and model.lm_head is not None:
        lm_head = model.lm_head
        if hasattr(lm_head, "weight"):
            W_U = lm_head.weight.detach().float()
        else:
            W_U = model.get_output_embeddings().weight.detach().float()
    else:
        W_U = model.get_output_embeddings().weight.detach().float()

    norm = getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        raise ValueError(
            f"Cannot locate final norm for {type(model).__name__}; extend get_unembedding_params."
        )

    norm_weight = getattr(norm, "weight", None)
    if norm_weight is not None:
        rms_weight: Optional[torch.Tensor] = norm_weight.detach().float()
    else:
        rms_weight = None

    eps = float(
        getattr(
            model.config,
            "rms_norm_eps",
            getattr(model.config, "layer_norm_eps", 1e-5),
        )
    )
    return W_U, rms_weight, eps


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def layer_norm_no_affine(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.to(torch.float32).mean(dim=-1, keepdim=True)
    variance = (x.to(torch.float32) - mean).pow(2).mean(dim=-1, keepdim=True)
    return ((x - mean) * torch.rsqrt(variance + eps)).to(x.dtype)


def logit_lens_decode(
    hidden: torch.Tensor,
    W_U: torch.Tensor,
    rms_weight: Optional[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h = hidden.float()
    if rms_weight is not None:
        h_normed = rms_norm(h, rms_weight, eps)
    else:
        h_normed = layer_norm_no_affine(h, eps)
    logits = h_normed @ W_U.T
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


# ────────────────────────────────────────────────────────────
# Hidden-state extraction  (same as logit_lens_mistral.py)
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def get_hidden_states(prompt: str, tokenizer, model, device: str, model_short: str):
    text = format_sotopia_prompt(model_short, prompt, tokenizer)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    outputs = model.model(input_ids, output_hidden_states=True)
    hidden_states = tuple(h.detach().float().cpu() for h in outputs.hidden_states)
    return hidden_states, tokens, input_ids.detach().cpu()


# ────────────────────────────────────────────────────────────
# Scenario metadata
# ────────────────────────────────────────────────────────────

_SCENARIO_VOCAB_PATH = os.path.join(SOTOPIA_DATA_DIR, "scenario_vocab_false_belief.json")
with open(_SCENARIO_VOCAB_PATH, encoding="utf-8") as _f:
    SCENARIO_VOCAB = json.load(_f)
for _k, _v in SCENARIO_VOCAB.items():
    _v["locations"] = tuple(_v["locations"])


def load_scenario_meta(episode: int,
                       envs_basename: str = "envs_false_belief.json",
                       combos_basename: str = "env_agent_combos_false_belief_fixed_two_agents.json",
                       ) -> Dict[str, Any]:
    """Load the full scenario metadata for an episode index."""
    combos_path = os.path.join(SOTOPIA_DATA_DIR, combos_basename)
    envs_path = os.path.join(SOTOPIA_DATA_DIR, envs_basename)
    with open(combos_path) as f:
        combos = json.load(f)
    with open(envs_path) as f:
        envs = json.load(f)
    combo = combos[episode]
    env = envs[combo["env_id"]]
    meta = env["_meta"]
    vocab = SCENARIO_VOCAB[meta["scenario_type"]]
    return {
        "env_id": combo["env_id"],
        "codename": env["codename"],
        "scenario_text": env["scenario"],
        "agent_goals": env["agent_goals"],
        **meta,
        "item": vocab["item"],
        "loc_a": vocab["locations"][0],
        "loc_b": vocab["locations"][1],
        "prep": vocab["prep"],
    }


def _loc_display(meta: Dict[str, Any], raw_loc: str) -> str:
    """Convert stored location id (e.g. 'red_box') to display form ('red box')."""
    return raw_loc.replace("_", " ")


# ────────────────────────────────────────────────────────────
# Prompt / data helpers
# ────────────────────────────────────────────────────────────

def load_prompt_records(pair_name: str, episode: int, seed: int, temp: float,
                        results_dir: str = "sotopia_results_false_belief_100") -> List[str]:
    path = os.path.join(
        REPO_ROOT, results_dir, "dialogs", pair_name,
        "prompt_records", f"{episode}_temp{temp}_seed{seed}.csv",
    )
    prompts: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            prompts.append(row[0])
    return prompts


def load_dialog_record(pair_name: str, episode: int, seed: int, temp: float,
                       results_dir: str = "sotopia_results_false_belief_100") -> Dict[str, str]:
    path = os.path.join(
        REPO_ROOT, results_dir, "dialogs", pair_name,
        f"{episode}_temp{temp}_seed{seed}.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No dialog rows found in {path}")
    return rows[0]


def infer_agent_names(dialog_record: Dict[str, str]) -> Tuple[str, str]:
    intro = dialog_record.get("Intro", "")
    m = re.search(r"Participants:\s*(.+?)\s+and\s+(.+?)(?:\n|$)", intro)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Mia Sanders", "Ava Thompson"


def short_name(full_name: str) -> str:
    return full_name.split()[0].strip()


def parse_dialog_turns(dialog_record: Dict[str, str]) -> List[Dict[str, Any]]:
    dialog_raw = dialog_record.get("Dialog", "")
    turns: List[Dict[str, Any]] = []
    try:
        chunks = ast.literal_eval(dialog_raw)
    except Exception:
        chunks = []
    for chunk in chunks:
        m = re.match(r"Turn\s+(\d+):\s+(.+?)\s+said:(.*)", chunk, re.DOTALL)
        if not m:
            continue
        turns.append({
            "turn": int(m.group(1)),
            "speaker": m.group(2).strip(),
            "text": m.group(3).strip(),
        })
    return turns


def turn_to_agent(turn_idx: int) -> int:
    return 1 if turn_idx % 2 == 0 else 2


# ────────────────────────────────────────────────────────────
# Probe builders  (false-belief specific)
# ────────────────────────────────────────────────────────────

def build_continuation_stems_fb(
    agent1_name: str,
    agent2_name: str,
    meta: Dict[str, Any],
) -> List[str]:
    a1 = short_name(agent1_name)
    a2 = short_name(agent2_name)
    item = meta["item"]
    prep = meta["prep"]

    return [
        # Location belief
        f"The {item} is {prep}",
        f"I think the {item} is {prep}",
        f"I believe the {item} is {prep}",
        f"The other person thinks the {item} is {prep}",
        f"{a2} thinks the {item} is {prep}",
        f"{a1} thinks the {item} is {prep}",
        # Goal / intent
        f"I should tell them the {item} is {prep}",
        "I should",
        f"I want them to check the",
        "I am trying to",
        # Deception / honesty
        "I am being",
        "The other person is being",
        f"{a2} is being",
        f"{a1} is being",
        "I should tell the truth about",
        "The other person is trying to",
        f"{a2} is trying to",
        f"{a1} is trying to",
        # Decision / outcome
        "I will check the",
        f"The correct location is the",
        # Open-ended
        "I know that",
        "The truth is",
    ]


def build_contrastive_classes_fb(
    agent1_name: str,
    agent2_name: str,
    meta: Dict[str, Any],
) -> Dict[str, List[str]]:
    a1 = short_name(agent1_name)
    a2 = short_name(agent2_name)
    item = meta["item"]
    prep = meta["prep"]
    loc_a = meta["loc_a"]
    loc_b = meta["loc_b"]

    return {
        "self_belief_location": [
            f"The {item} is {prep} {loc_a}",
            f"The {item} is {prep} {loc_b}",
        ],
        "self_think_location": [
            f"I think the {item} is {prep} {loc_a}",
            f"I think the {item} is {prep} {loc_b}",
        ],
        "self_believe_location": [
            f"I believe the {item} is {prep} {loc_a}",
            f"I believe the {item} is {prep} {loc_b}",
        ],
        "other_think_location": [
            f"The other person thinks the {item} is {prep} {loc_a}",
            f"The other person thinks the {item} is {prep} {loc_b}",
        ],
        f"{a2.lower()}_think_location": [
            f"{a2} thinks the {item} is {prep} {loc_a}",
            f"{a2} thinks the {item} is {prep} {loc_b}",
        ],
        f"{a1.lower()}_think_location": [
            f"{a1} thinks the {item} is {prep} {loc_a}",
            f"{a1} thinks the {item} is {prep} {loc_b}",
        ],
        "tell_location": [
            f"I should tell them the {item} is {prep} {loc_a}",
            f"I should tell them the {item} is {prep} {loc_b}",
        ],
        "want_check": [
            f"I want them to check the {loc_a}",
            f"I want them to check the {loc_b}",
        ],
        "will_check": [
            f"I will check the {loc_a}",
            f"I will check the {loc_b}",
        ],
        "self_honesty": [
            "I am being honest",
            "I am being deceptive",
        ],
        "other_honesty": [
            "The other person is being honest",
            "The other person is being deceptive",
        ],
        "other_strategy": [
            "The other person is trying to help",
            "The other person is trying to mislead",
        ],
        f"{a2.lower()}_strategy": [
            f"{a2} is trying to help",
            f"{a2} is trying to mislead",
        ],
        f"{a1.lower()}_strategy": [
            f"{a1} is trying to help",
            f"{a1} is trying to mislead",
        ],
        "decision_confidence": [
            "I am sure about my choice",
            "I am unsure about my choice",
        ],
    }


# ────────────────────────────────────────────────────────────
# Behavioral summary  (false-belief specific)
# ────────────────────────────────────────────────────────────

def _summarize_turn_fb(turn_text: str, meta: Dict[str, Any]) -> List[str]:
    text = turn_text.lower()
    tags: List[str] = []
    loc_a = meta["loc_a"].lower()
    loc_b = meta["loc_b"].lower()
    item = meta["item"].lower()
    if loc_a in text:
        tags.append(f"mentioned {meta['loc_a']}")
    if loc_b in text:
        tags.append(f"mentioned {meta['loc_b']}")
    if item in text:
        tags.append(f"mentioned {meta['item']}")
    if "leave" in text:
        tags.append("said LEAVE")
    if any(k in text for k in ["i think", "i believe", "i'll check", "i will check"]):
        tags.append("stated belief or decision")
    return tags


def _extract_final_choice(turns: List[Dict[str, Any]], meta: Dict[str, Any]) -> Optional[str]:
    """Extract the seeker's choice from the last turn, only if it is
    Ava's turn and contains LEAVE."""
    if not turns:
        return None
    last = turns[-1]
    if last["speaker"] != "Ava Thompson":
        return None
    text = last["text"].lower()
    if "leave" not in text:
        return None
    loc_a = meta["loc_a"].lower()
    loc_b = meta["loc_b"].lower()
    has_a = loc_a in text
    has_b = loc_b in text
    if has_a and not has_b:
        return meta["loc_a"]
    if has_b and not has_a:
        return meta["loc_b"]
    if has_a and has_b:
        leave_pos = text.index("leave")
        last_a = text.rfind(loc_a, 0, leave_pos)
        last_b = text.rfind(loc_b, 0, leave_pos)
        if last_a > last_b:
            return meta["loc_a"]
        elif last_b > last_a:
            return meta["loc_b"]
    return None


def build_behavioral_summary_fb(
    dialog_record: Dict[str, str],
    agent1_name: str,
    agent2_name: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    turns = parse_dialog_turns(dialog_record)
    final_choice = _extract_final_choice(turns, meta)
    true_loc_display = _loc_display(meta, meta["true_location"])
    correct = (final_choice is not None and
               final_choice.lower().replace(" ", "_") == meta["true_location"]
               or (final_choice is not None and final_choice.lower() == true_loc_display.lower()))

    turn_summaries = []
    for turn in turns:
        turn_summaries.append({
            "turn": turn["turn"],
            "speaker": turn["speaker"],
            "text": turn["text"],
            "tags": _summarize_turn_fb(turn["text"], meta),
        })

    return {
        "participants": [agent1_name, agent2_name],
        "scenario_meta": {
            "scenario_id": meta["scenario_id"],
            "scenario_type": meta["scenario_type"],
            "item": meta["item"],
            "true_location": meta["true_location"],
            "belief_condition": meta["belief_condition"],
            "goal_condition": meta["goal_condition"],
            "b_belief": meta["b_belief"],
            "mismatch": meta["mismatch"],
            "loc_a": meta["loc_a"],
            "loc_b": meta["loc_b"],
        },
        "final_choice": final_choice,
        "correct": correct,
        "n_turns": len(turns),
        "turns": turn_summaries,
    }


# ────────────────────────────────────────────────────────────
# Stem heatmap  (same as logit_lens_mistral.py)
# ────────────────────────────────────────────────────────────

def stem_heatmap_for_turn(
    prompt: str,
    stem: str,
    tokenizer,
    model,
    W_U: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    device: str,
    model_short: str,
    compute_matrix: bool = True,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    prompt_text = format_sotopia_prompt(model_short, prompt, tokenizer)
    full_text = prompt_text + stem

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
    full_inputs = tokenizer(full_text, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_input_ids = full_inputs["input_ids"].to(device)

    stem_ids = full_input_ids[0, prompt_len:].detach().cpu().tolist()
    stem_tokens = tokenizer.convert_ids_to_tokens(stem_ids)

    outputs = model.model(full_input_ids, output_hidden_states=True)
    all_hidden = outputs.hidden_states
    n_total = len(all_hidden)

    compute_layers = layer_indices if layer_indices is not None else list(range(n_total))
    n_out = len(compute_layers)
    n_stem = len(stem_ids)

    cont_prob_vec = np.zeros(n_out, dtype=np.float32)
    cont_pred_id_vec = np.zeros(n_out, dtype=np.int64)

    if compute_matrix:
        prob_matrix = np.zeros((n_out, n_stem), dtype=np.float32)
        pred_id_matrix = np.zeros((n_out, n_stem), dtype=np.int64)
        target_positions = list(range(prompt_len, full_input_ids.shape[1]))
    else:
        prob_matrix = np.zeros((0, 0), dtype=np.float32)
        pred_id_matrix = np.zeros((0, 0), dtype=np.int64)

    for out_idx, li in enumerate(compute_layers):
        hs = all_hidden[li].detach().float().cpu()
        if compute_matrix:
            for col, pos in enumerate(target_positions):
                h = hs[0, pos - 1, :]
                logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
                probs = torch.softmax(logits[0], dim=-1)
                top_prob, top_id = torch.max(probs, dim=-1)
                prob_matrix[out_idx, col] = top_prob.item()
                pred_id_matrix[out_idx, col] = top_id.item()

        h = hs[0, full_input_ids.shape[1] - 1, :]
        logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
        probs = torch.softmax(logits[0], dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        cont_prob_vec[out_idx] = top_prob.item()
        cont_pred_id_vec[out_idx] = top_id.item()

    all_labels = ["Embed"] + [str(i) for i in range(1, n_total)]
    layer_labels = [all_labels[i] for i in compute_layers]
    return prob_matrix, pred_id_matrix, cont_prob_vec, cont_pred_id_vec, stem_tokens, layer_labels


def _clean_token(tok: str, max_len: int = 10) -> str:
    tok = tok.replace("▁", "·").replace("<0x0A>", "\\n")
    if len(tok) > max_len:
        tok = tok[: max_len - 1] + "…"
    return tok


def plot_stem_heatmap(
    prob_matrix: np.ndarray,
    pred_id_matrix: np.ndarray,
    stem_tokens: List[str],
    layer_labels: List[str],
    tokenizer,
    title: str,
    save_path: str,
) -> None:
    n_layers, n_tokens = prob_matrix.shape
    fig_w = max(7, 1.4 * n_tokens)
    fig_h = max(6, 0.45 * n_layers)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(prob_matrix, aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="Top-1 next-token probability")

    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels([_clean_token(t) for t in stem_tokens], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=7)
    ax.set_xlabel("Sequential stem token position")
    ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=9)

    vmax = float(prob_matrix.max()) if prob_matrix.size else 1.0
    for i in range(n_layers):
        for j in range(n_tokens):
            pred_tok = tokenizer.convert_ids_to_tokens([int(pred_id_matrix[i, j])])[0]
            pred_tok = _clean_token(pred_tok, max_len=8)
            color = "white" if prob_matrix[i, j] < 0.5 * vmax else "black"
            ax.text(j, i, pred_tok, ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Contrastive scoring  (same as logit_lens_mistral.py)
# ────────────────────────────────────────────────────────────

def score_contrastive_class(
    prompt: str,
    model,
    tokenizer,
    completions: List[str],
    W_U: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    device: str,
    model_short: str,
    layer_indices: Optional[List[int]] = None,
) -> np.ndarray:
    all_scores = []
    for completion in completions:
        full_text = format_sotopia_prompt(model_short, prompt, tokenizer) + completion
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(device)

        prompt_text = format_sotopia_prompt(model_short, prompt, tokenizer)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        outputs = model.model(full_input_ids, output_hidden_states=True)
        all_hidden = outputs.hidden_states
        n_total = len(all_hidden)

        compute_layers = layer_indices if layer_indices is not None else list(range(n_total))
        n_out = len(compute_layers)
        layer_scores = np.zeros(n_out, dtype=np.float32)

        target_positions = list(range(prompt_len, full_input_ids.shape[1]))
        if len(target_positions) == 0:
            layer_scores[:] = -np.inf
            all_scores.append(layer_scores)
            continue

        input_ids_cpu = full_input_ids.detach().cpu()
        for out_idx, li in enumerate(compute_layers):
            hs = all_hidden[li].detach().float().cpu()
            total = 0.0
            count = 0
            for pos in target_positions:
                h = hs[0, pos - 1, :]
                logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                target_id = input_ids_cpu[0, pos].item()
                total += log_probs[target_id].item()
                count += 1
            layer_scores[out_idx] = total / count

        all_scores.append(layer_scores)

    return np.stack(all_scores, axis=1)


def plot_contrastive_class(
    scores: np.ndarray,
    completions: List[str],
    class_name: str,
    title: str,
    save_path: str,
) -> None:
    n_layers = scores.shape[0]
    xs = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, comp in enumerate(completions):
        ax.plot(xs, scores[:, ci], marker="o", markersize=3, label=comp)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Emb"] + [str(i) for i in range(1, n_layers)], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg cond. token log-prob")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contrastive_across_turns(
    all_scores: Dict[int, np.ndarray],
    completions: List[str],
    class_name: str,
    layer: int,
    title: str,
    save_path: str,
) -> None:
    turns = sorted(all_scores.keys())
    n_comp = len(completions)
    x = np.arange(len(turns))
    width = 0.8 / n_comp

    fig, ax = plt.subplots(figsize=(max(8, len(turns) * 1.5), 5))
    for ci, comp in enumerate(completions):
        vals = [all_scores[t][layer, ci] for t in turns]
        ax.bar(x + ci * width, vals, width, label=comp)
    ax.set_xticks(x + width * (n_comp - 1) / 2)
    ax.set_xticklabels([f"T{t}" for t in turns])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Avg cond. token log-prob")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _agent_label(agent: int, model_short: str) -> str:
    return f"Agent{agent} ({model_short})"


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def compact_layer_indices(num_hidden_layers: int) -> List[int]:
    """Return hidden_states indices for the 30%, 70%, and last layers."""
    return sorted(set([
        round(num_hidden_layers * 0.3),
        round(num_hidden_layers * 0.7),
        num_hidden_layers,
    ]))


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit-lens analysis for Mistral dialog agents in false-belief scenarios."
    )
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index (0-599), maps to a specific false-belief scenario.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf-cache", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="sotopia_results_false_belief_100",
                        help="Results root directory name (under REPO_ROOT).")
    parser.add_argument("--summary_layers", type=int, nargs="+", default=[0, 8, 16, 24, 31])
    parser.add_argument("--verbose_analysis", action="store_true", default=False,
                        help="When set, compute and save all outputs (heatmaps, contrastive plots, "
                             "per-turn .npz files, cross-turn summaries). Default: only save the "
                             "three episode-level JSON summaries.")
    args = parser.parse_args()

    hf_cache = resolve_hf_cache_dir(args.hf_cache)
    if hf_cache:
        os.makedirs(hf_cache, exist_ok=True)
        print(f"Hugging Face cache: {hf_cache}")

    # ── scenario metadata ──
    meta = load_scenario_meta(args.episode)
    print(f"Episode {args.episode}: {meta['codename']}")
    print(f"  scenario_type={meta['scenario_type']}, item={meta['item']}")
    print(f"  true_location={meta['true_location']}, belief={meta['belief_condition']}, goal={meta['goal_condition']}")
    print(f"  loc_a={meta['loc_a']}, loc_b={meta['loc_b']}, b_belief={meta['b_belief']}")

    # ── pair name ──
    affected_type = "None"
    value = 0
    m1_full = f"{args.model_1}_{affected_type}_{value}"
    m2_full = f"{args.model_2}_{affected_type}_{value}"
    pair_name = f"{m1_full}_{m2_full}_false_belief_fixed_two_agents"

    dialog_record = load_dialog_record(pair_name, args.episode, args.seed, args.temp,
                                       results_dir=args.results_dir)
    agent1_name, agent2_name = infer_agent_names(dialog_record)

    continuation_stems = build_continuation_stems_fb(agent1_name, agent2_name, meta)
    contrastive_classes = build_contrastive_classes_fb(agent1_name, agent2_name, meta)

    print(f"Pair : {pair_name}")
    print(f"Agent 1 (A): {agent1_name}  |  Agent 2 (B): {agent2_name}")
    print(f"  {len(continuation_stems)} continuation stems, {len(contrastive_classes)} contrastive classes")

    # ── output dir ──
    out_dir = os.path.join(
        REPO_ROOT, "logit_lens_results_false_belief_100", pair_name,
        f"episode_{args.episode}_{meta['codename']}",
    )
    os.makedirs(out_dir, exist_ok=True)
    verbose = args.verbose_analysis

    if verbose:
        stem_dir = os.path.join(out_dir, "stem_heatmaps")
        os.makedirs(stem_dir, exist_ok=True)
        contrastive_dir = os.path.join(out_dir, "contrastive")
        os.makedirs(contrastive_dir, exist_ok=True)
        summary_dir = os.path.join(out_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

    # ── skip if already completed ──
    if verbose:
        contrastive_summary_path = os.path.join(out_dir, "contrastive_summary.json")
        stem_summary_path = os.path.join(out_dir, "stem_summary.json")
    else:
        contrastive_summary_path = os.path.join(out_dir, "contrastive_summary_compact.json")
        stem_summary_path = os.path.join(out_dir, "stem_summary_compact.json")
    if os.path.exists(contrastive_summary_path) and os.path.exists(stem_summary_path):
        print(f"  Already completed — skipping (found {os.path.basename(contrastive_summary_path)} & {os.path.basename(stem_summary_path)})")
        return

    # ── load both models ──
    print(f"Loading agent-1 model ({args.model_1}) ...")
    tok1, mdl1, arch1 = load_any_model(args.model_1, args.device, cache_dir=hf_cache)
    W_U1, rms_w1, eps1 = get_unembedding_params(mdl1, arch1)
    W_U1, rms_w1 = W_U1.cpu(), rms_w1.cpu()

    if args.model_1 == args.model_2:
        print(f"Agent-2 uses the same model — reusing loaded weights.")
        tok2, mdl2, arch2, W_U2, rms_w2, eps2 = tok1, mdl1, arch1, W_U1, rms_w1, eps1
    else:
        print(f"Loading agent-2 model ({args.model_2}) ...")
        tok2, mdl2, arch2 = load_any_model(args.model_2, args.device, cache_dir=hf_cache)
        W_U2, rms_w2, eps2 = get_unembedding_params(mdl2, arch2)
        W_U2, rms_w2 = W_U2.cpu(), rms_w2.cpu()

    agent_resources = {
        1: (tok1, mdl1, W_U1, rms_w1, eps1, args.model_1),
        2: (tok2, mdl2, W_U2, rms_w2, eps2, args.model_2),
    }

    if not verbose:
        n_hl_1 = int(getattr(mdl1.config, "num_hidden_layers", 31))
        compact_layers_1 = compact_layer_indices(n_hl_1)
        if args.model_1 == args.model_2:
            compact_layers_2 = compact_layers_1
        else:
            n_hl_2 = int(getattr(mdl2.config, "num_hidden_layers", 31))
            compact_layers_2 = compact_layer_indices(n_hl_2)
        compact_layers_for_agent = {1: compact_layers_1, 2: compact_layers_2}
        print(f"  Compact layers: agent1={compact_layers_1}, agent2={compact_layers_2}")
    else:
        compact_layers_for_agent = {1: None, 2: None}

    # ── load prompts ──
    prompts = load_prompt_records(pair_name, args.episode, args.seed, args.temp,
                                  results_dir=args.results_dir)
    n_turns = len(prompts)
    print(f"  {n_turns} dialog turns")

    # ── per-turn analysis ──
    contrastive_all: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {
        1: {cls: {} for cls in contrastive_classes},
        2: {cls: {} for cls in contrastive_classes},
    }
    stem_summary: Dict[str, Any] = {
        "pair": pair_name,
        "episode": args.episode,
        "scenario": meta["codename"],
        "model_1": args.model_1,
        "model_2": args.model_2,
        "participants": {"agent_1": agent1_name, "agent_2": agent2_name},
        "scenario_meta": meta,
        "turns": [],
    }

    for turn_idx in range(n_turns):
        agent = turn_to_agent(turn_idx)
        tok, mdl, W_U, rms_w, eps, model_short = agent_resources[agent]
        label = _agent_label(agent, model_short)
        prompt = prompts[turn_idx]
        speaker_name = agent1_name if agent == 1 else agent2_name
        print(f"\n─── Turn {turn_idx}  ({label}, {speaker_name}) ───")

        if verbose:
            hidden_states, tokens, _ = get_hidden_states(prompt, tok, mdl, args.device, model_short)
            print(f"  Prompt tokens: {len(tokens)}")

        # ── Stem heatmaps ──
        if verbose:
            turn_stem_dir = os.path.join(stem_dir, f"turn_{turn_idx}_agent{agent}")
            os.makedirs(turn_stem_dir, exist_ok=True)
        turn_stem_summary: Dict[str, Any] = {
            "turn": turn_idx,
            "agent_index": agent,
            "agent_identity": speaker_name,
            "model": model_short,
            "stems": {},
        }

        _layer_indices = compact_layers_for_agent[agent]
        for si, stem in enumerate(continuation_stems):
            (
                prob_mat, pred_id_mat, cont_prob_vec, cont_pred_id_vec,
                stem_toks, layer_labels,
            ) = stem_heatmap_for_turn(prompt, stem, tok, mdl, W_U, rms_w, eps, args.device, model_short,
                                      compute_matrix=verbose, layer_indices=_layer_indices)

            if verbose:
                safe_stem = stem.replace(" ", "_").replace("/", "_")[:40]
                save_path = os.path.join(turn_stem_dir, f"{si:02d}_{safe_stem}.png")
                plot_stem_heatmap(
                    prob_mat, pred_id_mat, stem_toks, layer_labels, tok,
                    title=f"{label} | Turn {turn_idx} | \"{stem}\"",
                    save_path=save_path,
                )
            per_layer = []
            for out_idx, layer_name in enumerate(layer_labels):
                actual_layer = _layer_indices[out_idx] if _layer_indices is not None else out_idx
                pred_token = tok.convert_ids_to_tokens([int(cont_pred_id_vec[out_idx])])[0]
                per_layer.append({
                    "layer_index": actual_layer,
                    "layer_name": layer_name,
                    "predicted_token": pred_token,
                    "predicted_token_clean": _clean_token(pred_token, max_len=30),
                    "probability": float(cont_prob_vec[out_idx]),
                })
            turn_stem_summary["stems"][stem] = {
                "stem_tokens": stem_toks,
                "continuation_prediction_after_full_stem": per_layer,
            }
        if verbose:
            save_json(os.path.join(turn_stem_dir, "stem_summary.json"), turn_stem_summary)
        stem_summary["turns"].append(turn_stem_summary)

        # ── Contrastive completions ──
        if verbose:
            turn_contr_dir = os.path.join(contrastive_dir, f"turn_{turn_idx}_agent{agent}")
            os.makedirs(turn_contr_dir, exist_ok=True)

        for cls_name, completions in contrastive_classes.items():
            scores = score_contrastive_class(
                prompt, mdl, tok, completions, W_U, rms_w, eps, args.device, model_short,
                layer_indices=_layer_indices,
            )
            contrastive_all[agent][cls_name][turn_idx] = scores
            if verbose:
                save_path = os.path.join(turn_contr_dir, f"{cls_name}.png")
                plot_contrastive_class(
                    scores, completions, cls_name,
                    title=f"{label} | Turn {turn_idx} | {cls_name}",
                    save_path=save_path,
                )

        if verbose:
            np.savez_compressed(
                os.path.join(turn_contr_dir, "contrastive_scores.npz"),
                **{cls: contrastive_all[agent][cls][turn_idx] for cls in contrastive_classes},
            )

        print(f"  Computed {len(continuation_stems)} stems, "
              f"{len(contrastive_classes)} contrastive classes"
              + (" (saved plots)" if verbose else ""))

    # ── Summary across turns (per agent) ──
    if verbose:
        print("\n─── Cross-turn summaries ───")
        for agent in (1, 2):
            model_short = agent_resources[agent][5]
            mdl_sum = agent_resources[agent][1]
            max_layer_idx = int(getattr(mdl_sum.config, "num_hidden_layers", 31))
            label = _agent_label(agent, model_short)
            for layer in args.summary_layers:
                li = min(int(layer), max_layer_idx)
                layer_dir = os.path.join(summary_dir, f"agent{agent}", f"layer_{li}")
                os.makedirs(layer_dir, exist_ok=True)
                for cls_name, completions in contrastive_classes.items():
                    turn_scores = contrastive_all[agent][cls_name]
                    if not turn_scores:
                        continue
                    save_path = os.path.join(layer_dir, f"{cls_name}.png")
                    plot_contrastive_across_turns(
                        turn_scores, completions, cls_name, li,
                        title=f"{label} | Layer {li} | {cls_name}",
                        save_path=save_path,
                    )

    # ── Save all contrastive scores as JSON ──
    summary_json: Dict[str, Any] = {
        "pair": pair_name,
        "model_1": args.model_1,
        "model_2": args.model_2,
        "episode": args.episode,
        "scenario": meta["codename"],
        "n_turns": n_turns,
        "participants": {"agent_1": agent1_name, "agent_2": agent2_name},
        "scenario_meta": meta,
    }
    if not verbose:
        summary_json["compact_layers"] = {
            f"agent_{a}": compact_layers_for_agent[a] for a in (1, 2)
        }
        stem_summary["compact_layers"] = {
            f"agent_{a}": compact_layers_for_agent[a] for a in (1, 2)
        }
    for agent in (1, 2):
        agent_key = f"agent_{agent}"
        summary_json[agent_key] = {}
        for cls_name in contrastive_classes:
            summary_json[agent_key][cls_name] = {}
            for t, arr in contrastive_all[agent][cls_name].items():
                summary_json[agent_key][cls_name][str(t)] = arr.tolist()
    save_json(contrastive_summary_path, summary_json)
    save_json(stem_summary_path, stem_summary)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
