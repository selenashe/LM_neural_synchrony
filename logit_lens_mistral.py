#!/usr/bin/env python3
"""
Logit-lens analysis for Mistral-7B dialog agents in Sotopia.

For each model pair and a chosen episode, this script:
  1. Loads *both* agent models (may be the same checkpoint if a self-play
     pair).
  2. Reads the saved per-turn prompts from the prompt_records CSV.
  3. Iterates through *all* turns of the dialog in order.  At each turn the
     speaking agent's model is used to run a forward pass and extract hidden
     states at every layer / token position.
  4. For a set of continuation stems, applies the layerwise logit lens
     (RMSNorm → lm_head → softmax) and saves heatmaps.
  5. For contrastive completion classes, computes the average conditional
     token log-prob under the layerwise logit lens and saves bar-chart /
     line-plot comparisons.

Both agents are analysed jointly in a single run so that their interleaved
turns can be compared side-by-side.

Usage:
    python logit_lens_mistral.py \
        --model_1 Mistral-7B-Instruct-v0.3 \
        --model_2 Mistral-7B-Instruct-v0.2 \
        --episode 6 \
        --seed 0 \
        --temp 0.7

The script mirrors the generation settings from sample_normal_agent.py
(temperature, seed, max_new_tokens, max_turns) so prompts are reproducible.

Hugging Face Hub downloads can exceed AFS home quota: use --hf-cache, or set
LM_HF_CACHE / HF_HUB_CACHE, or point model_paths.json at a local snapshot.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# ────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────

MODEL_PATHS_JSON = os.path.join(REPO_ROOT, "model_paths.json")


def resolve_hf_cache_dir(cli_cache: str | None) -> str | None:
    """
    Where to store Hugging Face Hub downloads. Default home cache (~/.cache/huggingface)
    often hits AFS quota; set --hf-cache or LM_HF_CACHE / HF_HUB_CACHE to scratch.
    """
    if cli_cache:
        return cli_cache
    for key in ("LM_HF_CACHE", "HF_HUB_CACHE"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def _resolve_model_path(short_name: str) -> str:
    """Map a short model name to its HF hub id or local path."""
    if os.path.exists(MODEL_PATHS_JSON):
        with open(MODEL_PATHS_JSON) as f:
            paths = json.load(f)
        if short_name in paths:
            return paths[short_name]
    hub_map = {
        "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    if short_name in hub_map:
        return hub_map[short_name]
    raise ValueError(f"Cannot resolve model path for {short_name!r}")


def load_mistral(
    short_name: str,
    device: str = "cuda",
    cache_dir: str | None = None,
):
    """Load a Mistral model + tokenizer."""
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
    path = _resolve_model_path(short_name)
    kw: Dict[str, Any] = {"trust_remote_code": True}
    if cache_dir is not None:
        kw["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(path, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **kw,
    )
    model.to(device).eval()
    return tokenizer, model

# ────────────────────────────────────────────────────────────
# RMSNorm + unembedding (Mistral / Llama family)
# ────────────────────────────────────────────────────────────

def get_unembedding_params(model) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Extract lm_head weight and final RMSNorm weight + eps."""
    W_U = model.lm_head.weight.detach().float()          # (V, d)
    rms_weight = model.model.norm.weight.detach().float() # (d,)
    eps = model.config.rms_norm_eps
    return W_U, rms_weight, eps


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm (no bias) — same as MistralRMSNorm."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def logit_lens_decode(
    hidden: torch.Tensor, W_U: torch.Tensor, rms_weight: torch.Tensor, eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """hidden (..., d) → logits (..., V), probs (..., V)."""
    h = hidden.float()
    h_normed = rms_norm(h, rms_weight, eps)
    logits = h_normed @ W_U.T
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


# ────────────────────────────────────────────────────────────
# Extract hidden states for a prompt
# ────────────────────────────────────────────────────────────

def wrap_inst(prompt: str) -> str:
    """Apply Mistral [INST] template (same as LM_hf.py)."""
    return f"[INST]{prompt}[/INST]"


@torch.no_grad()
def get_hidden_states(
    prompt: str,
    tokenizer,
    model,
    device: str,
) -> Tuple[Tuple[torch.Tensor, ...], List[str], torch.Tensor]:
    """
    Forward pass returning all hidden states (embed + 32 layers).

    Returns
    -------
    hidden_states : tuple of (1, seq_len, d) tensors, length n_layers+1
    tokens        : list of token strings
    input_ids     : (1, seq_len)
    """
    text = wrap_inst(prompt)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    outputs = model.model(input_ids, output_hidden_states=True)
    hidden_states = tuple(h.detach().float().cpu() for h in outputs.hidden_states)
    return hidden_states, tokens, input_ids.detach().cpu()


# ────────────────────────────────────────────────────────────
# Prompt / data helpers
# ────────────────────────────────────────────────────────────

def load_prompt_records(pair_name: str, episode: int, seed: int, temp: float) -> List[str]:
    """Load per-turn prompts from the CSV written by sample_normal_agent.py."""
    path = os.path.join(
        REPO_ROOT, "sotopia_results", "dialogs", pair_name,
        "prompt_records", f"{episode}_temp{temp}_seed{seed}.csv",
    )
    prompts: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # noqa: F841
        for row in reader:
            prompts.append(row[0])
    return prompts


def load_dialog_record(pair_name: str, episode: int, seed: int, temp: float) -> Dict[str, str]:
    """Load the single-row dialog CSV written by sample_normal_agent.py."""
    path = os.path.join(
        REPO_ROOT,
        "sotopia_results",
        "dialogs",
        pair_name,
        f"{episode}_temp{temp}_seed{seed}.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No dialog rows found in {path}")
    return rows[0]


def infer_agent_names(dialog_record: Dict[str, str]) -> Tuple[str, str]:
    """Infer full participant names from the dialog intro."""
    intro = dialog_record.get("Intro", "")
    m = re.search(r"Participants:\s*(.+?)\s+and\s+(.+?)(?:\n|$)", intro)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Isabelle Martinez", "Donovan Reeves"


def short_name(full_name: str) -> str:
    return full_name.split()[0].strip()


def build_continuation_stems(agent1_name: str, agent2_name: str) -> List[str]:
    a1 = short_name(agent1_name)
    a2 = short_name(agent2_name)
    return [
        "I value the",
        "I care more about the",
        "The most important item for me is the",
        f"{a2} values the",
        f"{a1} values the",
        f"{a2} wants the",
        f"{a1} wants the",
        f"{a2} really wants the",
        f"{a1} really wants the",
        f"{a2} still wants the",
        f"{a1} still wants the",
        f"{a2} thinks I value the",
        f"{a1} thinks I value the",
        "I should",
        "I should take the",
        "I should give up the",
        "I should ask for the",
        "I should propose",
        "I should respond by",
        "The best deal is",
        "The final deal will be",
        "I will take the",
        "We are",
        "We are likely to",
        f"{a2} is being",
        f"{a1} is being",
        f"{a2} is trying to",
        f"{a1} is trying to",
    ]


def build_contrastive_classes(agent1_name: str, agent2_name: str) -> Dict[str, List[str]]:
    a1 = short_name(agent1_name)
    a2 = short_name(agent2_name)
    return {
        "self_value_item": [
            "I value the ball",
            "I value the hats",
            "I value the books",
        ],
        "self_want_item": [
            "I want the ball",
            "I want the hats",
            "I want the books",
        ],
        f"{a2.lower()}_value_item": [
            f"{a2} values the ball",
            f"{a2} values the hats",
            f"{a2} values the books",
        ],
        f"{a2.lower()}_want_item": [
            f"{a2} wants the ball",
            f"{a2} wants the hats",
            f"{a2} wants the books",
        ],
        f"{a1.lower()}_value_item": [
            f"{a1} values the ball",
            f"{a1} values the hats",
            f"{a1} values the books",
        ],
        f"{a1.lower()}_want_item": [
            f"{a1} wants the ball",
            f"{a1} wants the hats",
            f"{a1} wants the books",
        ],
        f"{a2.lower()}_thinks_i_want_item": [
            f"{a2} thinks I want the ball",
            f"{a2} thinks I want the hats",
            f"{a2} thinks I want the books",
        ],
        f"{a1.lower()}_thinks_i_want_item": [
            f"{a1} thinks I want the ball",
            f"{a1} thinks I want the hats",
            f"{a1} thinks I want the books",
        ],
        "action_take_item": [
            "I should take the ball",
            "I should take the hats",
            "I should take the books",
        ],
        "action_give_up_item": [
            "I should give up the ball",
            "I should give up the hats",
            "I should give up the books",
        ],
        "action_ask_for_item": [
            "I should ask for the ball",
            "I should ask for the hats",
            "I should ask for the books",
        ],
        "final_take_item": [
            "I will take the ball",
            "I will take the hats",
            "I will take the books",
        ],
        "agreement_state": [
            "We are near agreement",
            "We are far agreement",
        ],
        "agreement_state_fluent": [
            "We are near a deal",
            "We are far from a deal",
        ],
        f"{a2.lower()}_behavior": [
            f"{a2} is being fair",
            f"{a2} is being selfish",
            f"{a2} is being stubborn",
        ],
        f"{a1.lower()}_behavior": [
            f"{a1} is being fair",
            f"{a1} is being selfish",
            f"{a1} is being stubborn",
        ],
        f"{a2.lower()}_strategy": [
            f"{a2} is trying to compromise",
            f"{a2} is trying to maximize points",
            f"{a2} is trying to get the ball",
        ],
        f"{a1.lower()}_strategy": [
            f"{a1} is trying to compromise",
            f"{a1} is trying to maximize points",
            f"{a1} is trying to get the ball",
        ],
    }


def parse_dialog_turns(dialog_record: Dict[str, str]) -> List[Dict[str, Any]]:
    """Parse the stored dialog list string into structured turns."""
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
        turns.append(
            {
                "turn": int(m.group(1)),
                "speaker": m.group(2).strip(),
                "text": m.group(3).strip(),
            }
        )
    return turns


def parse_point_values(intro: str, agent1_name: str, agent2_name: str) -> Dict[str, Dict[str, int]]:
    """Extract per-item values from the intro when available."""
    values: Dict[str, Dict[str, int]] = {}
    for agent_name in (agent1_name, agent2_name):
        pat = (
            re.escape(agent_name)
            + r"'s goal:.*?each book is worth (\d+) points, each hat is worth (\d+) points, and the ball is worth (\d+) points"
        )
        m = re.search(pat, intro, flags=re.DOTALL)
        if m:
            values[agent_name] = {
                "books": int(m.group(1)),
                "hats": int(m.group(2)),
                "ball": int(m.group(3)),
            }
    return values


def parse_scenario_inventory(intro: str) -> Dict[str, int]:
    """
    Parse global item counts from scenario text, e.g. '3 books, 2 hats, and 1 ball'.
    Returns {} if not found (caller may skip validation).
    """
    m = re.search(
        r"(?i)(\d+)\s+books?.*?(\d+)\s+hats?.*?(\d+)\s+balls?",
        intro,
    )
    if m:
        return {
            "books": int(m.group(1)),
            "hats": int(m.group(2)),
            "ball": int(m.group(3)),
        }
    return {}


def _fresh_allocation(agent1_name: str, agent2_name: str) -> Dict[str, Dict[str, int]]:
    z = {"books": 0, "hats": 0, "ball": 0}
    return {agent1_name: dict(z), agent2_name: dict(z)}


def _add_items(
    alloc: Dict[str, Dict[str, int]],
    agent: str,
    books: int = 0,
    hats: int = 0,
    ball: int = 0,
) -> None:
    alloc[agent]["books"] += books
    alloc[agent]["hats"] += hats
    if ball:
        alloc[agent]["ball"] = min(1, alloc[agent]["ball"] + ball)


def _parse_book_hat_qty(fragment: str, inv: Dict[str, int], kind: str) -> int:
    """Return count for 'books' or 'hats' from a short English fragment."""
    f = fragment.lower().strip()
    if kind == "books":
        inv_max = inv.get("books", 0)
        if re.search(r"\b(?:all|every)\b.*\bbooks?\b", f) or (
            re.search(r"\bthe\s+books?\b", f) and not re.search(r"\b(one|two|three|both|\d+)\b", f)
        ):
            return inv_max
        m = re.search(r"\b(both|one|two|three|a|an|\d+)\s+books?\b", f)
        if not m:
            return 0
        w = m.group(1)
        if w == "both":
            return min(2, inv_max) if inv_max else 2
        if w in ("one", "a", "an"):
            return 1
        if w == "two":
            return 2
        if w == "three":
            return 3
        if w.isdigit():
            return int(w)
        return 0
    # hats
    inv_max = inv.get("hats", 0)
    if re.search(r"\b(?:all|every|both)\b.*\bhats?\b", f) or (
        re.search(r"\bthe\s+hats?\b", f) and not re.search(r"\b(one|two|three|both|\d+)\b", f)
    ):
        return inv_max
    if re.search(r"\bone of the hats\b", f) or re.search(r"\b(?:a|one)\s+hat\b", f):
        return 1
    if re.search(r"\bboth\s+hats?\b", f):
        return min(2, inv_max) if inv_max else 2
    m = re.search(r"\b(one|two|three|\d+)\s+hats?\b", f)
    if m:
        w = m.group(1)
        if w == "one":
            return 1
        if w == "two":
            return 2
        if w == "three":
            return 3
        if w.isdigit():
            return int(w)
    return 0


def _parse_ball(fragment: str) -> int:
    return 1 if re.search(r"\bball\b", fragment.lower()) else 0


def _parse_allocation_phrase_to_agent(
    phrase: str,
    recipient: str,
    inv: Dict[str, int],
    alloc: Dict[str, Dict[str, int]],
) -> None:
    """Parse a substring like 'the ball and one of the hats' onto recipient."""
    p = phrase.lower()
    # Split on 'and' for multi-item phrases
    parts = re.split(r"\s+and\s+", p)
    books = hats = ball = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "book" in part:
            books += _parse_book_hat_qty(part, inv, "books")
        elif "hat" in part:
            hats += _parse_book_hat_qty(part, inv, "hats")
        elif "ball" in part:
            ball += _parse_ball(part)
    if books or hats or ball:
        _add_items(alloc, recipient, books=books, hats=hats, ball=ball)


def parse_final_allocation_from_turn(
    text: str,
    speaker: str,
    agent1_name: str,
    agent2_name: str,
    inv: Dict[str, int],
) -> Dict[str, Dict[str, int]] | None:
    """
    Heuristic: extract who receives which items from one turn's text.
    Resolves I/me -> speaker, you -> other, and optional first-name mentions.
    """
    if not inv:
        return None
    other = agent2_name if speaker == agent1_name else agent1_name
    a1_short = short_name(agent1_name)
    a2_short = short_name(agent2_name)
    alloc = _fresh_allocation(agent1_name, agent2_name)
    t = " ".join(text.split())

    # Explicit split: "I'll take … and you can have …" (common in this scenario)
    m_split = re.search(
        r"(?i)I(?:'ll| will)?\s+(?:take|have|get)\s+(.+?)\s+and\s+you\s+can\s+have\s+(.+?)(?:\.|$)",
        t,
    )
    split_ok = False
    if m_split:
        _parse_allocation_phrase_to_agent(m_split.group(1).strip(), speaker, inv, alloc)
        _parse_allocation_phrase_to_agent(m_split.group(2).strip(), other, inv, alloc)
        split_ok = True

    def recv_for_name_block(name_pat: str, recipient: str) -> None:
        for m in re.finditer(
            rf"(?i){name_pat}\s+(?:can|will)\s+have\s+([^\.]+?)(?:\.|$)",
            t,
        ):
            _parse_allocation_phrase_to_agent(m.group(1), recipient, inv, alloc)
        for m in re.finditer(
            rf"(?i){name_pat}\s+(?:takes|will take|gets)\s+([^\.]+?)(?:\.|$)",
            t,
        ):
            _parse_allocation_phrase_to_agent(m.group(1), recipient, inv, alloc)

    recv_for_name_block(re.escape(agent1_name), agent1_name)
    recv_for_name_block(re.escape(agent2_name), agent2_name)
    recv_for_name_block(re.escape(a1_short), agent1_name)
    recv_for_name_block(re.escape(a2_short), agent2_name)

    if not split_ok:
        # I / I'll / I will ... take|have|get ...
        for m in re.finditer(
            r"(?i)(?:I(?:'ll| will)?)\s+(?:take|have|get)\s+([^\.]+?)(?=(?:\.|$|and you|,\s*you))",
            t,
        ):
            _parse_allocation_phrase_to_agent(m.group(1), speaker, inv, alloc)

        # you ... have|get|take
        for m in re.finditer(
            r"(?i)you\s+(?:can\s+)?(?:have|get|take)\s+([^\.]+?)(?:\.|$)",
            t,
        ):
            _parse_allocation_phrase_to_agent(m.group(1), other, inv, alloc)

    return alloc


def validate_allocation(
    alloc: Dict[str, Dict[str, int]],
    inv: Dict[str, int],
    agent1_name: str,
    agent2_name: str,
) -> Tuple[bool, str]:
    if not inv:
        return False, "scenario inventory not parsed"
    for key in ("books", "hats", "ball"):
        s = alloc[agent1_name][key] + alloc[agent2_name][key]
        if s != inv.get(key, -1):
            return False, f"item {key}: allocated {s}, inventory {inv.get(key)}"
    if alloc[agent1_name]["ball"] + alloc[agent2_name]["ball"] > inv.get("ball", 1):
        return False, "ball count invalid"
    return True, "ok"


def compute_outcome_rewards(
    alloc: Dict[str, Dict[str, int]],
    point_values: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    """Private-utility points from final item counts × per-item values."""
    rewards: Dict[str, int] = {}
    for agent_name, counts in alloc.items():
        pv = point_values.get(agent_name, {})
        if not pv:
            rewards[agent_name] = 0
            continue
        r = (
            counts["books"] * pv["books"]
            + counts["hats"] * pv["hats"]
            + counts["ball"] * pv["ball"]
        )
        rewards[agent_name] = int(r)
    return rewards


def infer_final_allocation(
    turns: List[Dict[str, Any]],
    agent1_name: str,
    agent2_name: str,
    inv: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, int]] | None, Dict[str, Any]]:
    """
    Scan late turns (prefer those mentioning take/have/ball/book/hat) for a
    complete allocation that sums to inventory.
    """
    meta: Dict[str, Any] = {
        "source_turn": None,
        "validation": None,
        "method": "heuristic_regex",
    }
    if not inv or not turns:
        meta["validation"] = "skipped_no_inventory_or_turns"
        return None, meta

    candidates: List[Tuple[int, str, Dict[str, Dict[str, int]]]] = []
    for turn in reversed(turns[-12:]):
        text = turn["text"]
        if not re.search(
            r"(?i)(take|have|get|books?|hats?|ball)",
            text,
        ):
            continue
        trial = parse_final_allocation_from_turn(
            text,
            turn["speaker"],
            agent1_name,
            agent2_name,
            inv,
        )
        if trial is None:
            continue
        ok, reason = validate_allocation(trial, inv, agent1_name, agent2_name)
        candidates.append((turn["turn"], reason, trial))
        if ok:
            meta["source_turn"] = turn["turn"]
            meta["validation"] = {"ok": True, "detail": reason}
            return trial, meta

    if candidates:
        # Return best-effort last attempt with explicit invalid flag
        turn_no, reason, trial = candidates[0]
        meta["source_turn"] = turn_no
        meta["validation"] = {"ok": False, "detail": reason}
        return trial, meta

    meta["validation"] = "no_parseable_allocation_in_late_turns"
    return None, meta


def _map_me_you_rewards(speaker: str, agent1_name: str, agent2_name: str, me_pts: int, you_pts: int) -> Dict[str, int]:
    if speaker == agent1_name:
        return {agent1_name: me_pts, agent2_name: you_pts}
    return {agent1_name: you_pts, agent2_name: me_pts}


def parse_claimed_rewards(turn_text: str, speaker: str, agent1_name: str, agent2_name: str) -> Dict[str, int] | None:
    """Heuristically parse claimed point totals from a turn."""
    text = " ".join(turn_text.split())
    patterns = [
        r"I(?:'ll| will)? have (?:a )?total of (\d+) points,? and you(?:'ll| will)? have (\d+) points",
        r"This will give me (?:a )?total of (\d+) points and you (\d+) points",
        r"giving me (?:a )?total of (\d+) points.*?giving you (?:a )?total of (\d+) points",
        r"I(?:'ll| will)? take .*?giving me (?:a )?total of (\d+) points.*?you(?:'ll| will)? take .*?giving you (?:a )?total of (\d+) points",
        r"I(?:'ll| will)? take .*?worth (\d+) points.*?you(?:'ll| will)? take .*?worth (\d+) points",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _map_me_you_rewards(
                speaker,
                agent1_name,
                agent2_name,
                int(m.group(1)),
                int(m.group(2)),
            )
    return None


def summarize_turn_accomplishments(turn_text: str) -> List[str]:
    """Conservative, keyword-based turn accomplishments."""
    text = turn_text.lower()
    accomplishments: List[str] = []
    if any(k in text for k in ["proposal", "propose", "how about", "arrangement", "suggest"]):
        accomplishments.append("proposal or arrangement suggested")
    if "ball" in text and any(k in text for k in ["take the ball", "have the ball", "choose it", "you can have the ball", "i'll take the ball"]):
        accomplishments.append("ball allocation proposed or confirmed")
    if any(k in text for k in ["extra point", "additional point", "slight modification"]):
        accomplishments.append("up-negotiation or side-payment proposed")
    if any(k in text for k in ["agree", "agreement", "satisfied", "confirm", "let's proceed", "finalize"]):
        accomplishments.append("agreement or confirmation")
    if "leave" in text:
        accomplishments.append("conversation termination / leave")
    if any(k in text for k in ["fair", "mutually beneficial", "win-win"]):
        accomplishments.append("fairness framing")
    return accomplishments


def build_behavioral_summary(
    dialog_record: Dict[str, str],
    agent1_name: str,
    agent2_name: str,
) -> Dict[str, Any]:
    """Build a lightweight behavioral summary from the dialog CSV row."""
    intro = dialog_record.get("Intro", "")
    turns = parse_dialog_turns(dialog_record)
    point_values = parse_point_values(intro, agent1_name, agent2_name)
    scenario_inventory = parse_scenario_inventory(intro)
    final_allocation, allocation_inference = infer_final_allocation(
        turns, agent1_name, agent2_name, scenario_inventory
    )
    outcome_rewards: Dict[str, int] | None = None
    if final_allocation is not None and point_values:
        outcome_rewards = compute_outcome_rewards(final_allocation, point_values)

    allocation_validated = False
    v = allocation_inference.get("validation")
    if isinstance(v, dict):
        allocation_validated = bool(v.get("ok"))

    latest_claimed_rewards: Dict[str, int] | None = None
    turn_summaries: List[Dict[str, Any]] = []
    deal_reached = False

    for turn in turns:
        accomplishments = summarize_turn_accomplishments(turn["text"])
        claimed = parse_claimed_rewards(turn["text"], turn["speaker"], agent1_name, agent2_name)
        if claimed:
            latest_claimed_rewards = claimed
        if any(tag in accomplishments for tag in ["agreement or confirmation", "conversation termination / leave"]):
            deal_reached = True
        turn_summaries.append(
            {
                "turn": turn["turn"],
                "speaker": turn["speaker"],
                "accomplishments": accomplishments,
                "claimed_rewards": claimed,
                "text": turn["text"],
            }
        )

    return {
        "participants": [agent1_name, agent2_name],
        "deal_reached": deal_reached,
        "point_values": point_values,
        "scenario_inventory": scenario_inventory,
        "final_allocation_inferred": final_allocation,
        "allocation_inference": allocation_inference,
        "allocation_sums_to_inventory": allocation_validated,
        "outcome_rewards_private_utility": outcome_rewards,
        "final_claimed_rewards": latest_claimed_rewards,
        "turns": turn_summaries,
    }


def turn_to_agent(turn_idx: int) -> int:
    """Agent 1 speaks on even turns (0,2,4,...), agent 2 on odd turns."""
    return 1 if turn_idx % 2 == 0 else 2


# ────────────────────────────────────────────────────────────
# Continuation stems  &  contrastive completions
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
# Core analysis: continuation-stem heatmap
# sequential next-token prediction under logit lens
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build a (n_layers, n_stem_tokens) heatmap for a continuation stem using
    sequential next-token prediction.

    For each stem token position k:
      - run the full wrapped prompt + stem through the model
      - use the hidden state at the position immediately before token k
      - apply the logit lens at each layer
      - record the top-1 predicted token id and its probability

    Returns
    -------
    prob_matrix : (n_layers, n_stem_tokens) top-1 probabilities
    pred_id_matrix : (n_layers, n_stem_tokens) top-1 token ids
    cont_prob_vec : (n_layers,) top-1 probability after the full stem
    cont_pred_id_vec : (n_layers,) top-1 token id after the full stem
    stem_tokens : tokenized stem (for x-axis labels)
    layer_labels : ["Embed", "1", ..., "32"]
    """
    prompt_text = wrap_inst(prompt)
    full_text = prompt_text + stem

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
    full_inputs = tokenizer(full_text, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_input_ids = full_inputs["input_ids"].to(device)

    # stem tokens are exactly the tokens after the prompt prefix
    stem_ids = full_input_ids[0, prompt_len:].detach().cpu().tolist()
    stem_tokens = tokenizer.convert_ids_to_tokens(stem_ids)

    outputs = model.model(full_input_ids, output_hidden_states=True)
    hidden_states = tuple(h.detach().float().cpu() for h in outputs.hidden_states)

    n_layers = len(hidden_states)
    n_stem = len(stem_ids)

    prob_matrix = np.zeros((n_layers, n_stem), dtype=np.float32)
    pred_id_matrix = np.zeros((n_layers, n_stem), dtype=np.int64)
    cont_prob_vec = np.zeros(n_layers, dtype=np.float32)
    cont_pred_id_vec = np.zeros(n_layers, dtype=np.int64)

    # To predict stem token at absolute position pos, use hidden state at pos-1
    target_positions = list(range(prompt_len, full_input_ids.shape[1]))

    for layer_idx, hs in enumerate(hidden_states):
        for col, pos in enumerate(target_positions):
            h = hs[0, pos - 1, :]  # predicts token at `pos`
            logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
            probs = torch.softmax(logits[0], dim=-1)
            top_prob, top_id = torch.max(probs, dim=-1)

            prob_matrix[layer_idx, col] = top_prob.item()
            pred_id_matrix[layer_idx, col] = top_id.item()

        # After the full stem, use the hidden state at the final stem token
        h = hs[0, full_input_ids.shape[1] - 1, :]
        logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
        probs = torch.softmax(logits[0], dim=-1)
        top_prob, top_id = torch.max(probs, dim=-1)
        cont_prob_vec[layer_idx] = top_prob.item()
        cont_pred_id_vec[layer_idx] = top_id.item()

    layer_labels = ["Embed"] + [str(i) for i in range(1, n_layers)]
    return prob_matrix, pred_id_matrix, cont_prob_vec, cont_pred_id_vec, stem_tokens, layer_labels


def _clean_token(tok: str, max_len: int = 10) -> str:
    """Pretty-print tokenizer tokens for heatmap overlays."""
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
    """
    Heatmap colored by top-1 probability, with predicted token overlaid in each cell.
    Columns correspond to sequential stem-token positions.
    """
    n_layers, n_tokens = prob_matrix.shape
    fig_w = max(7, 1.4 * n_tokens)
    fig_h = max(6, 0.45 * n_layers)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(prob_matrix, aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax, label="Top-1 next-token probability")

    # x-axis: show the actual stem tokens at those positions
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(
        [_clean_token(t) for t in stem_tokens],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=7)

    ax.set_xlabel("Sequential stem token position")
    ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=9)

    # Overlay predicted tokens
    vmax = float(prob_matrix.max()) if prob_matrix.size else 1.0
    for i in range(n_layers):
        for j in range(n_tokens):
            pred_tok = tokenizer.convert_ids_to_tokens([int(pred_id_matrix[i, j])])[0]
            pred_tok = _clean_token(pred_tok, max_len=8)

            # White text on darker cells, black on brighter cells
            color = "white" if prob_matrix[i, j] < 0.5 * vmax else "black"
            ax.text(
                j, i, pred_tok,
                ha="center", va="center",
                fontsize=6, color=color,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
# Core analysis: contrastive scoring
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
# Core analysis: contrastive scoring
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
) -> np.ndarray:
    """
    Returns (n_layers, n_completions) matrix of average conditional token
    log-prob scores under the layerwise logit lens.

    For each completion:
      1. Run the full wrapped prompt + completion through the model.
      2. At each layer, decode logits at every position.
      3. Score only the continuation tokens token-by-token.
      4. Average over continuation token count to normalize for length.

    hidden_states includes embeddings at index 0, then transformer layers.
    """
    all_scores = []

    for completion in completions:
        full_text = wrap_inst(prompt) + completion
        full_inputs = tokenizer(full_text, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(device)

        prompt_text = wrap_inst(prompt)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        outputs = model.model(full_input_ids, output_hidden_states=True)
        hidden_states = tuple(h.detach().float().cpu() for h in outputs.hidden_states)

        n_layers = len(hidden_states)
        layer_scores = np.zeros(n_layers, dtype=np.float32)

        # continuation tokens occupy positions [prompt_len, ..., seq_len-1]
        # token at position pos-1 predicts token at position pos
        target_positions = list(range(prompt_len, full_input_ids.shape[1]))
        if len(target_positions) == 0:
            layer_scores[:] = -np.inf
            all_scores.append(layer_scores)
            continue

        input_ids_cpu = full_input_ids.detach().cpu()

        for layer_idx, hs in enumerate(hidden_states):
            total = 0.0
            count = 0

            for pos in target_positions:
                h = hs[0, pos - 1, :]  # predict token at `pos`
                logits, _ = logit_lens_decode(h.unsqueeze(0), W_U, rms_weight, eps)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                target_id = input_ids_cpu[0, pos].item()
                total += log_probs[target_id].item()
                count += 1

            layer_scores[layer_idx] = total / count

        all_scores.append(layer_scores)

    return np.stack(all_scores, axis=1)  # (n_layers, n_completions)


def plot_contrastive_class(
    scores: np.ndarray,
    completions: List[str],
    class_name: str,
    title: str,
    save_path: str,
) -> None:
    """Line plot: one line per completion, x = layer, y = avg log-prob."""
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


# ────────────────────────────────────────────────────────────
# Summary plots across turns
# ────────────────────────────────────────────────────────────

def plot_contrastive_across_turns(
    all_scores: Dict[int, np.ndarray],
    completions: List[str],
    class_name: str,
    layer: int,
    title: str,
    save_path: str,
) -> None:
    """Bar chart: for a fixed layer, show scores at each turn."""
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
# Main
# ────────────────────────────────────────────────────────────

def _agent_label(agent: int, model_short: str) -> str:
    return f"Agent{agent} ({model_short})"


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit-lens analysis for Mistral dialog agents.")
    parser.add_argument("--model_1", type=str, required=True,
                        help="Short name of agent-1 model (e.g. Mistral-7B-Instruct-v0.3)")
    parser.add_argument("--model_2", type=str, required=True,
                        help="Short name of agent-2 model")
    parser.add_argument("--episode", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="Hugging Face Hub cache directory (avoids ~/.cache quota issues). "
        "Also reads LM_HF_CACHE or HF_HUB_CACHE if unset.",
    )
    parser.add_argument("--summary_layers", type=int, nargs="+",
                        default=[0, 8, 16, 24, 31],
                        help="Layers at which to plot cross-turn contrastive summaries")
    args = parser.parse_args()

    hf_cache = resolve_hf_cache_dir(args.hf_cache)
    if hf_cache:
        os.makedirs(hf_cache, exist_ok=True)
        print(f"Hugging Face cache: {hf_cache}")
    else:
        print("Hugging Face cache: (default; set --hf-cache or LM_HF_CACHE if quota errors)")

    affected_type = "None"
    value = 0
    m1_full = f"{args.model_1}_{affected_type}_{value}"
    m2_full = f"{args.model_2}_{affected_type}_{value}"
    pair_name = f"{m1_full}_{m2_full}"

    dialog_record = load_dialog_record(pair_name, args.episode, args.seed, args.temp)
    agent1_name, agent2_name = infer_agent_names(dialog_record)
    continuation_stems = build_continuation_stems(agent1_name, agent2_name)
    contrastive_classes = build_contrastive_classes(agent1_name, agent2_name)

    print(f"Pair : {pair_name}")
    print(f"Agent 1: {args.model_1}  |  Agent 2: {args.model_2}")
    print(f"Participants: {agent1_name}  |  {agent2_name}")
    print(f"Episode: {args.episode}, seed={args.seed}, temp={args.temp}")

    # ── output dir ──
    out_dir = os.path.join(
        REPO_ROOT, "logit_lens_results", pair_name, f"episode_{args.episode}",
    )
    os.makedirs(out_dir, exist_ok=True)
    stem_dir = os.path.join(out_dir, "stem_heatmaps")
    os.makedirs(stem_dir, exist_ok=True)
    contrastive_dir = os.path.join(out_dir, "contrastive")
    os.makedirs(contrastive_dir, exist_ok=True)
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    behavioral_summary = build_behavioral_summary(dialog_record, agent1_name, agent2_name)
    save_json(os.path.join(out_dir, "behavioral_summary.json"), behavioral_summary)

    # ── load both models (mirrors SotopiaEnv which always loads both) ──
    print(f"Loading agent-1 model ({args.model_1}) ...")
    tok1, mdl1 = load_mistral(args.model_1, args.device, cache_dir=hf_cache)
    W_U1, rms_w1, eps1 = get_unembedding_params(mdl1)
    W_U1, rms_w1 = W_U1.cpu(), rms_w1.cpu()
    print(f"  layers={mdl1.config.num_hidden_layers}, "
          f"d={mdl1.config.hidden_size}, V={mdl1.config.vocab_size}")

    print(f"Loading agent-2 model ({args.model_2}) ...")
    tok2, mdl2 = load_mistral(args.model_2, args.device, cache_dir=hf_cache)
    W_U2, rms_w2, eps2 = get_unembedding_params(mdl2)
    W_U2, rms_w2 = W_U2.cpu(), rms_w2.cpu()
    print(f"  layers={mdl2.config.num_hidden_layers}, "
          f"d={mdl2.config.hidden_size}, V={mdl2.config.vocab_size}")

    agent_resources = {
        1: (tok1, mdl1, W_U1, rms_w1, eps1, args.model_1),
        2: (tok2, mdl2, W_U2, rms_w2, eps2, args.model_2),
    }

    # ── load prompts ──
    prompts = load_prompt_records(pair_name, args.episode, args.seed, args.temp)
    n_turns = len(prompts)
    print(f"  {n_turns} total dialog turns")

    # ── per-turn analysis (both agents, interleaved) ──
    contrastive_all: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {
        1: {cls: {} for cls in contrastive_classes},
        2: {cls: {} for cls in contrastive_classes},
    }
    stem_summary: Dict[str, Any] = {
        "pair": pair_name,
        "episode": args.episode,
        "model_1": args.model_1,
        "model_2": args.model_2,
        "participants": {"agent_1": agent1_name, "agent_2": agent2_name},
        "turns": [],
    }

    for turn_idx in range(n_turns):
        agent = turn_to_agent(turn_idx)
        tok, mdl, W_U, rms_w, eps, model_short = agent_resources[agent]
        label = _agent_label(agent, model_short)
        prompt = prompts[turn_idx]
        speaker_name = agent1_name if agent == 1 else agent2_name
        print(f"\n─── Turn {turn_idx}  ({label}) ───")

        hidden_states, tokens, _ = get_hidden_states(prompt, tok, mdl, args.device)
        print(f"  Prompt tokens: {len(tokens)}")

        # ── Stem heatmaps ──
        turn_stem_dir = os.path.join(stem_dir, f"turn_{turn_idx}_agent{agent}")
        os.makedirs(turn_stem_dir, exist_ok=True)
        turn_stem_summary: Dict[str, Any] = {
            "turn": turn_idx,
            "agent_index": agent,
            "agent_identity": speaker_name,
            "model": model_short,
            "stems": {},
        }

        for si, stem in enumerate(continuation_stems):
            (
                prob_mat,
                pred_id_mat,
                cont_prob_vec,
                cont_pred_id_vec,
                stem_toks,
                layer_labels,
            ) = stem_heatmap_for_turn(
                prompt, stem, tok, mdl, W_U, rms_w, eps, args.device,
            )
            safe_stem = stem.replace(" ", "_").replace("/", "_")[:40]
            save_path = os.path.join(turn_stem_dir, f"{si:02d}_{safe_stem}.png")
            plot_stem_heatmap(
                prob_mat, pred_id_mat, stem_toks, layer_labels, tok,
                title=f"{label} | Turn {turn_idx} | \"{stem}\"",
                save_path=save_path,
            )
            per_layer = []
            for layer_idx, layer_name in enumerate(layer_labels):
                pred_token = tok.convert_ids_to_tokens([int(cont_pred_id_vec[layer_idx])])[0]
                per_layer.append(
                    {
                        "layer_index": layer_idx,
                        "layer_name": layer_name,
                        "predicted_token": pred_token,
                        "predicted_token_clean": _clean_token(pred_token, max_len=30),
                        "probability": float(cont_prob_vec[layer_idx]),
                    }
                )
            turn_stem_summary["stems"][stem] = {
                "stem_tokens": stem_toks,
                "continuation_prediction_after_full_stem": per_layer,
            }
        save_json(os.path.join(turn_stem_dir, "stem_summary.json"), turn_stem_summary)
        stem_summary["turns"].append(turn_stem_summary)

        # ── Contrastive completions ──
        turn_contr_dir = os.path.join(contrastive_dir, f"turn_{turn_idx}_agent{agent}")
        os.makedirs(turn_contr_dir, exist_ok=True)

        for cls_name, completions in contrastive_classes.items():
            scores = score_contrastive_class(
                prompt, mdl, tok, completions, W_U, rms_w, eps, args.device,
            )
            contrastive_all[agent][cls_name][turn_idx] = scores

            save_path = os.path.join(turn_contr_dir, f"{cls_name}.png")
            plot_contrastive_class(
                scores, completions, cls_name,
                title=f"{label} | Turn {turn_idx} | {cls_name}",
                save_path=save_path,
            )

        np.savez_compressed(
            os.path.join(turn_contr_dir, "contrastive_scores.npz"),
            **{cls: contrastive_all[agent][cls][turn_idx]
               for cls in contrastive_classes},
        )

        print(f"  Saved {len(continuation_stems)} stem heatmaps, "
              f"{len(contrastive_classes)} contrastive plots")

    # ── Summary across turns (per agent) ──
    print("\n─── Cross-turn summaries ───")
    for agent in (1, 2):
        model_short = agent_resources[agent][5]
        label = _agent_label(agent, model_short)
        for layer in args.summary_layers:
            layer_dir = os.path.join(summary_dir, f"agent{agent}", f"layer_{layer}")
            os.makedirs(layer_dir, exist_ok=True)

            for cls_name, completions in contrastive_classes.items():
                turn_scores = contrastive_all[agent][cls_name]
                if not turn_scores:
                    continue
                save_path = os.path.join(layer_dir, f"{cls_name}.png")
                plot_contrastive_across_turns(
                    turn_scores, completions, cls_name, layer,
                    title=f"{label} | Layer {layer} | {cls_name}",
                    save_path=save_path,
                )

    # ── Save all contrastive scores as a single JSON ──
    summary_json: Dict[str, Any] = {
        "pair": pair_name,
        "model_1": args.model_1,
        "model_2": args.model_2,
        "episode": args.episode,
        "n_turns": n_turns,
        "participants": {
            "agent_1": agent1_name,
            "agent_2": agent2_name,
        },
    }
    for agent in (1, 2):
        agent_key = f"agent_{agent}"
        summary_json[agent_key] = {}
        for cls_name in contrastive_classes:
            summary_json[agent_key][cls_name] = {}
            for t, arr in contrastive_all[agent][cls_name].items():
                summary_json[agent_key][cls_name][str(t)] = arr.tolist()
    save_json(os.path.join(out_dir, "contrastive_summary.json"), summary_json)
    save_json(os.path.join(out_dir, "stem_summary.json"), stem_summary)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
