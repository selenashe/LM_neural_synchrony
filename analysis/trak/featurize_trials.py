#!/usr/bin/env python3
"""Test-side TRAK gradient featurization for the 300-trial manifest.

For each trial we want the gradient of (approximately) the SAME scalar
function f used by the SFT side, evaluated on a trial-specific input. f is
the mean next-token cross-entropy over an assistant-text span:
model(input_ids, labels=labels).loss with labels=-100 outside the span.
This identity matters for TRAK — its whitening (ΦᵀΦ)⁻¹ only yields an
interpretable influence score when the trial- and SFT-side gradients are
∇f of the same f, differing only in the example z. (See featurize_sft.py
for the SFT-side computation.)

To match, we append a short assistant message whose tokens we attribute
toward — by default the canonical "({letter}) {option_text}" rendering,
or (with --use_emitted) the leading "({letter}) ..." clause extracted
from the model's actual T4 response. We build input_ids + loss_mask
through the shared SFT helper (build_chat_text_and_mask), force the loss
span to ONLY the cued-answer CONTENT tokens, and call
model(input_ids, labels=labels).loss. The eventual dot product
φ(trial)ᵀ (ΦᵀΦ)⁻¹ φ(sft) then approximates the leave-one-out influence
of each SFT example on the model's tendency to produce the cued answer
at T4.

Two deliberate asymmetries vs SFT, both made to keep the trial gradient
informative on a short target span:
  1. We mask out the "<|im_end|>\\n" trailer in the trial loss. On SFT
     the trailer is part of a real multi-token response; on the trial it
     is a generic turn-closer that is identical across all 300 trials
     and would dominate the gradient toward a constant direction.
  2. The SFT loss span is a full assistant response (dozens–hundreds of
     tokens); the trial span is ~5–15 tokens. Length normalization
     ("mean") means each side's gradient is scaled by 1/L with different
     L. We accept this as an interpretation ceiling — validate via the
     removal counterfactual rather than trying to fabricate a comparable
     "answer span" inside general SFT examples.

Output (per shard):
  trial_grads_shard{NN}_of{MM}.npy        shape (n_in_shard, proj_dim) fp16
  trial_grads_shard{NN}_of{MM}.idx.csv    trial_id ordering for this shard
  trial_grads_shard{NN}_of{MM}.meta.json  per-shard meta (shard_idx, n,
                                          n_empty, n_emitted_target,
                                          wallclock_sec, ...)

score_trak.py concatenates shards in shard_idx order; trial_ids across
all shards must match SFT scoring's row ordering.

Usage:
  # Single-process (the legacy 300-trial pilot)
  python analysis/trak/featurize_trials.py \\
      --checkpoint OLMo-3-7B-Instruct-SFT --proj_dim 4096 [--use_emitted]

  # 15-way SLURM array (the all-4,432-trials expansion)
  python analysis/trak/featurize_trials.py \\
      --trial_manifest analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv \\
      --shard_idx ${SLURM_ARRAY_TASK_ID} --num_shards 15 \\
      --out_dir /juice6/scr6/.../trial_grads_all
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "analysis" / "trak"))

# Prompt-format constants from the v6.0 simulation, plus the SFT-side
# tokenization/projection helpers so trial and SFT gradients share f AND P.
import sample_olmo3_checkpoints_v5_0_sycophancy as syco_base  # noqa: E402
from featurize_sft import (  # noqa: E402
    build_chat_text_and_mask, build_projector, flat_grad_into,
)


def load_model_paths():
    return json.loads((REPO_ROOT / "model_paths.json").read_text())


def extract_answer_span(emitted, cued_letter):
    """Extract the leading "({letter}) {text}" answer-bearing clause from
    the model's emitted T4 response.

    Returns the substring from the start of the first line up to the first
    sentence boundary ([.!?] followed by whitespace or end-of-string), or
    the entire first line if no sentence boundary is found. Returns None
    if the response does not start with "({cued_letter})" — caller falls
    back to canonical rendering.

    Targeting just this clause (not the whole emitted response) keeps the
    trial loss span on the cued assertion itself, not surrounding hedging
    or reasoning, and avoids reintroducing the length asymmetry the trial
    side is trying to limit.
    """
    text = emitted.lstrip()
    if not text.startswith(f"({cued_letter})"):
        return None
    first_line = text.split("\n", 1)[0]
    # Require [.!?] followed by whitespace OR end-of-string so we don't
    # split inside decimals like "Type 1.5 diabetes".
    m = re.match(r"^(.+?[.!?])(?:\s+|$)", first_line)
    return m.group(1) if m else first_line


def build_t4_messages(raw_row, cued_letter, use_emitted=True):
    """Build the T4 conversation as a chat-format messages list.

    Returns (prefix_messages, target_text, target_is_emitted):
      prefix_messages   = [system, T2-user, T2-assistant, T4-user]
      target_text       = the assistant answer we attribute toward.
                          If use_emitted and raw_row has a
                          turn4_raw_response whose leading "({letter}) ..."
                          clause can be extracted, that span is used.
                          Otherwise we fall back to the canonical
                          "({letter}) {option_text}".
      target_is_emitted = True iff the emitted-response path was taken
                          (audit count — surfaces in meta).
    """
    SYSTEM = syco_base.SYSTEM_CONTEXT
    ANSWER = syco_base.ANSWER_PROMPT
    qtext = raw_row["question_text"]
    opts_map = {"A": raw_row["option_a"], "B": raw_row["option_b"],
                "C": raw_row["option_c"], "D": raw_row["option_d"]}
    opts = "\n".join(f"({k}) {v}" for k, v in opts_map.items())
    cue = raw_row["sycophancy_cue"]
    qblock = f"{qtext}\n\n{opts}"
    t2_user = f"{qblock}{ANSWER}"
    t2_resp = raw_row["turn2_raw_response"]
    t4_user = f"{cue}\n\n{qblock}{ANSWER}"

    target_text = f"({cued_letter}) {opts_map[cued_letter]}"
    target_is_emitted = False
    if use_emitted and "turn4_raw_response" in raw_row.index:
        emitted = raw_row["turn4_raw_response"]
        if pd.notna(emitted):
            extracted = extract_answer_span(str(emitted), cued_letter)
            if extracted is not None:
                target_text = extracted
                target_is_emitted = True

    prefix = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": t2_user},
        {"role": "assistant", "content": t2_resp},
        {"role": "user", "content": t4_user},
    ]
    return prefix, target_text, target_is_emitted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial_manifest", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv",
        help="Trial manifest. Default is the 4,432-row all-trials manifest "
             "from `sample_trials.py --all_trials`.")
    ap.add_argument("--raw_results_dir", type=str,
        default="sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot")
    ap.add_argument("--checkpoint", type=str, default="OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--proj_seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--out_dir", type=str,
        default="/juice6/scr6/nlp/interp-models/OLMo-3-7B/"
                "trak_v6_0_olmo3_7b_pilot/trial_grads_all")
    ap.add_argument("--use_emitted", action="store_true",
        help="Target the leading '({letter}) ...' clause of the model's "
             "actual T4 response when it expresses the cued letter; else "
             "fall back to canonical '({letter}) {option_text}'.")
    ap.add_argument("--shard_idx", type=int, default=0,
        help="0-indexed shard to process (for SLURM array). With "
             "--num_shards N, the manifest is sliced into N contiguous "
             "linspace pieces and only this shard is featurized.")
    ap.add_argument("--num_shards", type=int, default=1,
        help="Total number of shards. Default 1 = single-process "
             "(processes the entire manifest in one job).")
    ap.add_argument("--smoke", type=int, default=0,
        help="If >0, process only first N trials of THIS shard "
             "(wiring check). At ~7.5 s/trial, SMOKE=120 ≈ 15 min. "
             "For true global smoke, use --num_shards 1 --shard_idx 0.")
    args = ap.parse_args()
    assert 0 <= args.shard_idx < args.num_shards, (
        f"--shard_idx {args.shard_idx} out of range [0, {args.num_shards})")

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")

    man_full = pd.read_csv(REPO_ROOT / args.trial_manifest)
    n_total = len(man_full)
    bounds = np.linspace(0, n_total, args.num_shards + 1, dtype=int)
    lo, hi = int(bounds[args.shard_idx]), int(bounds[args.shard_idx + 1])
    man = man_full.iloc[lo:hi].reset_index(drop=True)
    if args.smoke > 0:
        man = man.head(args.smoke).reset_index(drop=True)
    print(f"Manifest: {n_total} total trials → shard {args.shard_idx}/"
          f"{args.num_shards-1} = rows [{lo}, {hi}) = {len(man)} trials"
          f"{' (SMOKE)' if args.smoke else ''}")

    # Pull the raw SFT CSV — we need the exact T2 assistant response that
    # the model produced (and, if --use_emitted, the T4 response), plus the
    # T4 user message, to reconstruct the prompt.
    raw_csv = (REPO_ROOT / args.raw_results_dir
               / f"results_{args.checkpoint}_seed0.csv")
    raw = pd.read_csv(raw_csv)
    raw_idx = raw.set_index(["question_id", "cue_id"])

    paths = load_model_paths()
    model_path = paths[args.checkpoint]
    print(f"Loading model from {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for _, p in params)
    print(f"Model: {args.checkpoint}  n_params={n_params:,}")
    projector = build_projector(n_params, args.proj_dim, args.proj_seed, device)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if args.smoke > 0:
        out_dir = out_dir.parent / f"{out_dir.name}_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_buf = np.zeros((len(man), args.proj_dim), dtype=np.float16)

    # Pre-allocate flat grad buffer on GPU (fp16) — reused across trials.
    flat_buf = torch.zeros(n_params, device=device, dtype=torch.float16)

    # Trailer token count — used to mask the "<|im_end|>\n" trailer out of
    # the trial loss span (see module docstring on the two deliberate
    # asymmetries vs SFT).
    n_trailer = len(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))

    n_empty = 0; n_emitted = 0
    t_start = time.time()
    for i in range(len(man)):
        trial = man.iloc[i]
        qid, cid = trial["question_id"], trial["cue_id"]
        cued_letter = trial["cued_wrong_letter"]
        raw_row = raw_idx.loc[(qid, cid)]
        if isinstance(raw_row, pd.DataFrame):
            raw_row = raw_row.iloc[0]

        prefix_msgs, target_text, tgt_emitted = build_t4_messages(
            raw_row, cued_letter, use_emitted=args.use_emitted)
        if tgt_emitted:
            n_emitted += 1
        # Tokenize the prefix via the SFT helper; we discard its assistant
        # mask because we only want loss on the cued-answer target below.
        prefix_ids, _ = build_chat_text_and_mask(
            prefix_msgs, tokenizer, max_seq_len=None)
        # Tokenize the target as a single assistant turn. The helper marks
        # content+trailer as loss=1; we then zero out the trailer span so
        # loss lands on the cued-answer CONTENT tokens only.
        target_msg = [{"role": "assistant", "content": target_text}]
        target_ids, target_mask = build_chat_text_and_mask(
            target_msg, tokenizer, max_seq_len=None)
        if n_trailer <= len(target_mask):
            target_mask[-n_trailer:] = [0] * n_trailer

        ids = prefix_ids + target_ids
        mask = [0] * len(prefix_ids) + target_mask
        # Left-truncate (preserve the target span) if combined exceeds limit.
        if len(ids) > args.max_seq_len:
            overflow = len(ids) - args.max_seq_len
            ids = ids[overflow:]
            mask = mask[overflow:]
        if sum(mask) == 0:
            n_empty += 1
            print(f"  trial {i} → target span empty after truncation, skipping")
            continue

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        labels = input_ids.clone()
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        labels[~mask_t] = -100

        # Masked-mean CE matching the SFT-side reduction (modulo the
        # deliberate asymmetries noted in the module docstring).
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()

        with torch.no_grad():
            g = flat_grad_into(params, flat_buf)
            proj = projector.project(g, model_id=0)
            proj_buf[i] = proj.squeeze(0).detach().cpu().numpy()

        if (i + 1) % 10 == 0 or i == len(man) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(man) - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1:>3}/{len(man)}]  {rate:.2f} tr/s  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m  "
                  f"empty={n_empty}  emitted={n_emitted}")

    stem = (f"trial_grads_shard{args.shard_idx:02d}"
            f"_of{args.num_shards:02d}")
    np.save(out_dir / f"{stem}.npy", proj_buf)
    man[["trial_id"]].to_csv(out_dir / f"{stem}.idx.csv", index=False)
    meta = {
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "lo": lo, "hi": hi,
        "n_trials": len(man), "n_empty": n_empty,
        "n_emitted_target": n_emitted, "use_emitted": args.use_emitted,
        "proj_dim": args.proj_dim, "proj_seed": args.proj_seed,
        "checkpoint": args.checkpoint, "max_seq_len": args.max_seq_len,
        "smoke": args.smoke, "wallclock_sec": time.time() - t_start,
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Saved {len(man)} × {args.proj_dim} trial grads to "
          f"{out_dir}/{stem}.npy")
    print(f"Empty-target skipped: {n_empty}; "
          f"emitted-target used: {n_emitted}/{len(man)}")


if __name__ == "__main__":
    main()
