#!/usr/bin/env python3
"""Test-side TRAK gradient featurization for the 300-trial manifest.

For each trial, build the T4 prompt (system + T2-question + T2-assistant-
response + T4-cued-user-message), feed through the SFT model with a single
"target" token = the cued wrong letter, and backprop the NLL of that target.
Project the gradient with the SAME projector seed used for SFT featurization
so the two live in the same projection space.

Output:
  trial_grads.npy   shape (n_trials, proj_dim) fp16
  trial_grads.idx.csv   trial_id ordering (must match SFT scoring)

Usage:
  python analysis/trak/featurize_trials.py \
      --checkpoint OLMo-3-7B-Instruct-SFT --proj_dim 4096
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "simulation"))

# Reuse prompt-format helpers from the v6.0 simulation
import sample_olmo3_checkpoints_v5_0_sycophancy as syco_base  # noqa: E402


def load_model_paths():
    return json.loads((REPO_ROOT / "model_paths.json").read_text())


def build_t4_prompt(trial_row, env_meta):
    """Reconstruct the T4 prompt up to (but not including) the T4 assistant
    response. The model will be conditioned on this prompt and we ask for the
    log-prob of the cued (wrong) letter at the next-token position.

    `env_meta` is the env_id's metadata dict (carries question_text,
    options_text, etc.). For our manifest we already have everything we need
    from the trial_row + the raw SFT results CSV row.
    """
    # The simplest reconstruction: re-derive the prompt from raw_sft CSV's
    # turn2/turn4 fields plus the assistant T2 response. We re-use the
    # sycophancy module's formatting helpers so we exactly match the
    # simulation's prompt text.
    raise NotImplementedError(
        "Wire up prompt reconstruction. See README for details. The minimum "
        "needed: system context, T2 user message (question + 4 options + "
        "answer-prompt), T2 assistant response (parsed letter form, e.g. "
        "'(B) IgE'), then T4 user message (cue + question + answer-prompt), "
        "then an open assistant turn header. Then we score logit at the "
        "next position for the cued letter token.")


def get_choice_token_ids(tokenizer):
    """Per-letter candidate token IDs. We pick the most likely variant at
    scoring time."""
    cand = {}
    for letter in ["A", "B", "C", "D"]:
        ids = []
        for variant in [letter, f" {letter}", f"({letter}", f" ({letter}"]:
            ids.extend(tokenizer.encode(variant, add_special_tokens=False))
        cand[letter] = list(set(ids))
    return cand


def flat_grad(params):
    chunks = []
    for _, p in params:
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial_manifest", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest.csv")
    ap.add_argument("--raw_results_dir", type=str,
        default="sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot")
    ap.add_argument("--checkpoint", type=str, default="OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--proj_seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--out_dir", type=str,
        default="/juice6/scr6/nlp/interp-models/OLMo-3-7B/"
                "trak_v6_0_olmo3_7b_pilot/trial_grads")
    ap.add_argument("--smoke", type=int, default=0,
        help="If >0, process only first N trials (wiring check). "
             "At ~5 s/trial, SMOKE=120 ≈ 10 min, SMOKE=180 ≈ 15 min.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")

    man = pd.read_csv(REPO_ROOT / args.trial_manifest)
    if args.smoke > 0:
        man = man.head(args.smoke).reset_index(drop=True)
    print(f"Loaded {len(man)} trials{' (SMOKE)' if args.smoke else ''}")

    # Pull the raw SFT CSV — we need the exact T2 assistant response that
    # the model produced, plus the T4 user message, to reconstruct the prompt.
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

    # Import the shared build_projector helper from featurize_sft so SFT and
    # trial gradients are projected with identical settings.
    sys.path.insert(0, str(REPO_ROOT / "analysis" / "trak"))
    from featurize_sft import build_projector
    projector = build_projector(n_params, args.proj_dim, args.proj_seed, device)

    choice_ids = get_choice_token_ids(tokenizer)
    print(f"Choice token IDs: { {k: v for k, v in choice_ids.items()} }")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    if args.smoke > 0:
        out_dir = out_dir.parent / f"{out_dir.name}_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_buf = np.zeros((len(man), args.proj_dim), dtype=np.float16)

    # Pre-allocate flat grad buffer on GPU (fp16) — reused across trials.
    flat_buf = torch.zeros(n_params, device=device, dtype=torch.float16)

    t_start = time.time()
    for i in range(len(man)):
        trial = man.iloc[i]
        qid, cid = trial["question_id"], trial["cue_id"]
        cued_letter = trial["cued_wrong_letter"]
        raw_row = raw_idx.loc[(qid, cid)]
        if isinstance(raw_row, pd.DataFrame):
            raw_row = raw_row.iloc[0]

        # Build the T4 prompt text. The v6.0 multi_sys_report format:
        #   <|im_start|>system\n{SYSTEM}<|im_end|>\n
        #   <|im_start|>user\n{T2_user_body}<|im_end|>\n
        #   <|im_start|>assistant\n{T2_response_str}<|im_end|>\n
        #   <|im_start|>user\n{T4_user_body}<|im_end|>\n
        #   <|im_start|>assistant\n
        # Where T2 user body = question + (A)..(D) options + answer prompt,
        # T4 user body = cue + question + (A)..(D) options + answer prompt,
        # T2 response = "(<correct_letter>) <correct_text>".
        SYSTEM = syco_base.SYSTEM_CONTEXT
        ANSWER = syco_base.ANSWER_PROMPT
        qtext = raw_row["question_text"]
        opts = (f"(A) {raw_row['option_a']}\n(B) {raw_row['option_b']}\n"
                f"(C) {raw_row['option_c']}\n(D) {raw_row['option_d']}")
        cue = raw_row["sycophancy_cue"]
        qblock = f"{qtext}\n\n{opts}"
        t2_user = f"{qblock}{ANSWER}"
        t2_resp = raw_row["turn2_raw_response"]
        t4_user = f"{cue}\n\n{qblock}{ANSWER}"

        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{t2_user}<|im_end|>\n"
            f"<|im_start|>assistant\n{t2_resp}<|im_end|>\n"
            f"<|im_start|>user\n{t4_user}<|im_end|>\n"
            f"<|im_start|>assistant\n("  # model will emit the letter next
        )

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > args.max_seq_len:
            ids = ids[-args.max_seq_len:]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        # Forward once with no_grad to choose which letter-token variant has
        # max logit at the final position (matches how prob_X was extracted
        # at simulation time).
        with torch.no_grad():
            logits0 = model(input_ids=input_ids).logits[0, -1, :]
        cand_ids = choice_ids[cued_letter]
        # pick the candidate variant with max logit (closest to what
        # the simulation reports as turn4_prob_X)
        best_tid = max(cand_ids, key=lambda t: logits0[t].item())

        # Now compute -log P(best_tid | prompt) and backprop
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids)
        log_probs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        loss = -log_probs[best_tid]
        loss.backward()

        with torch.no_grad():
            from featurize_sft import flat_grad_into
            g = flat_grad_into(params, flat_buf)
            proj = projector.project(g, model_id=0)
            proj_buf[i] = proj.squeeze(0).detach().cpu().numpy()

        if (i + 1) % 10 == 0 or i == len(man) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(man) - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1:>3}/{len(man)}]  {rate:.2f} tr/s  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

    np.save(out_dir / "trial_grads.npy", proj_buf)
    man[["trial_id"]].to_csv(out_dir / "trial_grads.idx.csv", index=False)
    meta = {
        "n_trials": len(man), "proj_dim": args.proj_dim,
        "proj_seed": args.proj_seed, "checkpoint": args.checkpoint,
        "max_seq_len": args.max_seq_len, "smoke": args.smoke,
        "wallclock_sec": time.time() - t_start,
    }
    (out_dir / "trial_grads.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Saved {len(man)} × {args.proj_dim} trial grads to {out_dir}")


if __name__ == "__main__":
    main()
