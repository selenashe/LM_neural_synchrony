#!/usr/bin/env python3
"""Per-shard SFT-side TRAK gradient featurization for OLMo-3-7B-Instruct-SFT.

For each example in the assigned shard slice of the 50k subsample, computes
the standard SFT next-token cross-entropy loss on assistant tokens, then
backprops to get the per-example gradient w.r.t. all trainable params, then
JL-projects to `proj_dim` with TRAK's CudaProjector.

Output:
  {out_dir}/sft_grads_shard{NN}_of{M}.npy   shape (n_examples_in_shard, proj_dim) fp16
  {out_dir}/sft_grads_shard{NN}_of{M}.idx.npy    shape (n_examples_in_shard,) int64 sft_idx

Usage:
  python analysis/trak/featurize_sft.py \
      --shard_idx 0 --num_shards 15 \
      --checkpoint OLMo-3-7B-Instruct-SFT \
      --proj_dim 4096 --max_seq_len 1024

Smoke-test first with --smoke 16 --num_shards 15 --shard_idx 0 (one GPU,
processes 16 examples and prints timing). Verify shapes and that projected
gradients are non-zero before launching the full SLURM array.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_model_paths():
    return json.loads((REPO_ROOT / "model_paths.json").read_text())


def build_chat_text_and_mask(messages, tokenizer, max_seq_len):
    """Apply OLMo-3 ChatML format and produce input_ids + loss_mask.

    Loss mask is 1 for assistant-content tokens, 0 elsewhere. We tokenize
    incrementally so we can mark assistant-content token spans precisely
    without relying on (potentially absent) tokenizer.chat_template.
    """
    input_ids = []
    loss_mask = []
    eos_id = tokenizer.eos_token_id
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Header: "<|im_start|>{role}\n"   (loss=0)
        header = f"<|im_start|>{role}\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        input_ids.extend(header_ids); loss_mask.extend([0] * len(header_ids))
        # Content tokens (loss=1 iff role=='assistant')
        body_ids = tokenizer.encode(content, add_special_tokens=False)
        input_ids.extend(body_ids)
        loss_mask.extend([1 if role == "assistant" else 0] * len(body_ids))
        # Trailer: "<|im_end|>\n"   (loss=1 for assistant — model learns to close;
        # loss=0 for user/system)
        trailer = "<|im_end|>\n"
        trailer_ids = tokenizer.encode(trailer, add_special_tokens=False)
        input_ids.extend(trailer_ids)
        loss_mask.extend([1 if role == "assistant" else 0] * len(trailer_ids))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        loss_mask = loss_mask[:max_seq_len]
    return input_ids, loss_mask


def get_trainable_params(model):
    """Return list of (name, param) for trainable params with grad accumulation."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def build_projector(n_params, proj_dim, seed, device):
    """Use CudaProjector when fast_jl is available; else fall back to
    BasicProjector (pure-PyTorch, slower but no CUDA extension needed)."""
    # fast_jl needs `import torch` to load libc10 first; do this explicitly
    # so the projector ctor can find its CUDA extension.
    import torch  # noqa: F401
    try:
        import fast_jl  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pass
    from trak.projectors import CudaProjector, ProjectionType
    # fast_jl only ships kernels for max_batch_size ∈ {8, 16, 32}.
    proj = CudaProjector(
        grad_dim=n_params, proj_dim=proj_dim, seed=seed,
        proj_type=ProjectionType.rademacher, device=device,
        max_batch_size=8, dtype=torch.float16)
    print(f"Projector: CudaProjector(proj_dim={proj_dim}, "
          f"rademacher, fp16, max_batch_size=8, seed={seed})")
    return proj


def flat_grad_into(params, buf):
    """Copy per-param grads into pre-allocated flat buf (1D, fp16, on GPU).
    Sets each grad to None as it's copied to free memory immediately.
    Returns the buffer view (shape (1, n_params)) ready for projection."""
    offset = 0
    for _, p in params:
        n = p.numel()
        if p.grad is None:
            buf[offset:offset + n].zero_()
        else:
            buf[offset:offset + n].copy_(p.grad.detach().reshape(-1))
            p.grad = None  # free param grad memory immediately
        offset += n
    assert offset == buf.numel(), f"buf size {buf.numel()} != params {offset}"
    return buf.unsqueeze(0)


def flat_grad(params):
    """[Deprecated, kept for compatibility — uses cat and doubles memory.]"""
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
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, default="OLMo-3-7B-Instruct-SFT")
    ap.add_argument("--proj_dim", type=int, default=4096)
    ap.add_argument("--proj_seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--sft_subsample", type=str,
        default="/juice6/scr6/nlp/interp-models/OLMo-3-7B/"
                "trak_v6_0_olmo3_7b_pilot/sft_subsample.parquet")
    ap.add_argument("--out_dir", type=str,
        default="/juice6/scr6/nlp/interp-models/OLMo-3-7B/"
                "trak_v6_0_olmo3_7b_pilot/sft_grads")
    ap.add_argument("--smoke", type=int, default=0,
        help="If >0, process only first N examples of this shard for sanity.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available — required for this job")
    device = torch.device("cuda")

    # Load subsample, take this shard's slice. Accept absolute or repo-relative.
    sub_path = Path(args.sft_subsample)
    if not sub_path.is_absolute():
        sub_path = REPO_ROOT / sub_path
    sub = pd.read_parquet(sub_path)
    sub = sub.sort_values("sft_idx").reset_index(drop=True)
    n_total = len(sub)
    bounds = np.linspace(0, n_total, args.num_shards + 1, dtype=int)
    lo, hi = bounds[args.shard_idx], bounds[args.shard_idx + 1]
    shard = sub.iloc[lo:hi].reset_index(drop=True)
    if args.smoke > 0:
        shard = shard.head(args.smoke).reset_index(drop=True)
    n = len(shard)
    print(f"Shard {args.shard_idx}/{args.num_shards-1}: rows [{lo}, {hi}) "
          f"= {n} examples{' (SMOKE)' if args.smoke else ''}")

    # Load model & tokenizer
    paths = load_model_paths()
    model_path = paths[args.checkpoint]
    print(f"Loading model from {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()  # we want gradient flow but no dropout
    for p in model.parameters():
        p.requires_grad_(True)

    params = get_trainable_params(model)
    n_params = sum(p.numel() for _, p in params)
    print(f"Model: {args.checkpoint}  n_params={n_params:,}")

    projector = build_projector(n_params, args.proj_dim, args.proj_seed, device)

    # Output buffer. Absolute paths bypass REPO_ROOT.
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_buf = np.zeros((n, args.proj_dim), dtype=np.float16)
    idx_buf = shard["sft_idx"].to_numpy()
    n_oom = 0; n_empty_mask = 0

    # Pre-allocate flat grad buffer on GPU (fp16) — reused across examples.
    flat_buf = torch.zeros(n_params, device=device, dtype=torch.float16)

    ce_loss = CrossEntropyLoss(ignore_index=-100, reduction="mean")
    t_start = time.time()

    for i in range(n):
        row = shard.iloc[i]
        messages = list(row["messages"])
        # Convert messages from numpy/list-of-dicts as needed
        messages = [{"role": str(m["role"]), "content": str(m["content"])}
                    for m in messages]
        ids, mask = build_chat_text_and_mask(messages, tokenizer, args.max_seq_len)
        if sum(mask) == 0:
            n_empty_mask += 1
            continue  # no assistant tokens in window → skip

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        # Build labels: shifted-right CE expects labels aligned with input_ids;
        # HuggingFace handles the internal shift. So labels[i] = input_ids[i]
        # at positions where loss_mask=1, else -100.
        labels = input_ids.clone()
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        labels[~mask_t] = -100

        try:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            # Backward
            loss.backward()
            # Flatten into pre-allocated buffer + project
            with torch.no_grad():
                g = flat_grad_into(params, flat_buf)  # (1, n_params) fp16
                proj = projector.project(g, model_id=0)  # (1, proj_dim)
                proj_buf[i] = proj.squeeze(0).detach().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_oom += 1
            print(f"  example {i} ({len(ids)} tok) → OOM, skipping")
            continue

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1:>4}/{n}]  {rate:.2f} ex/s  elapsed={elapsed/60:.1f}m  "
                  f"ETA={eta/60:.1f}m  oom={n_oom}  empty_mask={n_empty_mask}")

    # Save (drop rows that hit OOM or empty mask — they're all-zero)
    np.save(out_dir / f"sft_grads_shard{args.shard_idx:02d}_of{args.num_shards:02d}.npy",
            proj_buf)
    np.save(out_dir / f"sft_grads_shard{args.shard_idx:02d}_of{args.num_shards:02d}.idx.npy",
            idx_buf)
    meta = {
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "n_examples": n, "n_oom": n_oom, "n_empty_mask": n_empty_mask,
        "proj_dim": args.proj_dim, "proj_seed": args.proj_seed,
        "checkpoint": args.checkpoint, "max_seq_len": args.max_seq_len,
        "wallclock_sec": time.time() - t_start,
    }
    (out_dir / f"sft_grads_shard{args.shard_idx:02d}_of{args.num_shards:02d}.meta.json"
     ).write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Saved {n} × {args.proj_dim} fp16 grads to {out_dir}")
    print(f"OOM skipped: {n_oom}, empty-mask skipped: {n_empty_mask}")


if __name__ == "__main__":
    main()
