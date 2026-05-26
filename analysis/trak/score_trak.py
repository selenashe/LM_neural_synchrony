#!/usr/bin/env python3
"""Compute TRAK attribution scores: (n_trials, n_sft) matrix.

Pipeline:
  1. Load and stack the per-shard SFT projected gradients (n_sft, D fp16).
  2. Load trial projected gradients (n_trials, D fp16).
  3. Compute Gram matrix G = P_sft.T @ P_sft / n_sft  (D × D)  + λ·I.
  4. Solve P_sft.T @ x = trial_grads^T  in (D × n_trials) via Cholesky.
  5. score = trial_grads @ G_inv @ P_sft.T.

Output:
  attribution_scores.npy   shape (n_trials, n_sft) fp32 (~600 MB for 300×50k)
  attribution_meta.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    SCRATCH = "/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot"
    ap.add_argument("--sft_grads_dir", type=str,
        default=f"{SCRATCH}/sft_grads")
    ap.add_argument("--trial_grads_dir", type=str,
        default=f"{SCRATCH}/trial_grads")
    ap.add_argument("--out_dir", type=str,
        default=f"{SCRATCH}/scores")
    ap.add_argument("--meta_dir", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak/scores",
        help="Small meta json + trial_id_order.csv go here (under REPO_ROOT).")
    ap.add_argument("--lam", type=float, default=1e-2,
        help="Tikhonov regularizer for the Gram matrix")
    ap.add_argument("--num_shards", type=int, default=15)
    ap.add_argument("--normalize", action="store_true",
        help="L2-normalize SFT (and trial) projected gradients before scoring. "
             "Removes per-example gradient-norm confound; effectively turns "
             "the inner product into a cosine similarity in projection space.")
    ap.add_argument("--out_name", type=str, default="attribution_scores.npy",
        help="Filename for the output scores. Use a distinct name when "
             "--normalize is set so the un-normalized run isn't overwritten.")
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = REPO_ROOT / args.meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = resolve(args.sft_grads_dir)
    tr_dir = resolve(args.trial_grads_dir)

    # Load + concat SFT grads
    print(f"Loading SFT grads from {sft_dir}")
    parts, parts_idx = [], []
    for s in range(args.num_shards):
        gp = sft_dir / f"sft_grads_shard{s:02d}_of{args.num_shards:02d}.npy"
        ip = sft_dir / f"sft_grads_shard{s:02d}_of{args.num_shards:02d}.idx.npy"
        if not gp.exists():
            print(f"  WARNING: missing {gp.name}, skipping")
            continue
        parts.append(np.load(gp))
        parts_idx.append(np.load(ip))
        print(f"  shard {s}: {parts[-1].shape}")
    P_sft = np.concatenate(parts, axis=0)         # (n_sft, D)
    sft_idx = np.concatenate(parts_idx, axis=0)   # (n_sft,)
    n_sft, D = P_sft.shape
    print(f"Stacked SFT grads: {P_sft.shape}")

    # Load trial grads
    P_tr = np.load(tr_dir / "trial_grads.npy")            # (n_trials, D)
    trial_ids = pd.read_csv(tr_dir / "trial_grads.idx.csv")["trial_id"].tolist()
    n_tr = P_tr.shape[0]
    assert P_tr.shape[1] == D, f"proj_dim mismatch: trial={P_tr.shape[1]} sft={D}"
    print(f"Trial grads: {P_tr.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing on {device}")

    P_sft_t = torch.from_numpy(P_sft).to(device, dtype=torch.float32)  # (n_sft, D)
    P_tr_t = torch.from_numpy(P_tr).to(device, dtype=torch.float32)    # (n_tr,  D)

    if args.normalize:
        sft_norms = P_sft_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        tr_norms = P_tr_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        print(f"  normalizing: sft norm mean={sft_norms.mean():.3g} "
              f"min={sft_norms.min():.3g} max={sft_norms.max():.3g}")
        print(f"  normalizing: trial norm mean={tr_norms.mean():.3g} "
              f"min={tr_norms.min():.3g} max={tr_norms.max():.3g}")
        P_sft_t = P_sft_t / sft_norms
        P_tr_t = P_tr_t / tr_norms

    t = time.time()
    # Gram in projection space: G = P_sft^T P_sft / n_sft + λI  (D × D)
    G = (P_sft_t.T @ P_sft_t) / n_sft
    G = G + args.lam * torch.eye(D, device=device, dtype=torch.float32)
    print(f"  Gram: {G.shape}  ({time.time() - t:.1f}s)")

    # Solve G x = P_tr^T  → x = G^{-1} P_tr^T (D × n_tr)
    t = time.time()
    L = torch.linalg.cholesky(G)
    x = torch.cholesky_solve(P_tr_t.T, L)  # (D, n_tr)
    print(f"  Cholesky solve: {x.shape}  ({time.time() - t:.1f}s)")

    # Scores: (n_tr, D) @ (D, n_sft) — but actually canonical TRAK formula is
    #   S = P_tr @ G^{-1} @ P_sft^T  = (G^{-1} P_tr^T)^T @ P_sft^T
    t = time.time()
    scores = (x.T @ P_sft_t.T).cpu().numpy()  # (n_tr, n_sft) fp32
    print(f"  Scores: {scores.shape}  ({time.time() - t:.1f}s)")

    scores_path = out_dir / args.out_name
    np.save(scores_path, scores)
    np.save(out_dir / "sft_idx_order.npy", sft_idx)
    pd.DataFrame({"trial_id": trial_ids}).to_csv(
        meta_dir / "trial_id_order.csv", index=False)
    meta_stem = Path(args.out_name).stem  # e.g. attribution_scores or attribution_scores_norm
    meta = {
        "n_trials": n_tr, "n_sft": n_sft, "proj_dim": D,
        "lambda": args.lam, "num_shards": args.num_shards,
        "normalize": bool(args.normalize),
        "scores_dtype": "float32",
        "scores_path": str(scores_path),
        "sft_idx_order_path": str(out_dir / "sft_idx_order.npy"),
    }
    (meta_dir / f"{meta_stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {args.out_name} ({scores.nbytes/1e6:.1f} MB) to {out_dir}")
    print(f"Wrote meta + trial_id_order.csv to {meta_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
