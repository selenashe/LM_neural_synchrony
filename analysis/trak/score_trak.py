#!/usr/bin/env python3
"""Compute TRAK attribution scores: (n_trials, n_sft) matrix.

Pipeline:
  1. Load and stack the per-shard SFT projected gradients (n_sft, D fp16).
  2. Load and stack the per-shard trial projected gradients (n_trials, D fp16).
  3. Compute Gram matrix  G = P_sft.T @ P_sft + λ·I   (D × D).
     This matches the TRAK paper's formulation (eq 13) and the upstream
     trak/score_computers.py — no 1/n normalization on the Gram.
     With 50k un-normalized SFT-side projected grads, max eigenvalue of
     ΦᵀΦ is ≈ 1.25e10, so meaningful λ values are in the 1e6 – 1e8 range
     (the canonical default below sits at 1e7 ≈ 1e-3 × max_eig, where
     Mahalanobis-like whitening is substantial — Frobenius ratio ≈ 0.69).
  4. Solve G x = P_trial.T  in (D × n_trials) via Cholesky.
  5. score = trial_grads @ G_inv @ P_sft.T.

NOTE: As of the 2026-06 lambda-fix, the canonical analysis runs WITHOUT
L2 row-normalization (`--normalize` defaults off). Normalizing the rows
strips the per-example gradient magnitude that the (ΦᵀΦ)⁻¹ Gram-inverse
is supposed to absorb; combined with the (now-corrected) /n bug, the
old --normalize path was effectively computing TracIn-style cosine
similarity rather than TRAK. The flag is kept for back-comparison only.
Until the group-removal counterfactual validates causality, call the
output "attribution scores," not "influence."

Output:
  attribution_scores_all_lam{N}.npy   shape (n_trials, n_sft) fp32
                                      (~887 MB for 4432×50k)
  attribution_scores_all_lam{N}.meta.json
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
        default=f"{SCRATCH}/trial_grads_all")
    ap.add_argument("--out_dir", type=str,
        default=f"{SCRATCH}/scores")
    ap.add_argument("--meta_dir", type=str,
        default="analysis_outputs/v6_0_olmo3_7b_pilot/trak/scores",
        help="Small meta json + trial_id_order.csv go here (under REPO_ROOT).")
    ap.add_argument("--lam", type=float, default=1e7,
        help="Tikhonov regularizer added to G = ΦᵀΦ + λI. Default 1e7 is "
             "~1e-3 × max eigenvalue of un-normalized ΦᵀΦ (Frobenius "
             "whitening-active ratio ≈ 0.69). Robustness-sweep variants: "
             "1e6 (lighter; ratio 0.29) and 1e8 (heavier; ratio 0.91).")
    ap.add_argument("--num_shards", type=int, default=15,
        help="Number of SFT-side shards to concatenate.")
    ap.add_argument("--num_trial_shards", type=int, default=15,
        help="Number of trial-side shards to concatenate. Default 15 "
             "matches the 15-way SLURM array in "
             "bash/run_trak_featurize_trials_array.sh.")
    ap.add_argument("--normalize", action="store_true",
        help="LEGACY: L2-normalize SFT and trial rows before scoring. "
             "Strips per-example gradient magnitude that the proper Gram "
             "inverse is supposed to absorb — use only for back-comparison "
             "with the pre-2026-06 _norm pipeline. Canonical analysis runs "
             "without this flag.")
    ap.add_argument("--out_name", type=str,
        default="attribution_scores_all_lam1e7.npy",
        help="Filename for the output scores. The default reflects the "
             "canonical λ=1e7 corrected-Gram un-normalized run.")
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

    # Load trial grads — sharded if --num_trial_shards > 1, else single file.
    if args.num_trial_shards > 1:
        print(f"Loading {args.num_trial_shards} trial-grad shards from {tr_dir}")
        tr_parts, tr_id_parts = [], []
        for s in range(args.num_trial_shards):
            gp = tr_dir / (f"trial_grads_shard{s:02d}"
                           f"_of{args.num_trial_shards:02d}.npy")
            ip = tr_dir / (f"trial_grads_shard{s:02d}"
                           f"_of{args.num_trial_shards:02d}.idx.csv")
            if not gp.exists():
                print(f"  WARNING: missing {gp.name}, skipping")
                continue
            tr_parts.append(np.load(gp))
            tr_id_parts.append(pd.read_csv(ip)["trial_id"].tolist())
            print(f"  shard {s}: {tr_parts[-1].shape}")
        P_tr = np.concatenate(tr_parts, axis=0)
        trial_ids = [tid for part in tr_id_parts for tid in part]
    else:
        P_tr = np.load(tr_dir / "trial_grads.npy")        # (n_trials, D)
        trial_ids = pd.read_csv(
            tr_dir / "trial_grads.idx.csv")["trial_id"].tolist()
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
    # Gram in projection space: G = ΦᵀΦ + λI  (D × D)
    # Matches TRAK paper eq (13) and upstream trak/score_computers.py.
    # The earlier code divided by n_sft, which combined with a typical λ
    # made λI swamp the data term and reduced the estimator to TracIn-style
    # cosine similarity. See v6_0_trak_lambda_fix.md for the diagnostic.
    G = P_sft_t.T @ P_sft_t
    G = G + args.lam * torch.eye(D, device=device, dtype=torch.float32)
    print(f"  Gram: {G.shape}  λ={args.lam:.3g}  "
          f"({time.time() - t:.1f}s)")

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
        "num_trial_shards": args.num_trial_shards,
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
