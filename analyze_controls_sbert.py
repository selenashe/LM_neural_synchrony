"""
Standalone Sentence-BERT control for faster text-only baseline runs.

This reuses the exact episode-level train/test split logic from
`analyze_controls.py`, but replaces hidden states with SBERT embeddings of the
saved turn prompts. Over-length prompts are chunked under the SBERT token limit
and chunk embeddings are mean-pooled into a single turn embedding.
"""

import argparse
import json
import os

import numpy as np
import torch

from analyze_controls import (
    ALL_MISTRAL_PAIRS,
    DEFAULT_PROMPT_SEEDS,
    DEFAULT_SBERT_MODEL,
    SCRIPT_DIR,
    collect_text_turns_from_episodes,
    load_sbert_prompt_features,
    split_episodes,
    train_affine_transformation,
    set_seed,
)
from analyze_alignment import find_best_layer_pair
from experiment_config import EPISODES_NUM, MIA_AVA_RESULTS_POSTFIX, STAGE_A_SEEDS
from utils import get_short_names_and_identifiers


def _apply_results_postfix(model_name: str, postfix: str) -> str:
    if not postfix or model_name.endswith(postfix):
        return model_name
    return model_name + postfix


# Full 3×3 grid (Mistral × Mistral × Meta-Llama), same as run_simulate_and_save_states.sh
ALL_STAGE_A_MODEL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Meta-Llama-3-8B-Instruct_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Meta-Llama-3-8B-Instruct_None_0",
    "Meta-Llama-3-8B-Instruct_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Meta-Llama-3-8B-Instruct_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Meta-Llama-3-8B-Instruct_None_0_Meta-Llama-3-8B-Instruct_None_0",
]

ALL_MISTRAL_PAIRS_MIA_AVA = [
    _apply_results_postfix(m, MIA_AVA_RESULTS_POSTFIX) for m in ALL_MISTRAL_PAIRS
]
ALL_STAGE_A_MODEL_PAIRS_MIA_AVA = [
    _apply_results_postfix(m, MIA_AVA_RESULTS_POSTFIX) for m in ALL_STAGE_A_MODEL_PAIRS
]


def _fit_affine_r2(train_A, train_B, test_A, test_B, seed):
    """Fit one affine map and return held-out R²."""
    if train_A is None or test_A is None or test_A.shape[0] < 2:
        print("  WARNING: insufficient samples for this split, returning R²=NaN")
        return float("nan")

    tA = torch.tensor(train_A).float().to("cuda")
    tB = torch.tensor(train_B).float().to("cuda")
    eA = torch.tensor(test_A).float().to("cuda")
    eB = torch.tensor(test_B).float().to("cuda")

    set_seed(seed)
    r2, _, _, _, _, _, _ = train_affine_transformation(
        train_X=tA,
        train_Y=tB,
        test_X=eA,
        test_Y=eB,
        device="cuda",
        verbose=False,
        seed=seed,
        save_compute=True,
    )
    return max(0.0, float(r2))


def _collect_text_turns_with_offset(allAs, allBs, episode_indices, b_offset=0):
    """Concatenate A_t and B_{t+offset} embeddings across episodes."""
    A_list, B_list = [], []
    for idx in episode_indices:
        ep_A = allAs[idx]
        ep_B = allBs[idx]
        if b_offset < 0:
            raise ValueError("b_offset must be non-negative")
        n = min(ep_A.shape[0], max(0, ep_B.shape[0] - b_offset))
        if n < 1:
            continue
        A_list.append(ep_A[:n])
        B_list.append(ep_B[b_offset : b_offset + n])
    if not A_list:
        return None, None
    return np.concatenate(A_list, axis=0), np.concatenate(B_list, axis=0)


def _shuffle_episode_pairing(episodes, rng):
    """Shuffle B prompts across episodes while preserving within-episode order."""
    perm = rng.permutation(len(episodes))
    shuffled = []
    for src_idx in perm:
        shuffled.append(episodes[src_idx])
    return shuffled


def run_sbert_within_turn_control(all_prompt_As, all_prompt_Bs, seeds=(0, 1, 2)):
    """Current SBERT baseline: aligned prompt rows (offset 0)."""
    real_r2s = []
    all_indices = list(range(len(all_prompt_As)))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)
        train_A, train_B = collect_text_turns_from_episodes(
            all_prompt_As, all_prompt_Bs, train_eps
        )
        test_A, test_B = collect_text_turns_from_episodes(
            all_prompt_As, all_prompt_Bs, test_eps
        )
        r2 = _fit_affine_r2(train_A, train_B, test_A, test_B, seed)
        real_r2s.append(r2)
        print(f"  Seed {seed}: SBERT aligned-text R²={r2:.4f}")

    return {"real_r2_mean": float(np.mean(real_r2s)), "real_r2s": real_r2s}


def run_sbert_shuffled_episode_control(all_prompt_As, all_prompt_Bs, seeds=(0, 1, 2)):
    """Shuffle B prompts across episodes but preserve each episode's turn order."""
    real_r2s = []
    shuffled_r2s = []
    all_indices = list(range(len(all_prompt_As)))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)

        train_A, train_B = collect_text_turns_from_episodes(
            all_prompt_As, all_prompt_Bs, train_eps
        )
        test_A, test_B = collect_text_turns_from_episodes(
            all_prompt_As, all_prompt_Bs, test_eps
        )
        real_r2 = _fit_affine_r2(train_A, train_B, test_A, test_B, seed)
        real_r2s.append(real_r2)

        shuffled_train_B_eps = _shuffle_episode_pairing([all_prompt_Bs[i] for i in train_eps], rng)
        shuffled_test_B_eps = _shuffle_episode_pairing([all_prompt_Bs[i] for i in test_eps], rng)
        train_A_shuf, train_B_shuf = collect_text_turns_from_episodes(
            [all_prompt_As[i] for i in train_eps], shuffled_train_B_eps, range(len(train_eps))
        )
        test_A_shuf, test_B_shuf = collect_text_turns_from_episodes(
            [all_prompt_As[i] for i in test_eps], shuffled_test_B_eps, range(len(test_eps))
        )
        shuffled_r2 = _fit_affine_r2(
            train_A_shuf, train_B_shuf, test_A_shuf, test_B_shuf, seed
        )
        shuffled_r2s.append(shuffled_r2)
        print(
            f"  Seed {seed}: SBERT aligned={real_r2:.4f}, "
            f"episode-shuffled={shuffled_r2:.4f}"
        )

    return {
        "real_r2_mean": float(np.mean(real_r2s)),
        "shuffled_r2_mean": float(np.mean(shuffled_r2s)),
        "real_r2s": real_r2s,
        "shuffled_r2s": shuffled_r2s,
    }


def run_sbert_cross_turn_control(all_prompt_As, all_prompt_Bs, offsets=(0, 1, 5), seeds=(0, 1, 2)):
    """Compare aligned prompt mapping against future-B offsets."""
    results = {}
    all_indices = list(range(len(all_prompt_As)))

    for offset in offsets:
        r2s = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)
            train_A, train_B = _collect_text_turns_with_offset(
                all_prompt_As, all_prompt_Bs, train_eps, b_offset=offset
            )
            test_A, test_B = _collect_text_turns_with_offset(
                all_prompt_As, all_prompt_Bs, test_eps, b_offset=offset
            )
            r2 = _fit_affine_r2(train_A, train_B, test_A, test_B, seed)
            r2s.append(r2)
            print(f"  Seed {seed}: SBERT A_t -> B_t+{offset} R²={r2:.4f}")

        key = f"offset_{offset}"
        results[key] = {
            "b_turn_offset": int(offset),
            "real_r2_mean": float(np.mean(r2s)),
            "real_r2s": r2s,
        }
    return results


def run_sbert_only_model(
    model_name,
    setting_str="A_forward",
    sbert_model_name=DEFAULT_SBERT_MODEL,
    sbert_batch_size=64,
    temp=0.7,
    prompt_seeds=None,
    output_tag="",
):
    """Run SBERT prompt controls plus shuffled and cross-turn baselines."""
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])
    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]
    if prompt_seeds is None:
        prompt_seeds = (
            tuple(STAGE_A_SEEDS) if EPISODES_NUM <= 200 else DEFAULT_PROMPT_SEEDS
        )

    print(f"\n{'#'*60}")
    print(f"SBERT control: {model_name} ({model_tag})")
    print(f"{'#'*60}")

    try:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)
    except FileNotFoundError:
        layer_A, layer_B = None, None

    sbert_As, sbert_Bs, sbert_meta = load_sbert_prompt_features(
        model_name,
        setting,
        seed_list=prompt_seeds,
        temp=temp,
        sbert_model_name=sbert_model_name,
        batch_size=sbert_batch_size,
    )

    aligned = run_sbert_within_turn_control(sbert_As, sbert_Bs)
    aligned["metadata"] = sbert_meta

    results = {
        "model": model_tag,
        "model_full": model_name,
        "setting": setting_str,
        "comparison_layer_A": layer_A,
        "comparison_layer_B": layer_B,
        "sbert": aligned,
        "shuffle_episode_text": run_sbert_shuffled_episode_control(sbert_As, sbert_Bs),
        "cross_turn": run_sbert_cross_turn_control(sbert_As, sbert_Bs),
    }

    out_suffix = f"_{output_tag}" if output_tag else ""
    output_path = os.path.join(
        SCRIPT_DIR, f"control_analysis_sbert_{model_tag}{out_suffix}.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run only the SBERT prompt control baseline"
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--sbert_model", type=str, default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--sbert_batch_size", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument(
        "--all_pairs",
        action="store_true",
        help="Run all 9 Stage-A pairs (Mistral + Meta-Llama). Use --mistral_only_pairs for the 4 Mistral-only pairs.",
    )
    parser.add_argument(
        "--mistral_only_pairs",
        action="store_true",
        help="With --all_pairs, restrict to ALL_MISTRAL_PAIRS (4 combos) instead of all 9.",
    )
    parser.add_argument(
        "--results_postfix",
        type=str,
        default=os.environ.get("SOTOPIA_RESULTS_POSTFIX", MIA_AVA_RESULTS_POSTFIX),
        help="Append to model folder names under sotopia_results/dialogs/ (default: Mia+Ava 90-ep runs).",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default="mia_ava_90ep",
        help="Suffix for output JSON filenames (empty = legacy names).",
    )
    args = parser.parse_args()

    if args.all_pairs:
        pair_list = ALL_MISTRAL_PAIRS if args.mistral_only_pairs else ALL_STAGE_A_MODEL_PAIRS
        all_results = []
        for model in pair_list:
            qualified = _apply_results_postfix(model, args.results_postfix)
            result = run_sbert_only_model(
                qualified,
                setting_str=args.setting,
                sbert_model_name=args.sbert_model,
                sbert_batch_size=args.sbert_batch_size,
                temp=args.temp,
                output_tag=args.output_tag,
            )
            all_results.append(result)

        summary_name = (
            f"control_analysis_sbert_summary_{args.output_tag}.json"
            if args.output_tag
            else "control_analysis_sbert_summary.json"
        )
        output_path = os.path.join(SCRIPT_DIR, summary_name)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved aggregate summary: {output_path}")
        return

    base = args.model or ALL_STAGE_A_MODEL_PAIRS[0]
    model = _apply_results_postfix(base, args.results_postfix)
    run_sbert_only_model(
        model,
        setting_str=args.setting,
        sbert_model_name=args.sbert_model,
        sbert_batch_size=args.sbert_batch_size,
        temp=args.temp,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
