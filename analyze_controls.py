"""
Surface-level controls for affine R².

Control A: Shuffle-paired baseline — break turn-by-turn pairing to test
           whether global R² depends on shared conversational context.
Control D: Layer-0 baseline — compare best-layer R² against layer 0
           (input embeddings) to test whether higher layers add structure
           beyond text-surface similarity.
Control E: Permutation test on cross-transfer gap — randomly reassign
           alignment labels and recompute the transfer gap to test
           whether the observed gap is label-specific.

Usage:
    python analyze_controls.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0
    python analyze_controls.py --all_pairs
    python analyze_controls.py --all_pairs --n_perms 200
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import r2_score as sklearn_r2_score
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import REPO_ROOT, set_seed, get_all_modes, get_short_names_and_identifiers
from affine_transformation import (
    get_preloaded_features,
    train_affine_transformation,
)
from analyze_alignment import find_best_layer_pair, build_flat_episode_map
from analyze_stratified import (
    load_labels,
    collect_turns_from_episodes,
    partition_episodes_by_label,
    partition_by_alignment_score,
    compute_transfer_gap,
    load_features,
    split_episodes,
)

import csv
import glob as globmod
from sentence_transformers import SentenceTransformer
from experiment_config import RESULTS_DIR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "playground", "control_plots")

DEFAULT_PROMPT_SEEDS = (0, 1, 2)
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"

ALL_MISTRAL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]


# ---------------------------------------------------------------------------
# SBERT / text-prompt helpers
# ---------------------------------------------------------------------------

def collect_text_turns_from_episodes(allAs, allBs, episode_indices):
    """Concatenate per-episode 2-D embedding arrays for given episode indices."""
    A_list, B_list = [], []
    for idx in episode_indices:
        ep_A = allAs[idx]
        ep_B = allBs[idx]
        n = min(ep_A.shape[0], ep_B.shape[0])
        if n < 1:
            continue
        A_list.append(ep_A[:n])
        B_list.append(ep_B[:n])
    if not A_list:
        return None, None
    return np.concatenate(A_list, axis=0), np.concatenate(B_list, axis=0)


def _encode_sbert(texts, model, batch_size=64):
    """Encode a list of strings with a SentenceTransformer, chunking long texts."""
    max_tokens = model.max_seq_length or 256
    all_embeddings = []
    for text in texts:
        tokens = model.tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            all_embeddings.append(model.encode([text], batch_size=1)[0])
        else:
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i : i + max_tokens]
                chunk_text = model.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
            chunk_embs = model.encode(chunks, batch_size=batch_size)
            all_embeddings.append(np.mean(chunk_embs, axis=0))
    return np.array(all_embeddings)


def load_sbert_prompt_features(
    model_name, setting, seed_list=(0,), temp=0.7,
    sbert_model_name="all-MiniLM-L6-v2", batch_size=64,
):
    """Load prompt records for all episodes and encode with SBERT.

    Returns:
        all_prompt_As: list of np.ndarray, one per episode (turns x embed_dim)
        all_prompt_Bs: list of np.ndarray
        meta: dict with diagnostics
    """
    sbert = SentenceTransformer(sbert_model_name)
    dialog_dir = os.path.join(RESULTS_DIR, "dialogs", model_name, "prompt_records")

    all_As, all_Bs = [], []
    found_episodes = 0

    for seed in seed_list:
        pattern = os.path.join(dialog_dir, f"*_temp{temp}_seed{seed}.csv")
        files = sorted(globmod.glob(pattern))
        for fpath in files:
            with open(fpath, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [r[0] for r in reader]

            if rows and rows[0] == "save_states":
                rows = rows[1:]

            prompts_A = [rows[i] for i in range(0, len(rows), 2)]
            prompts_B = [rows[i] for i in range(1, len(rows), 2)]
            n = min(len(prompts_A), len(prompts_B))
            if n < 1:
                continue

            emb_A = _encode_sbert(prompts_A[:n], sbert, batch_size)
            emb_B = _encode_sbert(prompts_B[:n], sbert, batch_size)
            all_As.append(emb_A)
            all_Bs.append(emb_B)
            found_episodes += 1

    meta = {
        "sbert_model": sbert_model_name,
        "n_episodes": found_episodes,
        "seeds": list(seed_list),
        "dialog_dir": dialog_dir,
    }
    print(f"  Loaded {found_episodes} episodes, SBERT dim={all_As[0].shape[1] if all_As else '?'}")
    return all_As, all_Bs, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_global_r2(allAs, allBs, layer_A, layer_B, seed=0):
    """Train a global affine map and return (r2, G)."""
    all_indices = list(range(len(allAs)))
    A_all, B_all = collect_turns_from_episodes(allAs, allBs, all_indices, layer_A, layer_B)

    rng = np.random.default_rng(seed)
    train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)
    train_A, train_B = collect_turns_from_episodes(allAs, allBs, train_eps, layer_A, layer_B)
    test_A, test_B = collect_turns_from_episodes(allAs, allBs, test_eps, layer_A, layer_B)

    tA = torch.tensor(train_A).float().to("cuda")
    tB = torch.tensor(train_B).float().to("cuda")
    eA = torch.tensor(test_A).float().to("cuda")
    eB = torch.tensor(test_B).float().to("cuda")

    set_seed(seed)
    r2, _, _, _, _, _, G = train_affine_transformation(
        train_X=tA, train_Y=tB,
        test_X=eA, test_Y=eB,
        device="cuda", verbose=False, seed=seed,
        save_compute=True,
    )
    return float(r2), G


# ---------------------------------------------------------------------------
# Control A: Shuffle-paired baseline
# ---------------------------------------------------------------------------

def run_shuffle_control(allAs, allBs, layer_A, layer_B, seeds=(0, 1, 2)):
    """Break turn-by-turn pairing by shuffling B activations."""
    real_r2s = []
    shuffled_r2s = []

    all_indices = list(range(len(allAs)))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)

        train_A, train_B = collect_turns_from_episodes(allAs, allBs, train_eps, layer_A, layer_B)
        test_A, test_B = collect_turns_from_episodes(allAs, allBs, test_eps, layer_A, layer_B)

        tA = torch.tensor(train_A).float().to("cuda")
        tB = torch.tensor(train_B).float().to("cuda")
        eA = torch.tensor(test_A).float().to("cuda")
        eB = torch.tensor(test_B).float().to("cuda")

        set_seed(seed)
        r2, _, _, _, _, _, _ = train_affine_transformation(
            train_X=tA, train_Y=tB,
            test_X=eA, test_Y=eB,
            device="cuda", verbose=False, seed=seed,
            save_compute=True,
        )
        real_r2s.append(max(0.0, float(r2)))

        perm = rng.permutation(train_B.shape[0])
        shuf_tB = torch.tensor(train_B[perm]).float().to("cuda")
        perm_test = rng.permutation(test_B.shape[0])
        shuf_eB = torch.tensor(test_B[perm_test]).float().to("cuda")

        set_seed(seed)
        shuf_r2, _, _, _, _, _, _ = train_affine_transformation(
            train_X=tA, train_Y=shuf_tB,
            test_X=eA, test_Y=shuf_eB,
            device="cuda", verbose=False, seed=seed,
            save_compute=True,
        )
        shuffled_r2s.append(max(0.0, float(shuf_r2)))
        print(f"  Seed {seed}: real R²={real_r2s[-1]:.4f}, shuffled R²={shuffled_r2s[-1]:.4f}")

    return {
        "real_r2_mean": float(np.mean(real_r2s)),
        "shuffled_r2_mean": float(np.mean(shuffled_r2s)),
        "real_r2s": real_r2s,
        "shuffled_r2s": shuffled_r2s,
    }


# ---------------------------------------------------------------------------
# Control D: Layer-0 baseline
# ---------------------------------------------------------------------------

def run_layer0_control(allAs, allBs, layer_A, layer_B, seeds=(0, 1, 2)):
    """Compare best-layer R² against layer-0 R²."""
    best_r2s = []
    layer0_r2s = []
    all_indices = list(range(len(allAs)))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)

        for la, lb, label in [(layer_A, layer_B, "best"), (0, 0, "layer0")]:
            train_A, train_B = collect_turns_from_episodes(allAs, allBs, train_eps, la, lb)
            test_A, test_B = collect_turns_from_episodes(allAs, allBs, test_eps, la, lb)

            tA = torch.tensor(train_A).float().to("cuda")
            tB = torch.tensor(train_B).float().to("cuda")
            eA = torch.tensor(test_A).float().to("cuda")
            eB = torch.tensor(test_B).float().to("cuda")

            set_seed(seed)
            r2, _, _, _, _, _, _ = train_affine_transformation(
                train_X=tA, train_Y=tB,
                test_X=eA, test_Y=eB,
                device="cuda", verbose=False, seed=seed,
                save_compute=True,
            )
            r2_val = max(0.0, float(r2))
            if label == "best":
                best_r2s.append(r2_val)
            else:
                layer0_r2s.append(r2_val)

        print(f"  Seed {seed}: best-layer R²={best_r2s[-1]:.4f}, layer-0 R²={layer0_r2s[-1]:.4f}")

    return {
        "best_layer_r2_mean": float(np.mean(best_r2s)),
        "layer0_r2_mean": float(np.mean(layer0_r2s)),
        "best_r2s": best_r2s,
        "layer0_r2s": layer0_r2s,
    }


# ---------------------------------------------------------------------------
# Control E: Permutation test on transfer gap
# ---------------------------------------------------------------------------

def run_permutation_test(
    allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
    dimension, n_perms=100, seed=42,
):
    """Randomly reassign labels and recompute transfer gap."""
    if dimension == "alignment_binary":
        real_groups = partition_by_alignment_score(flat_to_combo, labels)
    else:
        real_groups = partition_episodes_by_label(
            flat_to_combo, labels, dimension, len(allAs)
        )

    categories = sorted(real_groups.keys())
    if len(categories) < 2:
        return None

    all_eps = []
    ep_cats = []
    for cat in categories:
        for ep in real_groups[cat]:
            all_eps.append(ep)
            ep_cats.append(cat)

    rng = np.random.default_rng(seed)

    real_matrix = _compute_transfer_matrix(allAs, allBs, real_groups, categories,
                                            layer_A, layer_B, seed=0)
    real_gap = compute_transfer_gap(real_matrix, categories)["gap"]
    print(f"  Real transfer gap: {real_gap:.4f}")

    perm_gaps = []
    for p_idx in range(n_perms):
        shuffled_cats = rng.permutation(ep_cats)
        perm_groups = {cat: [] for cat in categories}
        for ep, cat in zip(all_eps, shuffled_cats):
            perm_groups[cat].append(ep)

        try:
            perm_matrix = _compute_transfer_matrix(allAs, allBs, perm_groups, categories,
                                                    layer_A, layer_B, seed=0)
            perm_gap = compute_transfer_gap(perm_matrix, categories)["gap"]
            perm_gaps.append(perm_gap)
        except Exception as e:
            print(f"    Perm {p_idx} failed: {e}")
            continue

        if (p_idx + 1) % 20 == 0:
            print(f"    Completed {p_idx + 1}/{n_perms} permutations")

    p_value = float(np.mean([g >= real_gap for g in perm_gaps])) if perm_gaps else 1.0
    print(f"  Permutation p-value: {p_value:.4f} ({len(perm_gaps)} permutations)")

    return {
        "real_gap": float(real_gap),
        "perm_gaps": [float(g) for g in perm_gaps],
        "p_value": p_value,
        "n_perms": len(perm_gaps),
    }


def _compute_transfer_matrix(allAs, allBs, groups, categories, layer_A, layer_B, seed=0):
    """Compute a single-seed transfer matrix with episode-level splits."""
    rng = np.random.default_rng(seed)
    matrix = {}

    for train_cat in categories:
        train_eps, held_out_eps = split_episodes(groups[train_cat], train_frac=0.8, rng=rng)

        train_A, train_B = collect_turns_from_episodes(allAs, allBs, train_eps, layer_A, layer_B)
        held_A, held_B = collect_turns_from_episodes(allAs, allBs, held_out_eps, layer_A, layer_B)

        if train_A is None or held_A is None or held_A.shape[0] < 2:
            for tc in categories:
                matrix[(train_cat, tc)] = 0.0
            continue

        tA = torch.tensor(train_A).float().to("cuda")
        tB = torch.tensor(train_B).float().to("cuda")
        hA = torch.tensor(held_A).float().to("cuda")
        hB = torch.tensor(held_B).float().to("cuda")

        set_seed(seed)
        in_r2, _, _, _, _, _, G = train_affine_transformation(
            train_X=tA, train_Y=tB,
            test_X=hA, test_Y=hB,
            device="cuda", verbose=False, seed=seed,
            save_compute=True,
        )
        G.eval()
        matrix[(train_cat, train_cat)] = max(0.0, float(in_r2))

        for test_cat in categories:
            if test_cat == train_cat:
                continue
            test_A, test_B = collect_turns_from_episodes(
                allAs, allBs, groups[test_cat], layer_A, layer_B)
            if test_A is None:
                matrix[(train_cat, test_cat)] = 0.0
                continue
            test_A_t = torch.tensor(test_A).float().to("cuda")
            with torch.no_grad():
                pred = G(test_A_t).cpu().numpy()
            matrix[(train_cat, test_cat)] = max(0.0, float(sklearn_r2_score(test_B, pred)))

    return matrix


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_r2_comparison(results, model_tag, layer_A, layer_B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    labels = ["Best Layer\n(real)", "Best Layer\n(shuffled)", "Layer 0\n(real)"]
    vals = [
        results["shuffle"]["real_r2_mean"],
        results["shuffle"]["shuffled_r2_mean"],
        results["layer0"]["layer0_r2_mean"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(labels)), vals, width=0.5, alpha=0.8)
    bars[1].set_color("tab:red")
    bars[2].set_color("tab:green")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("R²")
    ax.set_title(f"{model_tag} — Control Comparisons (L{layer_A}→L{layer_B})")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"r2_controls_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_permutation_histogram(perm_result, model_tag, dimension, layer_A, layer_B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if perm_result is None:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_result["perm_gaps"], bins=30, alpha=0.7, label="Permuted gaps")
    ax.axvline(perm_result["real_gap"], color="red", linestyle="--", linewidth=2,
               label=f"Real gap={perm_result['real_gap']:.4f}\np={perm_result['p_value']:.4f}")
    ax.set_xlabel("Transfer Gap")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_tag} — Permutation Test ({dimension}, L{layer_A}→L{layer_B})")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"perm_{dimension}_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_single_model(model_name, layer_A=None, layer_B=None,
                     setting_str="A_forward", n_perms=100):
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])
    labels = load_labels()

    if layer_A is None or layer_B is None:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)

    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    print(f"\n{'#'*60}")
    print(f"Controls: {model_name} ({model_tag})")
    print(f"Layer pair: A={layer_A}, B={layer_B}")
    print(f"{'#'*60}")

    allAs, allBs, flat_to_combo = load_features(model_name, setting)

    results = {
        "model": model_tag,
        "model_full": model_name,
        "layer_A": layer_A,
        "layer_B": layer_B,
    }

    print(f"\n--- Control A: Shuffle-Paired Baseline ---")
    results["shuffle"] = run_shuffle_control(allAs, allBs, layer_A, layer_B)

    print(f"\n--- Control D: Layer-0 Baseline ---")
    results["layer0"] = run_layer0_control(allAs, allBs, layer_A, layer_B)

    plot_r2_comparison(results, model_tag, layer_A, layer_B)

    print(f"\n--- Control E: Permutation Tests ---")
    results["permutation"] = {}
    for dim in ["alignment_binary", "goal_structure", "value_compatibility"]:
        print(f"\n  [{dim}]")
        perm_result = run_permutation_test(
            allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
            dimension=dim, n_perms=n_perms,
        )
        results["permutation"][dim] = perm_result
        plot_permutation_histogram(perm_result, model_tag, dim, layer_A, layer_B)

    output_path = os.path.join(
        SCRIPT_DIR, f"control_analysis_{model_tag}_L{layer_A}_L{layer_B}.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Surface-level controls for affine R²")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer_A", type=int, default=None)
    parser.add_argument("--layer_B", type=int, default=None)
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--n_perms", type=int, default=100)
    parser.add_argument("--all_pairs", action="store_true")
    args = parser.parse_args()

    if args.all_pairs:
        all_results = []
        for model in ALL_MISTRAL_PAIRS:
            r = run_single_model(model, args.layer_A, args.layer_B,
                                 args.setting, args.n_perms)
            all_results.append(r)

        agg_path = os.path.join(SCRIPT_DIR, "control_analysis_summary.json")
        with open(agg_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved aggregate summary: {agg_path}")

    elif args.model:
        run_single_model(args.model, args.layer_A, args.layer_B,
                         args.setting, args.n_perms)
    else:
        run_single_model(ALL_MISTRAL_PAIRS[0], args.layer_A, args.layer_B,
                         args.setting, args.n_perms)


if __name__ == "__main__":
    main()
