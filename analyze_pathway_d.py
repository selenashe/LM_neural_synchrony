"""
Pathway D: Alignment-conditioned affine maps.

Train a separate affine map for each alignment category and compare:
  1. R² achieved by each category-specific map
  2. Similarity of weight matrices across categories (cosine similarity,
     Frobenius distance)

Train/test split is at the EPISODE level within each category.

Usage:
    python analyze_pathway_d.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0
    python analyze_pathway_d.py --all_pairs
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
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import REPO_ROOT, set_seed, get_all_modes, get_short_names_and_identifiers
from affine_transformation import get_preloaded_features, train_affine_transformation
from analyze_alignment import find_best_layer_pair, build_flat_episode_map
from analyze_stratified import (
    load_labels,
    collect_turns_from_episodes,
    partition_episodes_by_label,
    partition_by_alignment_score,
    load_features,
    split_episodes,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "playground", "pathway_d_plots")

ALL_MISTRAL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]


# ---------------------------------------------------------------------------
# Core: Train per-category maps and extract weights
# ---------------------------------------------------------------------------

def extract_weights(G):
    """Extract weight matrix and bias from a LinearProjection model."""
    W = G.linear.weight.detach().cpu().numpy()
    b = G.linear.bias.detach().cpu().numpy()
    return W, b


def train_per_category_maps(
    allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
    dimension, seeds=(0, 1, 2),
):
    """
    Train a separate ridge regression for each alignment category.
    Train/test split is at the EPISODE level within each category.

    Returns:
        cat_results: dict of category -> {r2, weights, bias, n_turns, ...}
    """
    if dimension == "alignment_binary":
        groups = partition_by_alignment_score(flat_to_combo, labels)
    else:
        groups = partition_episodes_by_label(
            flat_to_combo, labels, dimension, len(allAs)
        )

    categories = sorted(groups.keys())
    if len(categories) < 2:
        print(f"  < 2 categories for '{dimension}', skipping.")
        return None, categories

    for cat in categories:
        n_eps = len(groups[cat])
        A_check, _ = collect_turns_from_episodes(allAs, allBs, groups[cat], layer_A, layer_B)
        n_turns = A_check.shape[0] if A_check is not None else 0
        if n_eps < 5 or n_turns < 20:
            print(f"  Too few data for '{cat}' ({n_eps} eps, {n_turns} turns), skipping.")
            return None, categories
        print(f"    {cat}: {n_eps} episodes, {n_turns} turns")

    cat_results = {}

    for cat in categories:
        r2_per_seed = []
        weights_per_seed = []
        biases_per_seed = []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            train_eps, test_eps = split_episodes(groups[cat], train_frac=0.8, rng=rng)

            train_A, train_B = collect_turns_from_episodes(
                allAs, allBs, train_eps, layer_A, layer_B)
            test_A, test_B = collect_turns_from_episodes(
                allAs, allBs, test_eps, layer_A, layer_B)

            if train_A is None or test_A is None or test_A.shape[0] < 2:
                continue

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
            r2_per_seed.append(max(0.0, float(r2)))
            W, b = extract_weights(G)
            weights_per_seed.append(W)
            biases_per_seed.append(b)

        if not r2_per_seed:
            print(f"    {cat}: no valid seeds, skipping.")
            return None, categories

        mean_W = np.mean(weights_per_seed, axis=0)
        mean_b = np.mean(biases_per_seed, axis=0)

        A_all, _ = collect_turns_from_episodes(allAs, allBs, groups[cat], layer_A, layer_B)
        cat_results[cat] = {
            "r2_mean": float(np.mean(r2_per_seed)),
            "r2_std": float(np.std(r2_per_seed)),
            "r2_per_seed": r2_per_seed,
            "n_episodes": len(groups[cat]),
            "n_turns": A_all.shape[0],
            "mean_weight": mean_W,
            "mean_bias": mean_b,
        }
        print(f"    {cat}: R²={np.mean(r2_per_seed):.4f} ± {np.std(r2_per_seed):.4f}")

    return cat_results, categories


# ---------------------------------------------------------------------------
# Weight matrix similarity analysis
# ---------------------------------------------------------------------------

def compute_weight_similarity(cat_results, categories):
    """
    Compute pairwise cosine similarity and Frobenius distance
    between the mean weight matrices of each category.
    """
    sim_matrix = {}

    for cat_a, cat_b in combinations(categories, 2):
        W_a = cat_results[cat_a]["mean_weight"]
        W_b = cat_results[cat_b]["mean_weight"]

        W_a_flat = W_a.flatten()
        W_b_flat = W_b.flatten()

        cos_sim = float(np.dot(W_a_flat, W_b_flat) /
                        (np.linalg.norm(W_a_flat) * np.linalg.norm(W_b_flat) + 1e-12))

        frob_dist = float(np.linalg.norm(W_a - W_b, 'fro'))
        frob_norm_a = float(np.linalg.norm(W_a, 'fro'))
        frob_norm_b = float(np.linalg.norm(W_b, 'fro'))
        relative_frob = frob_dist / ((frob_norm_a + frob_norm_b) / 2 + 1e-12)

        b_a = cat_results[cat_a]["mean_bias"]
        b_b = cat_results[cat_b]["mean_bias"]
        bias_cos = float(np.dot(b_a, b_b) /
                         (np.linalg.norm(b_a) * np.linalg.norm(b_b) + 1e-12))
        bias_l2 = float(np.linalg.norm(b_a - b_b))

        sim_matrix[(cat_a, cat_b)] = {
            "cosine_similarity": cos_sim,
            "frobenius_distance": frob_dist,
            "relative_frobenius": relative_frob,
            "bias_cosine": bias_cos,
            "bias_l2_distance": bias_l2,
        }
        print(f"    {cat_a} vs {cat_b}: cos={cos_sim:.4f}, "
              f"rel_frob={relative_frob:.4f}, bias_cos={bias_cos:.4f}")

    return sim_matrix


def compute_global_map_similarity(allAs, allBs, layer_A, layer_B, cat_results, categories, seeds=(0, 1, 2)):
    """
    Train a global (all-data) affine map with episode-level splits,
    then compare each category-specific map to it.
    """
    all_indices = list(range(len(allAs)))

    global_weights = []
    global_biases = []
    global_r2s = []

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
        r2, _, _, _, _, _, G = train_affine_transformation(
            train_X=tA, train_Y=tB,
            test_X=eA, test_Y=eB,
            device="cuda", verbose=False, seed=seed,
            save_compute=True,
        )
        global_r2s.append(max(0.0, float(r2)))
        W, b = extract_weights(G)
        global_weights.append(W)
        global_biases.append(b)

    mean_global_W = np.mean(global_weights, axis=0)
    mean_global_b = np.mean(global_biases, axis=0)
    global_W_flat = mean_global_W.flatten()

    cat_vs_global = {}
    for cat in categories:
        cat_W = cat_results[cat]["mean_weight"]
        cat_W_flat = cat_W.flatten()

        cos_sim = float(np.dot(cat_W_flat, global_W_flat) /
                        (np.linalg.norm(cat_W_flat) * np.linalg.norm(global_W_flat) + 1e-12))
        frob_dist = float(np.linalg.norm(cat_W - mean_global_W, 'fro'))
        frob_norm_global = float(np.linalg.norm(mean_global_W, 'fro'))
        relative_frob = frob_dist / (frob_norm_global + 1e-12)

        cat_vs_global[cat] = {
            "cosine_similarity": cos_sim,
            "frobenius_distance": frob_dist,
            "relative_frobenius": relative_frob,
        }
        print(f"    {cat} vs global: cos={cos_sim:.4f}, rel_frob={relative_frob:.4f}")

    return {
        "global_r2_mean": float(np.mean(global_r2s)),
        "cat_vs_global": cat_vs_global,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_r2_by_category(cat_results, categories, model_tag, dimension, layer_A, layer_B, global_r2=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    means = [cat_results[c]["r2_mean"] for c in categories]
    stds = [cat_results[c]["r2_std"] for c in categories]
    n_info = [f"{c}\n({cat_results[c]['n_episodes']} eps, {cat_results[c]['n_turns']} turns)"
              for c in categories]

    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.5), 5))
    x = range(len(categories))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, width=0.5)
    if global_r2 is not None:
        ax.axhline(global_r2, color="red", linestyle="--", linewidth=1.5, label=f"Global R²={global_r2:.3f}")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(n_info, rotation=30, ha="right")
    ax.set_ylabel("In-Distribution R²")
    ax.set_title(f"{model_tag} — Per-Category R² ({dimension}, L{layer_A}→L{layer_B})")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"r2_per_cat_{dimension}_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_similarity_heatmap(sim_matrix, categories, model_tag, dimension, layer_A, layer_B, metric="cosine_similarity"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = len(categories)
    mat = np.ones((n, n))

    for (ca, cb), sims in sim_matrix.items():
        i, j = categories.index(ca), categories.index(cb)
        mat[i, j] = sims[metric]
        mat[j, i] = sims[metric]

    metric_label = metric.replace("_", " ").title()
    cmap = "RdYlGn" if "cosine" in metric else "YlOrRd_r"

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    ax.set_title(f"{model_tag} — Weight {metric_label} ({dimension}, L{layer_A}→L{layer_B})")

    fmt = ".4f" if "cosine" in metric else ".2f"
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if (("cosine" in metric and val < 0.5) or
                                ("cosine" not in metric and val > mat.max() * 0.6)) else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label=metric_label)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"sim_{metric}_{dimension}_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cat_vs_global(cat_vs_global, categories, model_tag, dimension, layer_A, layer_B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    cos_vals = [cat_vs_global[c]["cosine_similarity"] for c in categories]
    frob_vals = [cat_vs_global[c]["relative_frobenius"] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(categories))
    ax1.bar(x, cos_vals, alpha=0.8, width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=30, ha="right")
    ax1.set_ylabel("Cosine Similarity to Global Map")
    ax1.set_title(f"Weight Similarity to Global")
    ax1.set_ylim(0, 1.05)

    ax2.bar(x, frob_vals, alpha=0.8, width=0.5, color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=30, ha="right")
    ax2.set_ylabel("Relative Frobenius Distance")
    ax2.set_title(f"Weight Divergence from Global")

    fig.suptitle(f"{model_tag} — Category vs Global Map ({dimension}, L{layer_A}→L{layer_B})")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"cat_vs_global_{dimension}_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_dimension(allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
                  dimension, model_tag, seeds=(0, 1, 2)):
    print(f"\n{'='*60}")
    print(f"Pathway D: {dimension}")
    print(f"{'='*60}")

    cat_results, categories = train_per_category_maps(
        allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
        dimension=dimension, seeds=seeds,
    )
    if cat_results is None:
        return None

    print(f"\n  --- Pairwise weight similarity ---")
    sim_matrix = compute_weight_similarity(cat_results, categories)

    print(f"\n  --- Category vs global map ---")
    global_info = compute_global_map_similarity(
        allAs, allBs, layer_A, layer_B, cat_results, categories, seeds=seeds,
    )

    plot_r2_by_category(cat_results, categories, model_tag, dimension,
                        layer_A, layer_B, global_r2=global_info["global_r2_mean"])
    plot_similarity_heatmap(sim_matrix, categories, model_tag, dimension,
                            layer_A, layer_B, metric="cosine_similarity")
    plot_similarity_heatmap(sim_matrix, categories, model_tag, dimension,
                            layer_A, layer_B, metric="relative_frobenius")
    plot_cat_vs_global(global_info["cat_vs_global"], categories, model_tag,
                       dimension, layer_A, layer_B)

    serializable_cat = {}
    for cat in categories:
        r = cat_results[cat]
        serializable_cat[cat] = {
            "r2_mean": r["r2_mean"],
            "r2_std": r["r2_std"],
            "r2_per_seed": r["r2_per_seed"],
            "n_episodes": r["n_episodes"],
            "n_turns": r["n_turns"],
        }

    serializable_sim = {}
    for (ca, cb), sims in sim_matrix.items():
        serializable_sim[f"{ca}_vs_{cb}"] = sims

    return {
        "categories": categories,
        "per_category_r2": serializable_cat,
        "pairwise_similarity": serializable_sim,
        "global_r2": global_info["global_r2_mean"],
        "cat_vs_global": global_info["cat_vs_global"],
    }


def run_single_model(model_name, layer_A=None, layer_B=None, setting_str="A_forward"):
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])
    labels = load_labels()

    if layer_A is None or layer_B is None:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)

    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    print(f"\n{'#'*60}")
    print(f"Model: {model_name} ({model_tag})")
    print(f"Layer pair: A={layer_A}, B={layer_B}")
    print(f"{'#'*60}")

    allAs, allBs, flat_to_combo = load_features(model_name, setting)

    summary = {
        "model": model_tag,
        "model_full": model_name,
        "layer_A": layer_A,
        "layer_B": layer_B,
    }

    for dim in ["alignment_binary", "goal_structure", "value_compatibility"]:
        result = run_dimension(
            allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
            dimension=dim, model_tag=model_tag, seeds=(0, 1, 2),
        )
        if result is not None:
            summary[dim] = result

    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_tag}")
    print(f"{'='*60}")
    for dim in ["alignment_binary", "goal_structure", "value_compatibility"]:
        if dim not in summary:
            continue
        info = summary[dim]
        print(f"\n  [{dim}]  global R²={info['global_r2']:.4f}")
        for cat in info["categories"]:
            cr = info["per_category_r2"][cat]
            print(f"    {cat:25s}: R²={cr['r2_mean']:.4f} ± {cr['r2_std']:.4f}  "
                  f"({cr['n_episodes']} eps, {cr['n_turns']} turns)")
        print(f"    Pairwise weight cosine similarities:")
        for pair_key, sims in info["pairwise_similarity"].items():
            print(f"      {pair_key:40s}: cos={sims['cosine_similarity']:.4f}  "
                  f"rel_frob={sims['relative_frobenius']:.4f}")

    output_path = os.path.join(
        SCRIPT_DIR, f"pathway_d_{model_tag}_L{layer_A}_L{layer_B}.json"
    )
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pathway D: per-category affine maps")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer_A", type=int, default=None)
    parser.add_argument("--layer_B", type=int, default=None)
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--all_pairs", action="store_true")
    args = parser.parse_args()

    if args.all_pairs:
        all_summaries = []
        for model in ALL_MISTRAL_PAIRS:
            s = run_single_model(model, args.layer_A, args.layer_B, args.setting)
            all_summaries.append(s)

        print(f"\n\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        for s in all_summaries:
            tag = s["model"]
            print(f"\n  {tag}:")
            for dim in ["alignment_binary", "goal_structure", "value_compatibility"]:
                if dim not in s:
                    continue
                info = s[dim]
                cats = info["categories"]
                r2s = [info["per_category_r2"][c]["r2_mean"] for c in cats]
                cos_vals = [v["cosine_similarity"] for v in info["pairwise_similarity"].values()]
                print(f"    [{dim}] R²=[{', '.join(f'{r:.3f}' for r in r2s)}]  "
                      f"mean_cos={np.mean(cos_vals):.4f}  "
                      f"min_cos={min(cos_vals):.4f}")

        agg_path = os.path.join(SCRIPT_DIR, "pathway_d_summary.json")
        with open(agg_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nSaved aggregate summary: {agg_path}")

    elif args.model:
        run_single_model(args.model, args.layer_A, args.layer_B, args.setting)
    else:
        run_single_model(ALL_MISTRAL_PAIRS[0], args.layer_A, args.layer_B, args.setting)


if __name__ == "__main__":
    main()
