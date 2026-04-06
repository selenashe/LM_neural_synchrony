"""
Alignment-stratified affine analysis.

Pathway A: Train affine map on one alignment subset, test on another.
           Produces a cross-transfer R² matrix.
           Train/test split is at the EPISODE level within each category.
Pathway C: Residual analysis of a globally-trained affine map,
           stratified by alignment label.
           Train/test split is at the EPISODE level.

Usage:
    python analyze_stratified.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0
    python analyze_stratified.py --all_pairs
    python analyze_stratified.py --all_pairs --pathway C
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(SCRIPT_DIR, "goal_alignment_labels.json")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "playground", "stratified_plots")

ALL_MISTRAL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def partition_episodes_by_label(flat_to_combo, labels, dimension, num_episodes):
    """
    Partition flat episode indices into groups based on an alignment dimension.

    Returns dict: category_name -> list of flat episode indices
    """
    combo_to_flat = defaultdict(list)
    for flat_idx, combo_idx in enumerate(flat_to_combo):
        combo_to_flat[combo_idx].append(flat_idx)

    groups = defaultdict(list)
    skipped = 0
    for combo_idx, flat_indices in combo_to_flat.items():
        label = labels.get(str(combo_idx))
        if label is None or label.get(dimension) is None:
            skipped += 1
            continue
        cat = label[dimension]
        groups[cat].extend(flat_indices)

    print(f"  Partitioned by '{dimension}': "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items()))
          + f"  (skipped {skipped} unlabeled)")
    return dict(groups)


def partition_by_alignment_score(flat_to_combo, labels, high_thresh=0.5, low_thresh=-0.5):
    """Split into high-alignment vs low-alignment episodes by score threshold.
    Scores are on a [-1, 1] scale."""
    combo_to_flat = defaultdict(list)
    for flat_idx, combo_idx in enumerate(flat_to_combo):
        combo_to_flat[combo_idx].append(flat_idx)

    groups = {"high_align": [], "low_align": []}
    for combo_idx, flat_indices in combo_to_flat.items():
        label = labels.get(str(combo_idx))
        if label is None or label.get("alignment_score") is None:
            continue
        score = label["alignment_score"]
        if score >= high_thresh:
            groups["high_align"].extend(flat_indices)
        elif score <= low_thresh:
            groups["low_align"].extend(flat_indices)

    print(f"  Binary alignment split: high(>={high_thresh})={len(groups['high_align'])}, "
          f"low(<={low_thresh})={len(groups['low_align'])}")
    return groups


def collect_turns_from_episodes(allAs, allBs, episode_indices, layer_A, layer_B):
    """Concatenate turn-level features from a set of episodes at specified layers."""
    A_list, B_list = [], []
    for idx in episode_indices:
        ep_A = allAs[idx]
        ep_B = allBs[idx]
        n = min(ep_A.shape[0], ep_B.shape[0])
        if n < 1:
            continue
        A_list.append(ep_A[:n, layer_A, :])
        B_list.append(ep_B[:n, layer_B, :])
    if not A_list:
        return None, None
    return np.concatenate(A_list, axis=0), np.concatenate(B_list, axis=0)


def split_episodes(episode_indices, train_frac=0.8, rng=None):
    """Split a list of episode indices into train/test at the episode level."""
    if rng is None:
        rng = np.random.default_rng()
    indices = np.array(episode_indices)
    perm = rng.permutation(len(indices))
    split = int(train_frac * len(indices))
    return indices[perm[:split]].tolist(), indices[perm[split:]].tolist()


# ---------------------------------------------------------------------------
# Pathway A: Cross-transfer R² matrix
# ---------------------------------------------------------------------------

def run_cross_transfer(
    allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
    dimension, seeds=(0, 1, 2), alpha=0.1,
):
    """
    For each pair of categories (train_cat, test_cat), train ridge on
    train_cat episodes and evaluate R² on test_cat episodes.

    Train/test split is at the EPISODE level: 80% of episodes for training,
    20% held out for in-distribution evaluation.

    Returns:
        transfer_matrix: dict of {(train_cat, test_cat): mean_r2}
        per_seed_matrices: list of per-seed matrices
    """
    if dimension == "alignment_binary":
        groups = partition_by_alignment_score(flat_to_combo, labels)
    else:
        groups = partition_episodes_by_label(
            flat_to_combo, labels, dimension, len(allAs)
        )

    categories = sorted(groups.keys())
    if len(categories) < 2:
        print(f"  Only {len(categories)} category found for '{dimension}', skipping.")
        return None, None, categories

    for cat in categories:
        n_eps = len(groups[cat])
        if n_eps < 5:
            print(f"  Too few episodes for '{cat}' ({n_eps}), skipping dimension.")
            return None, None, categories
        A_check, _ = collect_turns_from_episodes(allAs, allBs, groups[cat], layer_A, layer_B)
        print(f"    {cat}: {n_eps} episodes, {A_check.shape[0] if A_check is not None else 0} turns")

    per_seed_matrices = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        rng = np.random.default_rng(seed)
        matrix = {}

        for train_cat in categories:
            train_eps, held_out_eps = split_episodes(groups[train_cat], train_frac=0.8, rng=rng)

            train_A_np, train_B_np = collect_turns_from_episodes(
                allAs, allBs, train_eps, layer_A, layer_B)
            held_A_np, held_B_np = collect_turns_from_episodes(
                allAs, allBs, held_out_eps, layer_A, layer_B)

            if train_A_np is None or held_A_np is None or held_A_np.shape[0] < 2:
                print(f"    Skipping '{train_cat}': insufficient data after episode split")
                return None, None, categories

            tA = torch.tensor(train_A_np).float().to("cuda")
            tB = torch.tensor(train_B_np).float().to("cuda")
            held_A = torch.tensor(held_A_np).float().to("cuda")
            held_B = torch.tensor(held_B_np).float().to("cuda")

            set_seed(seed)
            in_dist_r2, _, _, _, _, _, G = train_affine_transformation(
                train_X=tA, train_Y=tB,
                test_X=held_A, test_Y=held_B,
                device="cuda", verbose=False, seed=seed,
                save_compute=True,
            )
            G.eval()
            matrix[(train_cat, train_cat)] = max(0.0, float(in_dist_r2))

            for test_cat in categories:
                if test_cat == train_cat:
                    continue
                test_A_np, test_B_np = collect_turns_from_episodes(
                    allAs, allBs, groups[test_cat], layer_A, layer_B)
                test_A_t = torch.tensor(test_A_np).float().to("cuda")

                with torch.no_grad():
                    pred = G(test_A_t).cpu().numpy()
                cross_r2 = max(0.0, float(sklearn_r2_score(test_B_np, pred)))
                matrix[(train_cat, test_cat)] = cross_r2

            print(f"    Trained on '{train_cat}' ({len(train_eps)} eps train, "
                  f"{len(held_out_eps)} eps held-out): "
                  + ", ".join(f"test '{tc}'={matrix[(train_cat, tc)]:.4f}" for tc in categories))

        per_seed_matrices.append(matrix)

    mean_matrix = {}
    for key in per_seed_matrices[0]:
        vals = [m[key] for m in per_seed_matrices]
        mean_matrix[key] = float(np.mean(vals))

    return mean_matrix, per_seed_matrices, categories


def plot_transfer_matrix(mean_matrix, categories, model_tag, dimension, layer_A, layer_B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = len(categories)
    mat = np.zeros((n, n))
    for i, tc in enumerate(categories):
        for j, ec in enumerate(categories):
            mat[i, j] = mean_matrix.get((tc, ec), 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)
    ax.set_xlabel("Test Category")
    ax.set_ylabel("Train Category")
    ax.set_title(f"{model_tag} — Cross-Transfer R² ({dimension}, L{layer_A}→L{layer_B})")

    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10, color=color)

    fig.colorbar(im, ax=ax, label="R²")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"transfer_{dimension}_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def compute_transfer_gap(mean_matrix, categories):
    """Compute diagonal vs off-diagonal gap as a summary statistic."""
    diag_vals, off_diag_vals = [], []
    for tc in categories:
        for ec in categories:
            val = mean_matrix.get((tc, ec), 0.0)
            if tc == ec:
                diag_vals.append(val)
            else:
                off_diag_vals.append(val)
    diag_mean = np.mean(diag_vals) if diag_vals else 0.0
    off_diag_mean = np.mean(off_diag_vals) if off_diag_vals else 0.0
    gap = diag_mean - off_diag_mean
    return {
        "diagonal_mean": float(diag_mean),
        "off_diagonal_mean": float(off_diag_mean),
        "gap": float(gap),
        "gap_ratio": float(gap / diag_mean) if diag_mean > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Pathway C: Residual analysis
# ---------------------------------------------------------------------------

def run_residual_analysis(
    allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
    seeds=(0, 1, 2), alpha=0.1,
):
    """
    Train a global affine map on 80% of EPISODES, then compute per-episode
    residual norms on the held-out 20% and stratify by alignment labels.
    """
    num_episodes = len(allAs)
    all_episode_indices = list(range(num_episodes))

    per_episode_residuals = defaultdict(list)

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        rng = np.random.default_rng(seed)

        train_eps, test_eps = split_episodes(all_episode_indices, train_frac=0.8, rng=rng)

        train_A_np, train_B_np = collect_turns_from_episodes(
            allAs, allBs, train_eps, layer_A, layer_B)
        test_A_np, test_B_np = collect_turns_from_episodes(
            allAs, allBs, test_eps, layer_A, layer_B)

        tA = torch.tensor(train_A_np).float().to("cuda")
        tB = torch.tensor(train_B_np).float().to("cuda")
        eA = torch.tensor(test_A_np).float().to("cuda")
        eB = torch.tensor(test_B_np).float().to("cuda")

        set_seed(seed)
        global_r2, _, _, _, _, _, G = train_affine_transformation(
            train_X=tA, train_Y=tB,
            test_X=eA, test_Y=eB,
            device="cuda", verbose=False, seed=seed,
            save_compute=True,
        )
        print(f"    Global R²: {global_r2:.4f}")

        G.eval()
        for ep_idx in test_eps:
            ep_A = allAs[ep_idx]
            ep_B = allBs[ep_idx]
            n_turns = min(ep_A.shape[0], ep_B.shape[0])
            if n_turns < 2:
                continue

            ep_A_layer = ep_A[:n_turns, layer_A, :]
            ep_B_layer = ep_B[:n_turns, layer_B, :]

            ep_A_t = torch.tensor(ep_A_layer).float().to("cuda")
            with torch.no_grad():
                pred_B = G(ep_A_t).cpu().numpy()

            residuals = ep_B_layer - pred_B
            mean_l2 = float(np.mean(np.linalg.norm(residuals, axis=1)))
            combo_idx = flat_to_combo[ep_idx]
            per_episode_residuals[combo_idx].append((seed, mean_l2))

    return dict(per_episode_residuals)


def analyze_residuals_by_label(per_episode_residuals, labels, model_tag, layer_A, layer_B):
    """Stratify residual norms by alignment dimensions and test for differences."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    dimensions = ["goal_structure", "value_compatibility", "power_dynamic", "stakes"]
    results = {}

    ep_mean_residuals = {}
    for combo_idx, entries in per_episode_residuals.items():
        vals = [v for _, v in entries]
        ep_mean_residuals[combo_idx] = float(np.mean(vals))

    for dim in dimensions:
        groups = defaultdict(list)
        for combo_idx, mean_res in ep_mean_residuals.items():
            label = labels.get(str(combo_idx))
            if label is None or label.get(dim) is None:
                continue
            groups[label[dim]].append(mean_res)

        if len(groups) < 2:
            continue

        print(f"\n  --- {dim} (residual L2 norms) ---")
        dim_stats = {}
        for cat, vals in sorted(groups.items()):
            m, s = np.mean(vals), np.std(vals)
            dim_stats[cat] = {"mean": float(m), "std": float(s), "n": len(vals)}
            print(f"    {cat:25s}: residual={m:.4f} ± {s:.4f}  (n={len(vals)})")

        group_vals = [np.array(v) for v in groups.values()]
        if len(group_vals) == 2:
            t_stat, t_p = stats.ttest_ind(*group_vals)
            print(f"    t-test: t={t_stat:.3f}, p={t_p:.4e}")
            dim_stats["test"] = {"type": "t-test", "statistic": float(t_stat), "p": float(t_p)}
        else:
            f_stat, f_p = stats.f_oneway(*group_vals)
            print(f"    ANOVA: F={f_stat:.3f}, p={f_p:.4e}")
            dim_stats["test"] = {"type": "ANOVA", "statistic": float(f_stat), "p": float(f_p)}

        results[dim] = dim_stats

        cats = sorted(groups.keys())
        data = [groups[c] for c in cats]
        labels_text = [f"{c}\n(n={len(groups[c])})" for c in cats]

        fig, ax = plt.subplots(figsize=(max(6, len(cats) * 1.5), 5))
        bp = ax.boxplot(data, labels=labels_text, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set_ylabel("Mean Residual L2 Norm")
        ax.set_title(f"{model_tag} — Residuals by {dim} (L{layer_A}→L{layer_B})")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"residual_{dim}_{model_tag}_L{layer_A}_L{layer_B}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {path}")

    score_vals, res_vals = [], []
    for combo_idx, mean_res in ep_mean_residuals.items():
        label = labels.get(str(combo_idx))
        if label is None or label.get("alignment_score") is None:
            continue
        score_vals.append(label["alignment_score"])
        res_vals.append(mean_res)

    if len(score_vals) > 10:
        score_arr = np.array(score_vals)
        res_arr = np.array(res_vals)
        pr, pp = stats.pearsonr(score_arr, res_arr)
        sr, sp = stats.spearmanr(score_arr, res_arr)
        print(f"\n  Alignment score vs residual L2:")
        print(f"    Pearson  r={pr:.4f}, p={pp:.4e}")
        print(f"    Spearman r={sr:.4f}, p={sp:.4e}")
        results["alignment_score_correlation"] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(score_arr, res_arr, alpha=0.4, s=20)
        z = np.polyfit(score_arr, res_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(score_arr.min(), score_arr.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2,
                label=f"Pearson r={pr:.3f} (p={pp:.2e})")
        ax.set_xlabel("Alignment Score")
        ax.set_ylabel("Mean Residual L2 Norm")
        ax.set_title(f"{model_tag} — Alignment vs Residual (L{layer_A}→L{layer_B})")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"residual_scatter_{model_tag}_L{layer_A}_L{layer_B}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"    Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_features(model_name, setting):
    modes, episodes_id = get_all_modes()
    flat_to_combo = build_flat_episode_map(modes, episodes_id)

    print(f"Loading features for {model_name}...")
    allAs, allBs, seed_boundaries = get_preloaded_features(
        modes, episodes_id, model_name,
        setting=setting, data_mode="combined_metrics", seed_list=[0, 1, 2, 3, 4],
    )
    print(f"Loaded {len(allAs)} episodes")
    return allAs, allBs, flat_to_combo


def run_pathway_a(model_name, layer_A, layer_B, setting, labels):
    allAs, allBs, flat_to_combo = load_features(model_name, setting)
    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    all_dimensions = [
        "alignment_binary",
        "goal_structure",
        "value_compatibility",
    ]

    all_results = {}
    for dim in all_dimensions:
        print(f"\n{'='*60}")
        print(f"Pathway A: {dim}")
        print(f"{'='*60}")

        mean_matrix, per_seed, categories = run_cross_transfer(
            allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
            dimension=dim, seeds=(0, 1, 2),
        )
        if mean_matrix is None:
            continue

        gap_info = compute_transfer_gap(mean_matrix, categories)
        print(f"\n  Transfer gap: diagonal={gap_info['diagonal_mean']:.4f}, "
              f"off-diagonal={gap_info['off_diagonal_mean']:.4f}, "
              f"gap={gap_info['gap']:.4f} ({gap_info['gap_ratio']:.1%})")

        plot_transfer_matrix(mean_matrix, categories, model_tag, dim, layer_A, layer_B)

        serializable_matrix = {f"{k[0]}_to_{k[1]}": v for k, v in mean_matrix.items()}
        all_results[dim] = {
            "categories": categories,
            "transfer_matrix": serializable_matrix,
            "gap": gap_info,
        }

    return all_results


def run_pathway_c(model_name, layer_A, layer_B, setting, labels):
    allAs, allBs, flat_to_combo = load_features(model_name, setting)
    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    print(f"\n{'='*60}")
    print(f"Pathway C: Residual Analysis")
    print(f"{'='*60}")

    per_episode_residuals = run_residual_analysis(
        allAs, allBs, flat_to_combo, labels, layer_A, layer_B,
        seeds=(0, 1, 2),
    )

    residual_results = analyze_residuals_by_label(
        per_episode_residuals, labels, model_tag, layer_A, layer_B,
    )

    return residual_results


def run_single_model(model_name, layer_A=None, layer_B=None,
                     setting_str="A_forward", pathway="both"):
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])
    labels = load_labels()

    if layer_A is None or layer_B is None:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)

    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    print(f"\n{'#'*60}")
    print(f"Model: {model_name} ({model_tag})")
    print(f"Layer pair: A={layer_A}, B={layer_B}")
    print(f"Pathway: {pathway}")
    print(f"{'#'*60}")

    summary = {
        "model": model_tag,
        "model_full": model_name,
        "layer_A": layer_A,
        "layer_B": layer_B,
    }

    if pathway in ("A", "both"):
        a_results = run_pathway_a(model_name, layer_A, layer_B, setting, labels)
        summary["pathway_a"] = a_results

    if pathway in ("C", "both"):
        c_results = run_pathway_c(model_name, layer_A, layer_B, setting, labels)
        summary["pathway_c"] = c_results

    output_path = os.path.join(
        SCRIPT_DIR, f"stratified_analysis_{model_tag}_L{layer_A}_L{layer_B}.json"
    )
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Alignment-stratified affine analysis")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layer_A", type=int, default=None)
    parser.add_argument("--layer_B", type=int, default=None)
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--pathway", type=str, default="both",
                        choices=["A", "C", "both"])
    parser.add_argument("--all_pairs", action="store_true")
    args = parser.parse_args()

    if args.all_pairs:
        all_summaries = []
        for model in ALL_MISTRAL_PAIRS:
            s = run_single_model(model, args.layer_A, args.layer_B,
                                 args.setting, args.pathway)
            all_summaries.append(s)

        print(f"\n\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        for s in all_summaries:
            tag = s["model"]
            print(f"\n  {tag}:")
            if "pathway_a" in s:
                for dim, info in s["pathway_a"].items():
                    g = info["gap"]
                    print(f"    [{dim}] diag={g['diagonal_mean']:.4f}  "
                          f"off-diag={g['off_diagonal_mean']:.4f}  "
                          f"gap={g['gap']:+.4f} ({g['gap_ratio']:+.1%})")
            if "pathway_c" in s:
                corr = s["pathway_c"].get("alignment_score_correlation", {})
                if corr:
                    print(f"    [residual~alignment] Pearson r={corr['pearson_r']:.4f} "
                          f"(p={corr['pearson_p']:.2e})")

        agg_path = os.path.join(SCRIPT_DIR, "stratified_analysis_summary.json")
        with open(agg_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nSaved aggregate summary: {agg_path}")

    elif args.model:
        run_single_model(args.model, args.layer_A, args.layer_B,
                         args.setting, args.pathway)
    else:
        run_single_model(ALL_MISTRAL_PAIRS[0], args.layer_A, args.layer_B,
                         args.setting, args.pathway)


if __name__ == "__main__":
    main()
