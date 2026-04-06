"""
SVD-based representational analysis of LLM agent dialogue turns.

Analysis 1: Singular value spectrum and explained variance.
Analysis 2: Turn-by-turn trajectory in SVD space, stratified by alignment.
Analysis 3: Cross-episode cosine similarity in SVD space.

Each agent is analyzed independently (separate SVD basis per agent),
making this valid for both same-model and cross-model pairs.
For same-model pairs, a joint SVD on concatenated A+B turns is also run.

Usage:
    python analyze_svd.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0
    python analyze_svd.py --all_pairs
    python analyze_svd.py --all_pairs --k 100 --layers 8 16
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import REPO_ROOT, set_seed, get_all_modes, get_short_names_and_identifiers
from affine_transformation import get_preloaded_features
from analyze_alignment import find_best_layer_pair, build_flat_episode_map
from analyze_stratified import (
    load_labels,
    partition_episodes_by_label,
    partition_by_alignment_score,
    split_episodes,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "playground", "svd_plots")

ALL_MISTRAL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]


# ---------------------------------------------------------------------------
# Core SVD functions
# ---------------------------------------------------------------------------

def stack_turns_single_agent(activations, layer, episode_indices=None):
    """Stack all turns from one agent at a given layer into (N, hidden_dim)."""
    if episode_indices is None:
        episode_indices = range(len(activations))
    rows = []
    episode_boundaries = []
    for idx in episode_indices:
        ep = activations[idx]
        n_turns = ep.shape[0]
        if n_turns < 1:
            episode_boundaries.append(len(rows))
            continue
        rows.append(ep[:n_turns, layer, :])
        episode_boundaries.append(len(rows))
    if not rows:
        return None, []
    return np.concatenate(rows, axis=0).astype(np.float32), episode_boundaries


def compute_svd_basis(activations, layer, episode_indices, k=50):
    """
    Compute top-k SVD basis from one agent's turns at a given layer.

    Returns:
        V_k: (k, hidden_dim) -- top-k right singular vectors
        S: (min(N, d),) -- all singular values
        mean_vec: (hidden_dim,) -- mean used for centering
        n_samples: int -- number of turns used
    """
    X, _ = stack_turns_single_agent(activations, layer, episode_indices)
    if X is None or X.shape[0] < 2:
        return None, None, None, 0

    mean_vec = X.mean(axis=0)
    X_centered = X - mean_vec

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(k, Vt.shape[0])
    V_k = Vt[:k]

    return V_k, S, mean_vec, X.shape[0]


def project_episodes(activations, layer, V_k, mean_vec, episode_indices=None):
    """
    Project each episode's turns into the k-dim SVD space.

    Returns list of arrays, each (n_turns_i, k).
    """
    if episode_indices is None:
        episode_indices = range(len(activations))

    trajectories = []
    for idx in episode_indices:
        ep = activations[idx]
        n_turns = ep.shape[0]
        if n_turns < 1:
            trajectories.append(np.zeros((0, V_k.shape[0])))
            continue
        X = ep[:n_turns, layer, :].astype(np.float32)
        Z = (X - mean_vec) @ V_k.T
        trajectories.append(Z)
    return trajectories


# ---------------------------------------------------------------------------
# Analysis 1: Spectrum
# ---------------------------------------------------------------------------

def analyze_spectrum(S, model_tag, agent_label, layer, k_highlight=50):
    """Compute explained variance statistics from singular values."""
    var = S ** 2
    total_var = var.sum()
    cum_var = np.cumsum(var) / total_var

    k90 = int(np.searchsorted(cum_var, 0.90)) + 1
    k95 = int(np.searchsorted(cum_var, 0.95)) + 1
    k99 = int(np.searchsorted(cum_var, 0.99)) + 1
    top_k_var = float(cum_var[min(k_highlight - 1, len(cum_var) - 1)])

    print(f"    Spectrum ({agent_label}, L{layer}): "
          f"k90={k90}, k95={k95}, k99={k99}, "
          f"top-{k_highlight} explains {top_k_var:.1%}")

    return {
        "k90": k90, "k95": k95, "k99": k99,
        "top_k_explained": float(top_k_var),
        "k_highlight": k_highlight,
        "singular_values_top100": [float(s) for s in S[:100]],
        "cumulative_variance_top100": [float(c) for c in cum_var[:100]],
    }


# ---------------------------------------------------------------------------
# Analysis 2: Trajectory metrics
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(trajectories, episode_indices, flat_to_combo):
    """
    Per-episode trajectory metrics in SVD space.

    Returns list of dicts with combo_idx, path_length, net_displacement,
    mean_step_cosine, n_turns.
    """
    results = []
    for i, traj in enumerate(trajectories):
        if traj.shape[0] < 2:
            continue
        ep_idx = episode_indices[i]
        combo_idx = flat_to_combo[ep_idx]

        steps = np.diff(traj, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        path_length = float(step_norms.sum())
        net_displacement = float(np.linalg.norm(traj[-1] - traj[0]))

        step_cosines = []
        for t in range(len(traj) - 1):
            n1 = np.linalg.norm(traj[t])
            n2 = np.linalg.norm(traj[t + 1])
            if n1 > 1e-8 and n2 > 1e-8:
                step_cosines.append(float(np.dot(traj[t], traj[t + 1]) / (n1 * n2)))
        mean_step_cosine = float(np.mean(step_cosines)) if step_cosines else 0.0

        straightness = net_displacement / (path_length + 1e-12)

        results.append({
            "combo_idx": combo_idx,
            "episode_idx": ep_idx,
            "path_length": path_length,
            "net_displacement": net_displacement,
            "mean_step_cosine": mean_step_cosine,
            "straightness": straightness,
            "n_turns": traj.shape[0],
        })
    return results


def stratify_metrics(metrics, labels, dimension):
    """Group trajectory metrics by alignment category."""
    if dimension == "alignment_binary":
        cat_map = {}
        for m in metrics:
            label = labels.get(str(m["combo_idx"]))
            if label is None or label.get("alignment_score") is None:
                continue
            score = label["alignment_score"]
            if score >= 0.5:
                cat_map.setdefault("high_align", []).append(m)
            elif score <= -0.5:
                cat_map.setdefault("low_align", []).append(m)
    else:
        cat_map = defaultdict(list)
        for m in metrics:
            label = labels.get(str(m["combo_idx"]))
            if label is None or label.get(dimension) is None:
                continue
            cat_map[label[dimension]].append(m)

    return dict(cat_map)


def summarize_stratified_metrics(cat_map, metric_name):
    """Print and return summary statistics for a metric across categories."""
    summary = {}
    for cat in sorted(cat_map.keys()):
        vals = [m[metric_name] for m in cat_map[cat]]
        summary[cat] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }
    return summary


# ---------------------------------------------------------------------------
# Analysis 3: Cross-episode cosine similarity
# ---------------------------------------------------------------------------

def compute_episode_mean_projections(trajectories, episode_indices):
    """Compute mean SVD projection per episode (episodes with >= 1 turn)."""
    means = {}
    for i, traj in enumerate(trajectories):
        if traj.shape[0] < 1:
            continue
        means[episode_indices[i]] = traj.mean(axis=0)
    return means


def compute_pairwise_category_similarity(
    episode_means, flat_to_combo, labels, dimension,
):
    """
    Compute mean cosine similarity for same-category vs cross-category
    episode pairs in SVD space.
    """
    ep_to_cat = {}
    for ep_idx, mean_vec in episode_means.items():
        combo_idx = flat_to_combo[ep_idx]
        label = labels.get(str(combo_idx))
        if label is None:
            continue
        if dimension == "alignment_binary":
            score = label.get("alignment_score")
            if score is None:
                continue
            if score >= 0.5:
                ep_to_cat[ep_idx] = "high_align"
            elif score <= -0.5:
                ep_to_cat[ep_idx] = "low_align"
        else:
            cat = label.get(dimension)
            if cat is None:
                continue
            ep_to_cat[ep_idx] = cat

    labeled_eps = [ep for ep in episode_means if ep in ep_to_cat]
    if len(labeled_eps) < 2:
        return None

    same_cat_sims = []
    cross_cat_sims = []

    for i in range(len(labeled_eps)):
        for j in range(i + 1, len(labeled_eps)):
            ep_i, ep_j = labeled_eps[i], labeled_eps[j]
            v_i = episode_means[ep_i]
            v_j = episode_means[ep_j]
            ni, nj = np.linalg.norm(v_i), np.linalg.norm(v_j)
            if ni < 1e-8 or nj < 1e-8:
                continue
            sim = float(np.dot(v_i, v_j) / (ni * nj))

            if ep_to_cat[ep_i] == ep_to_cat[ep_j]:
                same_cat_sims.append(sim)
            else:
                cross_cat_sims.append(sim)

    if not same_cat_sims or not cross_cat_sims:
        return None

    t_stat, t_p = stats.ttest_ind(same_cat_sims, cross_cat_sims)

    return {
        "same_cat_mean": float(np.mean(same_cat_sims)),
        "same_cat_std": float(np.std(same_cat_sims)),
        "cross_cat_mean": float(np.mean(cross_cat_sims)),
        "cross_cat_std": float(np.std(cross_cat_sims)),
        "n_same": len(same_cat_sims),
        "n_cross": len(cross_cat_sims),
        "t_stat": float(t_stat),
        "p_value": float(t_p),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(spectrum_info, model_tag, agent_label, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    sv = spectrum_info["singular_values_top100"]
    cv = spectrum_info["cumulative_variance_top100"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(sv) + 1), sv, "b-", linewidth=1.5)
    ax1.set_xlabel("Component Index")
    ax1.set_ylabel("Singular Value")
    ax1.set_title(f"Singular Value Spectrum")
    ax1.set_yscale("log")

    ax2.plot(range(1, len(cv) + 1), cv, "r-", linewidth=1.5)
    ax2.axhline(0.90, color="gray", linestyle="--", alpha=0.5, label=f"90% (k={spectrum_info['k90']})")
    ax2.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label=f"95% (k={spectrum_info['k95']})")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title(f"Explained Variance")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(f"{model_tag} — {agent_label} (L{layer})")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"spectrum_{agent_label}_{model_tag}_L{layer}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_trajectories_2d(
    trajectories, episode_indices, flat_to_combo, labels,
    dimension, model_tag, agent_label, layer, max_episodes=30,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    if dimension == "alignment_binary":
        cat_map = {}
        for i, ep_idx in enumerate(episode_indices):
            combo_idx = flat_to_combo[ep_idx]
            label = labels.get(str(combo_idx))
            if label is None or label.get("alignment_score") is None:
                continue
            score = label["alignment_score"]
            if score >= 0.5:
                cat_map[i] = "high_align"
            elif score <= -0.5:
                cat_map[i] = "low_align"
    else:
        cat_map = {}
        for i, ep_idx in enumerate(episode_indices):
            combo_idx = flat_to_combo[ep_idx]
            label = labels.get(str(combo_idx))
            if label is None or label.get(dimension) is None:
                continue
            cat_map[i] = label[dimension]

    categories = sorted(set(cat_map.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))
    cat_to_color = {cat: colors[i] for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = 0

    for cat in categories:
        cat_indices = [i for i, c in cat_map.items() if c == cat]
        np.random.seed(42)
        if len(cat_indices) > max_episodes // len(categories):
            cat_indices = list(np.random.choice(cat_indices, max_episodes // len(categories), replace=False))

        for i in cat_indices:
            traj = trajectories[i]
            if traj.shape[0] < 2 or traj.shape[1] < 2:
                continue
            ax.plot(traj[:, 0], traj[:, 1], "-", color=cat_to_color[cat],
                    alpha=0.4, linewidth=1)
            ax.scatter(traj[0, 0], traj[0, 1], color=cat_to_color[cat],
                       marker="o", s=30, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=cat_to_color[cat],
                       marker="x", s=30, zorder=5)
            plotted += 1

    for cat in categories:
        ax.plot([], [], "-", color=cat_to_color[cat], label=cat, linewidth=2)
    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.set_title(f"{model_tag} — {agent_label} Trajectories ({dimension}, L{layer})")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"traj2d_{dimension}_{agent_label}_{model_tag}_L{layer}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_trajectory_metrics_by_category(
    cat_map, metric_name, model_tag, agent_label, layer, dimension,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    cats = sorted(cat_map.keys())
    data = [[m[metric_name] for m in cat_map[c]] for c in cats]
    x_labels = [f"{c}\n(n={len(cat_map[c])})" for c in cats]

    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 1.5), 5))
    bp = ax.boxplot(data, labels=x_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)

    label_map = {
        "path_length": "Path Length",
        "net_displacement": "Net Displacement",
        "mean_step_cosine": "Mean Step Cosine",
        "straightness": "Straightness",
    }
    ax.set_ylabel(label_map.get(metric_name, metric_name))
    ax.set_title(f"{model_tag} — {agent_label} {label_map.get(metric_name, metric_name)} "
                 f"by {dimension} (L{layer})")
    fig.tight_layout()

    path = os.path.join(
        PLOTS_DIR,
        f"metric_{metric_name}_{dimension}_{agent_label}_{model_tag}_L{layer}.png",
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


def plot_pairwise_similarity_summary(
    sim_results, model_tag, agent_label, layer,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)

    dims = sorted(sim_results.keys())
    valid_dims = [d for d in dims if sim_results[d] is not None]
    if not valid_dims:
        return

    fig, axes = plt.subplots(1, len(valid_dims), figsize=(5 * len(valid_dims), 5))
    if len(valid_dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, valid_dims):
        r = sim_results[dim]
        x = [0, 1]
        means = [r["same_cat_mean"], r["cross_cat_mean"]]
        stds = [r["same_cat_std"], r["cross_cat_std"]]
        bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5, alpha=0.8)
        bars[0].set_color("tab:blue")
        bars[1].set_color("tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(["Same Cat", "Cross Cat"])
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title(f"{dim}\np={r['p_value']:.2e}")

    fig.suptitle(f"{model_tag} — {agent_label} Pairwise Cosine Sim (L{layer})")
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, f"pairwise_sim_{agent_label}_{model_tag}_L{layer}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Orchestration: per-agent analysis
# ---------------------------------------------------------------------------

def run_agent_analysis(
    activations, agent_label, layer, flat_to_combo, labels,
    model_tag, k=50, seed=0,
):
    """Run the full SVD pipeline for a single agent."""
    all_indices = list(range(len(activations)))
    rng = np.random.default_rng(seed)
    train_eps, test_eps = split_episodes(all_indices, train_frac=0.8, rng=rng)

    print(f"\n  --- {agent_label}, Layer {layer} ---")
    print(f"    Train: {len(train_eps)} episodes, Test: {len(test_eps)} episodes")

    # --- Analysis 1: Spectrum ---
    V_k, S, mean_vec, n_samples = compute_svd_basis(activations, layer, train_eps, k=k)
    if V_k is None:
        print(f"    No data for {agent_label}, skipping.")
        return None

    print(f"    SVD computed on {n_samples} turns, k={V_k.shape[0]}")
    spectrum = analyze_spectrum(S, model_tag, agent_label, layer, k_highlight=k)
    plot_spectrum(spectrum, model_tag, agent_label, layer)

    # --- Analysis 2: Trajectories on TEST episodes ---
    test_trajectories = project_episodes(activations, layer, V_k, mean_vec, test_eps)
    metrics = compute_trajectory_metrics(test_trajectories, test_eps, flat_to_combo)
    print(f"    Computed trajectory metrics for {len(metrics)} test episodes")

    trajectory_results = {}
    for dimension in ["goal_structure", "value_compatibility", "alignment_binary"]:
        cat_map = stratify_metrics(metrics, labels, dimension)
        if len(cat_map) < 2:
            continue

        print(f"\n    [{dimension}]")
        for metric_name in ["path_length", "net_displacement", "mean_step_cosine", "straightness"]:
            summary = summarize_stratified_metrics(cat_map, metric_name)
            for cat in sorted(summary.keys()):
                s = summary[cat]
                print(f"      {cat:25s}: {metric_name}={s['mean']:.4f} ± {s['std']:.4f} (n={s['n']})")

            plot_trajectory_metrics_by_category(
                cat_map, metric_name, model_tag, agent_label, layer, dimension)

        vals_by_cat = defaultdict(list)
        for cat, ms in cat_map.items():
            for m in ms:
                vals_by_cat[cat].append(m["path_length"])
        group_vals = [np.array(v) for v in vals_by_cat.values()]
        if len(group_vals) >= 2:
            if len(group_vals) == 2:
                t_stat, t_p = stats.ttest_ind(*group_vals)
                print(f"      path_length t-test: t={t_stat:.3f}, p={t_p:.4e}")
            else:
                f_stat, f_p = stats.f_oneway(*group_vals)
                print(f"      path_length ANOVA: F={f_stat:.3f}, p={f_p:.4e}")

        trajectory_results[dimension] = {
            metric_name: summarize_stratified_metrics(cat_map, metric_name)
            for metric_name in ["path_length", "net_displacement", "mean_step_cosine", "straightness"]
        }

        plot_trajectories_2d(
            test_trajectories, test_eps, flat_to_combo, labels,
            dimension, model_tag, agent_label, layer)

    # --- Analysis 3: Pairwise similarity ---
    all_trajectories = project_episodes(activations, layer, V_k, mean_vec, all_indices)
    episode_means = compute_episode_mean_projections(all_trajectories, all_indices)

    sim_results = {}
    print(f"\n    Pairwise cosine similarity:")
    for dimension in ["goal_structure", "value_compatibility", "alignment_binary"]:
        sim = compute_pairwise_category_similarity(
            episode_means, flat_to_combo, labels, dimension)
        sim_results[dimension] = sim
        if sim is not None:
            print(f"      [{dimension}] same={sim['same_cat_mean']:.4f}, "
                  f"cross={sim['cross_cat_mean']:.4f}, "
                  f"p={sim['p_value']:.2e}")

    plot_pairwise_similarity_summary(sim_results, model_tag, agent_label, layer)

    return {
        "agent": agent_label,
        "layer": layer,
        "k": k,
        "n_train_episodes": len(train_eps),
        "n_test_episodes": len(test_eps),
        "n_train_turns": n_samples,
        "spectrum": spectrum,
        "trajectory_metrics": trajectory_results,
        "pairwise_similarity": sim_results,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def is_same_model_pair(model_name):
    parts = model_name.split("_None_0_")
    if len(parts) != 2:
        return False
    return parts[0] == parts[1].rstrip("_None_0")


def run_single_model(model_name, layers=None, k=50, setting_str="A_forward",
                     output_prefix="svd_analysis"):
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])
    labels = load_labels()

    if layers is None:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)
        layers_A = [layer_A]
        layers_B = [layer_B]
    else:
        layers_A = layers
        layers_B = layers

    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    print(f"\n{'#'*60}")
    print(f"SVD Analysis: {model_name} ({model_tag})")
    print(f"Layers A: {layers_A}, Layers B: {layers_B}, k={k}")
    print(f"{'#'*60}")

    modes, episodes_id = get_all_modes()
    flat_to_combo = build_flat_episode_map(modes, episodes_id)

    print(f"Loading features for {model_name}...")
    allAs, allBs, _ = get_preloaded_features(
        modes, episodes_id, model_name,
        setting=setting, data_mode="combined_metrics", seed_list=[0, 1, 2, 3, 4],
    )
    print(f"Loaded {len(allAs)} episodes")

    # --- Try to load existing results to skip already-computed layers ---
    layers_str = "_".join(f"L{l}" for l in sorted(set(layers_A + layers_B)))
    output_path = os.path.join(
        SCRIPT_DIR, f"{output_prefix}_{model_tag}_{layers_str}.json"
    )

    summary = {
        "model": model_tag,
        "model_full": model_name,
        "k": k,
        "same_model": is_same_model_pair(model_name),
        "agent_A": {},
        "agent_B": {},
        "joint": {},
    }

    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            summary["agent_A"] = existing.get("agent_A", {})
            summary["agent_B"] = existing.get("agent_B", {})
            prev_joint = existing.get("joint", {})
            if isinstance(prev_joint, dict) and "layer" in prev_joint:
                summary["joint"] = {f"L{prev_joint['layer']}": prev_joint}
            else:
                summary["joint"] = prev_joint if isinstance(prev_joint, dict) else {}
            print(f"  Loaded existing results: {len(summary['agent_A'])} A layers, "
                  f"{len(summary['agent_B'])} B layers, "
                  f"{len(summary['joint'])} joint layers")
        except Exception as e:
            print(f"  Could not load existing results ({e}), computing from scratch")

    for layer in layers_A:
        lk = f"L{layer}"
        if lk in summary["agent_A"]:
            print(f"  Skipping Agent A {lk} (already computed)")
            continue
        result = run_agent_analysis(
            allAs, "AgentA", layer, flat_to_combo, labels, model_tag, k=k)
        if result is not None:
            summary["agent_A"][lk] = result

    for layer in layers_B:
        lk = f"L{layer}"
        if lk in summary["agent_B"]:
            print(f"  Skipping Agent B {lk} (already computed)")
            continue
        result = run_agent_analysis(
            allBs, "AgentB", layer, flat_to_combo, labels, model_tag, k=k)
        if result is not None:
            summary["agent_B"][lk] = result

    if is_same_model_pair(model_name):
        print(f"\n  --- Joint SVD (same-model pair) ---")
        all_indices = list(range(len(allAs)))
        joint_layers = sorted(set(layers_A) | set(layers_B))

        for layer in joint_layers:
            lk = f"L{layer}"
            if lk in summary["joint"]:
                print(f"  Skipping Joint {lk} (already computed)")
                continue

            print(f"  Joint SVD layer {layer}...")
            joint_acts = []
            for idx in all_indices:
                ep_A = allAs[idx]
                ep_B = allBs[idx]
                n = min(ep_A.shape[0], ep_B.shape[0])
                if n < 1:
                    continue
                combined = np.concatenate([
                    ep_A[:n, layer, :],
                    ep_B[:n, layer, :],
                ], axis=0)
                joint_acts.append(combined)

            joint_stacked = np.concatenate(joint_acts, axis=0).astype(np.float32)
            mean_vec = joint_stacked.mean(axis=0)
            X_centered = joint_stacked - mean_vec
            _, S_joint, Vt_joint = np.linalg.svd(X_centered, full_matrices=False)
            V_k_joint = Vt_joint[:k]

            spectrum_joint = analyze_spectrum(S_joint, model_tag, "Joint", layer, k_highlight=k)
            plot_spectrum(spectrum_joint, model_tag, "Joint", layer)

            proj_A = project_episodes(allAs, layer, V_k_joint, mean_vec)
            proj_B = project_episodes(allBs, layer, V_k_joint, mean_vec)

            ab_cosines = []
            for i in range(len(proj_A)):
                if proj_A[i].shape[0] < 1 or proj_B[i].shape[0] < 1:
                    continue
                mean_a = proj_A[i].mean(axis=0)
                mean_b = proj_B[i].mean(axis=0)
                na, nb = np.linalg.norm(mean_a), np.linalg.norm(mean_b)
                if na > 1e-8 and nb > 1e-8:
                    ab_cosines.append(float(np.dot(mean_a, mean_b) / (na * nb)))

            summary["joint"][lk] = {
                "layer": layer,
                "spectrum": spectrum_joint,
                "agent_ab_cosine_mean": float(np.mean(ab_cosines)) if ab_cosines else 0.0,
                "agent_ab_cosine_std": float(np.std(ab_cosines)) if ab_cosines else 0.0,
                "n_episodes": len(ab_cosines),
            }
            print(f"    L{layer}: Agent A vs B cosine = "
                  f"{np.mean(ab_cosines):.4f} ± {np.std(ab_cosines):.4f}")

    if not summary["joint"]:
        del summary["joint"]

    # --- Save ---
    def _make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_make_serializable)
    print(f"\nSaved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SVD analysis of dialogue representations")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to analyze (default: best from affine results)")
    parser.add_argument("--k", type=int, default=50, help="Number of SVD components")
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--all_pairs", action="store_true")
    parser.add_argument("--output_prefix", type=str, default="svd_analysis",
                        help="Prefix for output JSON filenames")
    args = parser.parse_args()

    if args.all_pairs:
        all_summaries = []
        for model in ALL_MISTRAL_PAIRS:
            s = run_single_model(model, args.layers, args.k, args.setting,
                                 args.output_prefix)
            all_summaries.append(s)

        print(f"\n\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        for s in all_summaries:
            tag = s["model"]
            print(f"\n  {tag} (same_model={s['same_model']}):")
            for agent_key in ["agent_A", "agent_B"]:
                for layer_key, info in s[agent_key].items():
                    sp = info["spectrum"]
                    print(f"    {agent_key} {layer_key}: k90={sp['k90']}, k95={sp['k95']}, "
                          f"top-{sp['k_highlight']} var={sp['top_k_explained']:.1%}")
            if "joint" in s:
                joint = s["joint"]
                if isinstance(joint, dict) and "layer" in joint:
                    print(f"    Joint L{joint['layer']}: A-B cosine={joint['agent_ab_cosine_mean']:.4f}")
                elif isinstance(joint, dict):
                    for jk in sorted(joint.keys(), key=lambda x: int(x[1:]) if x.startswith('L') else -1):
                        j = joint[jk]
                        print(f"    Joint {jk}: A-B cosine={j['agent_ab_cosine_mean']:.4f} "
                              f"± {j['agent_ab_cosine_std']:.4f}")

        agg_path = os.path.join(SCRIPT_DIR, f"{args.output_prefix}_summary.json")
        with open(agg_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=_make_serializable)
        print(f"\nSaved aggregate: {agg_path}")

    elif args.model:
        run_single_model(args.model, args.layers, args.k, args.setting,
                         args.output_prefix)
    else:
        run_single_model(ALL_MISTRAL_PAIRS[0], args.layers, args.k, args.setting,
                         args.output_prefix)


def _make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    main()
