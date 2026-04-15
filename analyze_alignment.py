"""
Per-episode R² analysis correlated with goal alignment labels.

Phase 1: Retrain affine map at the best layer pair, compute per-episode R².
Phase 2: Merge with alignment labels from goal_alignment_labels.json.
Phase 3: Statistical analysis and plots.

Usage:
    python analyze_alignment.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0
    python analyze_alignment.py --model Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0 --layer_A 10 --layer_B 8
    python analyze_alignment.py --all_pairs
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import REPO_ROOT, set_seed, get_all_modes, get_short_names_and_identifiers
from affine_transformation import (
    get_preloaded_features,
    sampling_with_full_episodes,
    train_affine_transformation,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(SCRIPT_DIR, "goal_alignment_labels.json")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "playground", "alignment_plots")


# ---------------------------------------------------------------------------
# Phase 1: Per-episode R²
# ---------------------------------------------------------------------------

def find_best_layer_pair(model_name, setting="A_forward", data_mode="combined_metrics"):
    """Scan existing affine results to find the layer pair with highest mean R²."""
    short_names, _ = get_short_names_and_identifiers([model_name])
    model_key = short_names[0]

    base = os.path.join(REPO_ROOT, "affine_transformation", setting)
    best_r2, best_pair = -999, (16, 16)

    for la in range(32):
        data_dir = os.path.join(base, f"layerA{la}", "data")
        for lb in range(32):
            fn = os.path.join(
                data_dir,
                f"{data_mode}_{model_key}[0, 1, 2, 3, 4]_layerA{la}_layerB{lb}_allreps.json",
            )
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                data = json.load(f)
            if model_key not in data:
                continue
            r2 = data[model_key]["r2_mean"][0]
            if r2 > best_r2:
                best_r2 = r2
                best_pair = (la, lb)

    print(f"Best layer pair for {model_key}: A={best_pair[0]}, B={best_pair[1]}, R²={best_r2:.4f}")
    return best_pair, best_r2


def build_flat_episode_map(modes, episodes_id):
    """Build mapping from flat index (as used by allAs/allBs) to combo index."""
    flat_to_combo = []
    for mode in modes:
        for ep in episodes_id[mode]:
            flat_to_combo.append(ep)
    return flat_to_combo


def compute_per_episode_r2(
    model_name,
    layer_A,
    layer_B,
    setting=("A", "forward"),
    seeds=(0, 1, 2),
    seed_list=None,
    data_mode="combined_metrics",
    precomputed_sample_num=6500,
):
    """
    Retrain the affine map at a single layer pair for each seed,
    then compute R² for each individual episode in the test set.

    Returns:
        per_episode_r2: dict mapping combo_index -> list of (seed, r2) tuples
        flat_to_combo: list mapping flat index -> combo index
        global_r2_per_seed: list of global R² per seed
    """
    if seed_list is None:
        seed_list = [0, 1, 2, 3, 4]

    modes, episodes_id = get_all_modes()
    flat_to_combo = build_flat_episode_map(modes, episodes_id)

    print(f"Loading features for {model_name}...")
    allAs, allBs, seed_boundaries = get_preloaded_features(
        modes, episodes_id, model_name,
        setting=setting, data_mode=data_mode, seed_list=seed_list,
    )
    print(f"Loaded {len(allAs)} episodes")

    per_episode_r2 = defaultdict(list)
    global_r2_per_seed = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        train_X, train_Y, test_X, test_Y, test_set_idx, train_set_idx = \
            sampling_with_full_episodes(
                allAs, allBs, precomputed_sample_num,
                shuffle=False, seed_boundaries=seed_boundaries,
            )

        train_A = torch.tensor(train_X[:, layer_A, :]).to("cuda")
        train_B = torch.tensor(train_Y[:, layer_B, :]).to("cuda")
        test_A = torch.tensor(test_X[:, layer_A, :]).to("cuda")
        test_B = torch.tensor(test_Y[:, layer_B, :]).to("cuda")

        set_seed(seed)
        global_r2, _, _, _, _, _, G = train_affine_transformation(
            train_X=train_A, train_Y=train_B,
            test_X=test_A, test_Y=test_B,
            device="cuda", verbose=False, seed=seed,
        )
        global_r2_per_seed.append(global_r2)
        print(f"Global test R²: {global_r2:.4f}")

        G.eval()
        for flat_idx in test_set_idx:
            combo_idx = flat_to_combo[flat_idx]
            ep_A = allAs[flat_idx]
            ep_B = allBs[flat_idx]
            n_turns = min(ep_A.shape[0], ep_B.shape[0])
            if n_turns < 2:
                continue

            ep_A_layer = torch.tensor(ep_A[:n_turns, layer_A, :]).float().to("cuda")
            ep_B_layer = ep_B[:n_turns, layer_B, :]

            with torch.no_grad():
                pred_B = G(ep_A_layer).cpu().numpy()

            try:
                ep_r2 = sklearn_r2_score(ep_B_layer, pred_B)
            except ValueError:
                continue

            per_episode_r2[combo_idx].append((seed, float(ep_r2)))

        print(f"Computed per-episode R² for {len(test_set_idx)} test episodes")

    return dict(per_episode_r2), flat_to_combo, global_r2_per_seed


# ---------------------------------------------------------------------------
# Phase 2: Merge with alignment labels
# ---------------------------------------------------------------------------

def merge_with_labels(per_episode_r2, labels_path=LABELS_PATH):
    """Join per-episode R² with alignment labels."""
    with open(labels_path) as f:
        labels = json.load(f)

    merged = []
    missing_label = 0
    missing_r2 = 0

    all_combo_indices = set(per_episode_r2.keys()) | set(int(k) for k in labels.keys())

    for combo_idx in sorted(all_combo_indices):
        combo_key = str(combo_idx)
        label = labels.get(combo_key)
        r2_entries = per_episode_r2.get(combo_idx)

        if label is None or label.get("alignment_score") is None:
            missing_label += 1
            continue
        if r2_entries is None or len(r2_entries) == 0:
            missing_r2 += 1
            continue

        r2_values = [v for _, v in r2_entries]
        r2_clamped = [max(0.0, v) for v in r2_values]
        entry = {
            "combo_index": combo_idx,
            "env_id": label.get("env_id"),
            "source": label.get("source"),
            "agent_1": label.get("agent_1"),
            "agent_2": label.get("agent_2"),
            "alignment_score": label["alignment_score"],
            "goal_structure": label["goal_structure"],
            "value_compatibility": label["value_compatibility"],
            "power_dynamic": label["power_dynamic"],
            "stakes": label["stakes"],
            "r2_mean": float(np.mean(r2_clamped)),
            "r2_std": float(np.std(r2_clamped)) if len(r2_clamped) > 1 else 0.0,
            "r2_all": r2_clamped,
            "r2_raw": r2_values,
            "n_seeds_in_test": len(r2_values),
        }
        merged.append(entry)

    print(f"\nMerged: {len(merged)} episodes with both R² and labels")
    print(f"Missing label: {missing_label}, Missing R²: {missing_r2}")
    return merged


# ---------------------------------------------------------------------------
# Phase 3: Statistical analysis and plots
# ---------------------------------------------------------------------------

def run_analysis(merged, model_name, layer_A, layer_B, global_r2_per_seed):
    """Correlation analysis, group comparisons, and plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS_DIR, exist_ok=True)
    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]

    alignment_scores = np.array([e["alignment_score"] for e in merged])
    r2_scores = np.array([e["r2_mean"] for e in merged])

    # --- Correlations ---
    pearson_r, pearson_p = stats.pearsonr(alignment_scores, r2_scores)
    spearman_r, spearman_p = stats.spearmanr(alignment_scores, r2_scores)

    print(f"\n{'='*60}")
    print(f"Results for {model_tag} (layers A={layer_A}, B={layer_B})")
    print(f"{'='*60}")
    print(f"Global R² (mean over seeds): {np.mean(global_r2_per_seed):.4f}")
    print(f"Episodes analyzed: {len(merged)}")
    print(f"\nPearson  r={pearson_r:.4f}, p={pearson_p:.4e}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4e}")

    # --- Group comparisons ---
    dimensions = ["goal_structure", "value_compatibility", "power_dynamic", "stakes"]
    group_stats = {}

    for dim in dimensions:
        groups = defaultdict(list)
        for e in merged:
            if e[dim] is not None:
                groups[e[dim]].append(e["r2_mean"])

        print(f"\n--- {dim} ---")
        group_means = {}
        for cat, vals in sorted(groups.items()):
            m, s = np.mean(vals), np.std(vals)
            group_means[cat] = {"mean": m, "std": s, "n": len(vals)}
            print(f"  {cat:25s}: R²={m:.4f} ± {s:.4f}  (n={len(vals)})")
        group_stats[dim] = group_means

        group_vals = [np.array(v) for v in groups.values()]
        if len(group_vals) >= 2:
            if len(group_vals) == 2:
                t_stat, t_p = stats.ttest_ind(*group_vals)
                print(f"  t-test: t={t_stat:.3f}, p={t_p:.4e}")
            else:
                f_stat, f_p = stats.f_oneway(*group_vals)
                print(f"  ANOVA: F={f_stat:.3f}, p={f_p:.4e}")

    # --- Second-stage regression ---
    print(f"\n--- OLS: R² ~ alignment_score + goal_structure + value_compat + stakes ---")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder

        feature_names = ["alignment_score"]
        X_features = [alignment_scores.reshape(-1, 1)]

        for dim in ["goal_structure", "value_compatibility", "stakes"]:
            le = LabelEncoder()
            vals = [e[dim] if e[dim] else "unknown" for e in merged]
            encoded = le.fit_transform(vals).reshape(-1, 1)
            X_features.append(encoded)
            feature_names.append(dim)

        X = np.hstack(X_features)
        reg = LinearRegression().fit(X, r2_scores)
        pred = reg.predict(X)
        ols_r2 = sklearn_r2_score(r2_scores, pred)
        print(f"  OLS R²: {ols_r2:.4f}")
        for name, coef in zip(feature_names, reg.coef_):
            print(f"  {name:25s}: coef={coef:.6f}")
        print(f"  {'intercept':25s}: {reg.intercept_:.6f}")
    except Exception as e:
        print(f"  OLS failed: {e}")

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(alignment_scores, r2_scores, alpha=0.4, s=20)
    z = np.polyfit(alignment_scores, r2_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(alignment_scores.min(), alignment_scores.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2,
            label=f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e})")
    ax.set_xlabel("Alignment Score")
    ax.set_ylabel("Per-Episode R²")
    ax.set_title(f"{model_tag} — Alignment vs Neural Synchrony (L{layer_A}→L{layer_B})")
    ax.legend()
    fig.tight_layout()
    scatter_path = os.path.join(PLOTS_DIR, f"scatter_{model_tag}_L{layer_A}_L{layer_B}.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {scatter_path}")

    for dim in dimensions:
        groups = defaultdict(list)
        for e in merged:
            if e[dim] is not None:
                groups[e[dim]].append(e["r2_mean"])
        if not groups:
            continue

        cats = sorted(groups.keys())
        data = [groups[c] for c in cats]
        labels_text = [f"{c}\n(n={len(groups[c])})" for c in cats]

        fig, ax = plt.subplots(figsize=(max(6, len(cats) * 1.5), 5))
        bp = ax.boxplot(data, labels=labels_text, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set_ylabel("Per-Episode R²")
        ax.set_title(f"{model_tag} — R² by {dim} (L{layer_A}→L{layer_B})")
        fig.tight_layout()
        box_path = os.path.join(PLOTS_DIR, f"box_{dim}_{model_tag}_L{layer_A}_L{layer_B}.png")
        fig.savefig(box_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {box_path}")

    if "goal_structure" in group_stats:
        gs = group_stats["goal_structure"]
        cats = sorted(gs.keys())
        means = [gs[c]["mean"] for c in cats]
        stds = [gs[c]["std"] for c in cats]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(cats))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("Mean Per-Episode R²")
        ax.set_title(f"{model_tag} — Mean R² by Goal Structure (L{layer_A}→L{layer_B})")
        fig.tight_layout()
        bar_path = os.path.join(PLOTS_DIR, f"bar_goal_structure_{model_tag}_L{layer_A}_L{layer_B}.png")
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {bar_path}")

    return {
        "model": model_tag,
        "layer_A": layer_A,
        "layer_B": layer_B,
        "global_r2_mean": float(np.mean(global_r2_per_seed)),
        "n_episodes": len(merged),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "group_stats": group_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_MISTRAL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]


def run_single_model(model_name, layer_A=None, layer_B=None, setting_str="A_forward"):
    setting = (setting_str.split("_")[0], setting_str.split("_")[1])

    if layer_A is None or layer_B is None:
        (layer_A, layer_B), _ = find_best_layer_pair(model_name, setting_str)

    print(f"\n{'#'*60}")
    print(f"Model: {model_name}")
    print(f"Layer pair: A={layer_A}, B={layer_B}")
    print(f"{'#'*60}")

    per_episode_r2, flat_to_combo, global_r2_per_seed = compute_per_episode_r2(
        model_name, layer_A, layer_B,
        setting=setting, seeds=(0, 1, 2),
        seed_list=[0, 1, 2, 3, 4],
    )

    merged = merge_with_labels(per_episode_r2)

    short_names, _ = get_short_names_and_identifiers([model_name])
    model_tag = short_names[0]
    output_path = os.path.join(
        SCRIPT_DIR, f"alignment_r2_analysis_{model_tag}_L{layer_A}_L{layer_B}.json"
    )
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged data: {output_path}")

    summary = run_analysis(merged, model_name, layer_A, layer_B, global_r2_per_seed)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Per-episode R² vs alignment analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Model pair string (e.g. Mistral-7B-Instruct-v0.3_None_0_...)")
    parser.add_argument("--layer_A", type=int, default=None)
    parser.add_argument("--layer_B", type=int, default=None)
    parser.add_argument("--setting", type=str, default="A_forward")
    parser.add_argument("--all_pairs", action="store_true",
                        help="Run for all 4 Mistral pairs")
    args = parser.parse_args()

    if args.all_pairs:
        all_summaries = []
        for model in ALL_MISTRAL_PAIRS:
            summary = run_single_model(model, args.layer_A, args.layer_B, args.setting)
            all_summaries.append(summary)

        print(f"\n\n{'='*60}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*60}")
        for s in all_summaries:
            print(f"  {s['model']:20s}  Pearson r={s['pearson_r']:+.4f} (p={s['pearson_p']:.2e})  "
                  f"Spearman r={s['spearman_r']:+.4f} (p={s['spearman_p']:.2e})  "
                  f"n={s['n_episodes']}")

        summary_path = os.path.join(SCRIPT_DIR, "alignment_r2_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nSaved aggregate summary: {summary_path}")

    elif args.model:
        run_single_model(args.model, args.layer_A, args.layer_B, args.setting)
    else:
        run_single_model(ALL_MISTRAL_PAIRS[0], args.layer_A, args.layer_B, args.setting)


if __name__ == "__main__":
    main()
