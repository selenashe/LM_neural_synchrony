#!/usr/bin/env python3
"""
Mia + Ava, false-belief (18-episode) analysis plots (same logic as visualize_analyses_mia_ava_90ep.ipynb).

Sections: A affine heatmaps, A2 SBERT controls, E control summary printout (optional).

Affine JSON names use ``combined_metrics_{short}falsebelieffixedtwoagents{str(STAGE_A_SEEDS)}_layerA...``
(``falsebelieffixedtwoagents`` comes from ``RESULTS_POSTFIX`` with underscores removed).

Section E prefers ``control_analysis_summary_false_belief_18ep.json`` when present; it does not load
``analysis_files/control_analysis_summary.json`` (legacy 450-ep aggregate) before repo-root fallbacks.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiment_config import STAGE_A_SEEDS
from utils import get_short_names_and_identifiers

# Must match `run_affine_and_controls_sbert_false_belief.sh` / simulation postfix.
RESULTS_POSTFIX = "_false_belief_fixed_two_agents"
SBERT_OUTPUT_TAG = "false_belief_18ep"


def _apply_results_postfix(model_name: str, postfix: str) -> str:
    if not postfix or model_name.endswith(postfix):
        return model_name
    return model_name + postfix


# Mistral-only pairs (same as false-belief simulation grid).
ALL_STAGE_A_MODEL_PAIRS = [
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.3_None_0",
    "Mistral-7B-Instruct-v0.2_None_0_Mistral-7B-Instruct-v0.2_None_0",
]

sns.set_theme(style="whitegrid", font_scale=1.1)

AFFINE_BASE = os.path.join(REPO, "affine_transformation")
SAVE_DIR = os.path.join(REPO, "playground", f"viz_outputs_{SBERT_OUTPUT_TAG}")
os.makedirs(SAVE_DIR, exist_ok=True)

AFFINE_TAG_SUFFIX = RESULTS_POSTFIX.replace("_", "")
SEED_BRACKET = str(list(STAGE_A_SEEDS))
N_LAYERS = 32


def sbert_model_key(short_pair_tag: str) -> str:
    """Key used inside the SBERT summary JSON (short tag + postfix suffix, no seed bracket)."""
    return f"{short_pair_tag}{AFFINE_TAG_SUFFIX}"


def short_tag_for_full_model_name(full_pair: str) -> str:
    sn, _ = get_short_names_and_identifiers([full_pair])
    return sn[0]


def affine_models_identifier(short_pair_tag: str) -> str:
    return f"{short_pair_tag}{AFFINE_TAG_SUFFIX}{SEED_BRACKET}"


PAIR_TITLES = [
    "Mis-v0.3 × Mis-v0.3",
    "Mis-v0.3 × Mis-v0.2",
    "Mis-v0.2 × Mis-v0.3",
    "Mis-v0.2 × Mis-v0.2",
]

rows: list[dict] = []
for base, title in zip(ALL_STAGE_A_MODEL_PAIRS, PAIR_TITLES):
    # Short tags and SBERT JSON keys match the *base* pair name only. Appending
    # the postfix before get_short_names_and_identifiers corrupts the tag.
    st = short_tag_for_full_model_name(base)
    full = _apply_results_postfix(base, RESULTS_POSTFIX)
    rows.append(
        {
            "short": st,
            "affine_id": affine_models_identifier(st),
            "full": full,
            "title": title,
        }
    )


def subplots_2col(n_panels, row_h=5.0, col_w=7.0, **kwargs):
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(col_w * ncols, row_h * nrows),
        squeeze=False,
        **kwargs,
    )
    axes_flat = np.array(axes).reshape(-1)
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)
    return fig, axes_flat[:n_panels].tolist(), nrows, ncols


def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def load_affine_heatmap(affine_id, setting="A_forward", data_mode="combined_metrics",
                        metric_key="r2_mean"):
    """Load 32x32 metric matrix from per-layer JSON files.

    Args:
        metric_key: JSON key to read per cell. Default ``"r2_mean"`` (CKA value
            when method is cka_cca). Use ``"cka_help"`` or ``"cka_deceive"`` for
            per-goal breakdowns.

    Returns (matrix, method) where method is 'cka_cca' or 'affine'.
    """
    mat = np.full((N_LAYERS, N_LAYERS), np.nan)
    detected_method = "affine"
    json_key = affine_id.split("[")[0] if "[" in affine_id else affine_id
    base = os.path.join(AFFINE_BASE, setting)
    for la in range(N_LAYERS):
        data_dir = os.path.join(base, f"layerA{la}", "data")
        if not os.path.isdir(data_dir):
            continue
        for lb in range(N_LAYERS):
            fn = os.path.join(
                data_dir,
                f"{data_mode}_{affine_id}_layerA{la}_layerB{lb}_allreps.json",
            )
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                d = json.load(f)
            v = d.get(json_key)
            if not isinstance(v, dict) or metric_key not in v:
                continue
            if v.get("method") == "cka_cca":
                detected_method = "cka_cca"
            vals = v[metric_key]
            mat[la, lb] = vals[0] if isinstance(vals, list) else vals
    return mat, detected_method


def plot_affine_heatmaps():
    n = len(rows)
    fig, axes, _, _ = subplots_2col(n, row_h=5.5, col_w=6.5)
    vmin, vmax = 0, 1.0
    im = None
    detected_method = "affine"
    for ax, row in zip(axes, rows):
        aid = row["affine_id"]
        r2, method = load_affine_heatmap(aid)
        if method == "cka_cca":
            detected_method = method
        im = ax.imshow(
            r2,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        metric_label = "CKA" if detected_method == "cka_cca" else "R²"
        if np.any(np.isfinite(r2)):
            best_idx = np.unravel_index(np.nanargmax(r2), r2.shape)
            ax.scatter(
                best_idx[1],
                best_idx[0],
                marker="*",
                s=200,
                c="red",
                zorder=5,
                label=f"Best: L{best_idx[0]}→L{best_idx[1]}  {metric_label}={r2[best_idx]:.3f}",
            )
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax.text(
                0.5,
                0.5,
                f"No {metric_label} data\n(check affine_transformation/ and affine_id)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        ax.set_xlabel("Layer B")
        ax.set_ylabel("Layer A")
        ax.set_title(row["title"])

    metric_label = "CKA" if detected_method == "cka_cca" else "R²"
    fig.suptitle(
        f"Layer-pair {metric_label} (18 ep false-belief, Mia+Ava)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.98])
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.82,
        label=metric_label,
        location="right",
        pad=0.02,
        fraction=0.035,
    )
    out = os.path.join(SAVE_DIR, "A1_layer_heatmap_false_belief_18ep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_affine_heatmaps_by_goal():
    """4 rows (model pairs) x 3 columns (All, Help, Deceive) CKA heatmap grid."""
    goal_keys = [("r2_mean", "All"), ("cka_help", "Help only"), ("cka_deceive", "Deceive only")]
    n_rows = len(rows)
    n_cols = len(goal_keys)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5.0 * n_rows),
        squeeze=False,
    )
    vmin, vmax = 0, 1.0
    im = None
    for i, row in enumerate(rows):
        aid = row["affine_id"]
        for j, (mk, col_title) in enumerate(goal_keys):
            ax = axes[i][j]
            mat, _ = load_affine_heatmap(aid, metric_key=mk)
            im = ax.imshow(
                mat, origin="lower", cmap="viridis",
                vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
            )
            if np.any(np.isfinite(mat)):
                best_idx = np.unravel_index(np.nanargmax(mat), mat.shape)
                ax.scatter(
                    best_idx[1], best_idx[0], marker="*", s=160, c="red", zorder=5,
                    label=f"L{best_idx[0]}→L{best_idx[1]} {mat[best_idx]:.3f}",
                )
                ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
            if i == 0:
                ax.set_title(col_title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{row['title']}\nLayer A", fontsize=9)
            else:
                ax.set_ylabel("")
            if i == n_rows - 1:
                ax.set_xlabel("Layer B")

    fig.suptitle("CKA by goal condition (18 ep false-belief, Mia+Ava)", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.98])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="CKA",
                     location="right", pad=0.02, fraction=0.025)
    out = os.path.join(SAVE_DIR, "A1b_layer_heatmap_by_goal_false_belief_18ep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_sbert_controls():
    sbert_path = first_existing(
        [
            os.path.join(REPO, f"control_analysis_sbert_summary_{SBERT_OUTPUT_TAG}.json"),
            os.path.join(REPO, "analysis_files", f"control_analysis_sbert_summary_{SBERT_OUTPUT_TAG}.json"),
            os.path.join(REPO, "control_analysis_sbert_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "analysis_files", "control_analysis_sbert_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "control_analysis_sbert_summary.json"),
            os.path.join(REPO, "analysis_files", "control_analysis_sbert_summary.json"),
        ]
    )
    if sbert_path is None:
        raise FileNotFoundError(
            "No SBERT summary found. Run: "
            "python analyze_controls_sbert.py --all_pairs --mistral_only_pairs "
            "--results_postfix _false_belief_fixed_two_agents --output_tag false_belief_18ep"
        )
    print(f"SBERT summary: {sbert_path}")
    with open(sbert_path) as f:
        sbert_by_model = {e["model"]: e for e in json.load(f)}

    detected_method = "affine"
    short_tags = [r["short"] for r in rows]
    best_hidden, best_help, best_deceive = [], [], []
    sbert_align, shuf_r2 = [], []
    ct0, ct1, ct5 = [], [], []
    for tag in short_tags:
        aid = affine_models_identifier(tag)
        r2, method = load_affine_heatmap(aid)
        if method == "cka_cca":
            detected_method = method
        best_hidden.append(float(np.nanmax(r2)) if np.any(np.isfinite(r2)) else float("nan"))

        r2_h, _ = load_affine_heatmap(aid, metric_key="cka_help")
        best_help.append(float(np.nanmax(r2_h)) if np.any(np.isfinite(r2_h)) else float("nan"))

        r2_d, _ = load_affine_heatmap(aid, metric_key="cka_deceive")
        best_deceive.append(float(np.nanmax(r2_d)) if np.any(np.isfinite(r2_d)) else float("nan"))

    metric_label = "CKA" if detected_method == "cka_cca" else "R²"
    bar_labels = [
        f"Best {metric_label} (all)",
        f"Best {metric_label} (help)",
        f"Best {metric_label} (deceive)",
        "SBERT (aligned)",
        "Ep. shuffle",
        r"$t$",
        r"$t+1$",
        r"$t+5$",
    ]
    bar_colors = [
        "#E6C200", "#4CAF50", "#F44336",
        "#FF8C00", "#7C3AED", "#1E3A8A", "#3B82F6", "#93C5FD",
    ]

    for tag in short_tags:
        skey = sbert_model_key(tag)
        e = sbert_by_model.get(skey) or sbert_by_model.get(tag)
        if e is None:
            sbert_align.append(float("nan"))
            shuf_r2.append(float("nan"))
            ct0.append(float("nan"))
            ct1.append(float("nan"))
            ct5.append(float("nan"))
            continue
        sbert_align.append(e["sbert"]["real_r2_mean"])
        shuf_r2.append(e["shuffle_episode_text"]["shuffled_r2_mean"])
        ct = e["cross_turn"]
        ct0.append(ct["offset_0"]["real_r2_mean"])
        ct1.append(ct["offset_1"]["real_r2_mean"])
        ct5.append(ct["offset_5"]["real_r2_mean"])

    series = [best_hidden, best_help, best_deceive, sbert_align, shuf_r2, ct0, ct1, ct5]
    n_models = len(short_tags)
    n_bars = len(series)
    x = np.arange(n_models, dtype=float)
    width = 0.09
    offsets = x[:, None] + (np.arange(n_bars) - (n_bars - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(14, 4.0))
    for j in range(n_bars):
        ax.bar(
            offsets[:, j],
            [series[j][i] for i in range(n_models)],
            width,
            label=bar_labels[j],
            color=bar_colors[j],
            edgecolor="0.25",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["title"] for r in rows],
        rotation=22,
        ha="right",
    )
    ax.set_ylabel(f"Test {metric_label}" if detected_method == "cka_cca" else "Test R²")
    ax.set_ylim(0, 1.05 if detected_method == "cka_cca" else 0.82)
    ax.set_title(f"Hidden (best layer {metric_label}) vs. SBERT controls — false-belief 18 ep")
    ax.legend(
        fontsize=6.5,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.92,
    )
    fig.tight_layout(rect=[0, 0, 0.74, 1])
    out = os.path.join(SAVE_DIR, "A2_sbert_controls_grouped_false_belief_18ep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_controls_summary():
    """Print a summary table from the SBERT and CKA results already loaded."""
    sbert_path = first_existing(
        [
            os.path.join(REPO, f"control_analysis_sbert_summary_{SBERT_OUTPUT_TAG}.json"),
            os.path.join(REPO, "analysis_files", f"control_analysis_sbert_summary_{SBERT_OUTPUT_TAG}.json"),
        ]
    )
    if sbert_path is None:
        print("\n--- Section E skipped: no SBERT summary found.\n")
        return
    with open(sbert_path) as f:
        sbert_data = {e["model"]: e for e in json.load(f)}

    print(f"\n{'='*80}")
    print("Section E — Controls summary (false-belief 18 ep)")
    print(f"{'='*80}")
    header = f"{'Pair':<20} {'Best CKA':>10} {'SBERT':>10} {'Shuf':>10} {'t+0':>10} {'t+1':>10} {'t+5':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        aid = affine_models_identifier(r["short"])
        r2, _ = load_affine_heatmap(aid)
        best_cka = float(np.nanmax(r2)) if np.any(np.isfinite(r2)) else float("nan")
        skey = sbert_model_key(r["short"])
        e = sbert_data.get(skey) or sbert_data.get(r["short"])
        if e is None:
            print(f"  {r['title']:<20}  (no SBERT entry)")
            continue
        sbert_r2 = e["sbert"]["real_r2_mean"]
        shuf_r2 = e["shuffle_episode_text"]["shuffled_r2_mean"]
        ct = e["cross_turn"]
        ct0 = ct["offset_0"]["real_r2_mean"]
        ct1 = ct["offset_1"]["real_r2_mean"]
        ct5 = ct["offset_5"]["real_r2_mean"]
        print(f"{r['title']:<20} {best_cka:10.4f} {sbert_r2:10.4f} {shuf_r2:10.4f} "
              f"{ct0:10.4f} {ct1:10.4f} {ct5:10.4f}")
    print()


def main():
    print(
        f"Results postfix: {RESULTS_POSTFIX!r} → affine tag {AFFINE_TAG_SUFFIX!r}, "
        f"seeds {SEED_BRACKET}; SBERT summary tag {SBERT_OUTPUT_TAG!r}"
    )
    print(f"Built {len(rows)} model rows (short → affine file id)")
    for r in rows:
        print(f"  {r['short']!r}  →  {r['affine_id']!r}")

    plot_affine_heatmaps()
    plot_affine_heatmaps_by_goal()
    plot_sbert_controls()
    print_controls_summary()


if __name__ == "__main__":
    main()
