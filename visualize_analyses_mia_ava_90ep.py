#!/usr/bin/env python3
"""
Mia + Ava, 90-episode analysis plots (same logic as visualize_analyses_mia_ava_90ep.ipynb).

Sections: A affine heatmaps, A2 SBERT controls, E control summary printout (optional).

Affine JSON names use ``combined_metrics_{short}miaavafixedtwoagents{str(STAGE_A_SEEDS)}_layerA...``
(``miaavafixedtwoagents`` comes from ``MIA_AVA_RESULTS_POSTFIX`` with underscores removed).

Section E does **not** load ``analysis_files/control_analysis_summary.json`` (legacy 450-ep aggregate);
it looks for a Mia/Ava-tagged summary first, then ``control_analysis_summary.json`` at repo root only.
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

from experiment_config import MIA_AVA_RESULTS_POSTFIX, STAGE_A_SEEDS
from utils import get_short_names_and_identifiers


def _apply_results_postfix(model_name: str, postfix: str) -> str:
    if not postfix or model_name.endswith(postfix):
        return model_name
    return model_name + postfix


# Same order as analyze_controls_sbert.ALL_STAGE_A_MODEL_PAIRS (avoid importing
# analyze_controls_sbert → analyze_controls, which may be absent in this checkout).
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

sns.set_theme(style="whitegrid", font_scale=1.1)

AFFINE_BASE = os.path.join(REPO, "affine_transformation")
SAVE_DIR = os.path.join(REPO, "playground", "viz_outputs_mia_ava_90ep")
os.makedirs(SAVE_DIR, exist_ok=True)

AFFINE_TAG_SUFFIX = MIA_AVA_RESULTS_POSTFIX.replace("_", "")
SEED_BRACKET = str(list(STAGE_A_SEEDS))
N_LAYERS = 32


def short_tag_for_full_model_name(full_pair: str) -> str:
    sn, _ = get_short_names_and_identifiers([full_pair])
    return sn[0]


def affine_models_identifier(short_pair_tag: str) -> str:
    return f"{short_pair_tag}{AFFINE_TAG_SUFFIX}{SEED_BRACKET}"


PAIR_TITLES = [
    "Mis-v0.3 × Mis-v0.3",
    "Mis-v0.3 × Mis-v0.2",
    "Mis-v0.3 × Llama-3-8B",
    "Mis-v0.2 × Mis-v0.3",
    "Mis-v0.2 × Mis-v0.2",
    "Mis-v0.2 × Llama-3-8B",
    "Llama-3-8B × Mis-v0.3",
    "Llama-3-8B × Mis-v0.2",
    "Llama-3-8B × Llama-3-8B",
]

rows: list[dict] = []
for base, title in zip(ALL_STAGE_A_MODEL_PAIRS, PAIR_TITLES):
    # Short tags and SBERT JSON keys match the *base* pair name only. Appending
    # _mia_ava_fixed_two_agents before get_short_names_and_identifiers corrupts
    # the tag (duplicates miaavafixedtwoagents in affine filenames).
    st = short_tag_for_full_model_name(base)
    full = _apply_results_postfix(base, MIA_AVA_RESULTS_POSTFIX)
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


def load_affine_heatmap(affine_id, setting="A_forward", data_mode="combined_metrics"):
    """Load 32×32 R² matrix from per-layer JSON files (Mia+Ava affine tag)."""
    r2 = np.full((N_LAYERS, N_LAYERS), np.nan)
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
            if not isinstance(v, dict) or "r2_mean" not in v:
                for _, v2 in d.items():
                    if isinstance(v2, dict) and "r2_mean" in v2:
                        v = v2
                        break
                else:
                    continue
            vals = v["r2_mean"]
            r2[la, lb] = vals[0] if isinstance(vals, list) else vals
    return r2


def plot_affine_heatmaps():
    n = len(rows)
    fig, axes, _, _ = subplots_2col(n, row_h=5.5, col_w=6.5)
    vmin, vmax = 0, 0.75
    im = None
    for ax, row in zip(axes, rows):
        aid = row["affine_id"]
        r2 = load_affine_heatmap(aid)
        im = ax.imshow(
            r2,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        if np.any(np.isfinite(r2)):
            best_idx = np.unravel_index(np.nanargmax(r2), r2.shape)
            ax.scatter(
                best_idx[1],
                best_idx[0],
                marker="*",
                s=200,
                c="red",
                zorder=5,
                label=f"Best: L{best_idx[0]}→L{best_idx[1]}  R²={r2[best_idx]:.3f}",
            )
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax.text(
                0.5,
                0.5,
                "No R² data\n(check affine_transformation/ and affine_id)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        ax.set_xlabel("Layer B")
        ax.set_ylabel("Layer A")
        ax.set_title(row["title"])

    fig.suptitle(
        "Affine transformation test R² (90 ep, Mia+Ava): layer pairs",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.98])
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.82,
        label="Test R²",
        location="right",
        pad=0.02,
        fraction=0.035,
    )
    out = os.path.join(SAVE_DIR, "A1_layer_heatmap_mia_ava_90ep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_sbert_controls():
    sbert_path = first_existing(
        [
            os.path.join(REPO, "control_analysis_sbert_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "analysis_files", "control_analysis_sbert_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "control_analysis_sbert_summary.json"),
            os.path.join(REPO, "analysis_files", "control_analysis_sbert_summary.json"),
        ]
    )
    if sbert_path is None:
        raise FileNotFoundError(
            "No SBERT summary found. Run: "
            "python analyze_controls_sbert.py --all_pairs --output_tag mia_ava_90ep"
        )
    print(f"SBERT summary: {sbert_path}")
    with open(sbert_path) as f:
        sbert_by_model = {e["model"]: e for e in json.load(f)}

    bar_labels = [
        "Best layer R²",
        "SBERT (aligned)",
        "Ep. shuffle",
        r"$t$",
        r"$t+1$",
        r"$t+5$",
    ]
    bar_colors = ["#E6C200", "#FF8C00", "#7C3AED", "#1E3A8A", "#3B82F6", "#93C5FD"]

    short_tags = [r["short"] for r in rows]
    best_hidden, sbert_align, shuf_r2 = [], [], []
    ct0, ct1, ct5 = [], [], []
    for tag in short_tags:
        r2 = load_affine_heatmap(affine_models_identifier(tag))
        if np.any(np.isfinite(r2)):
            best_hidden.append(float(np.nanmax(r2)))
        else:
            best_hidden.append(float("nan"))
        e = sbert_by_model.get(tag)
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

    series = [best_hidden, sbert_align, shuf_r2, ct0, ct1, ct5]
    n_models = len(short_tags)
    n_bars = len(series)
    x = np.arange(n_models, dtype=float)
    width = 0.11
    offsets = x[:, None] + (np.arange(n_bars) - (n_bars - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(12, 3.8))
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
    ax.set_ylabel("Test R²")
    ax.set_ylim(0, 0.82)
    ax.set_title("Hidden (best layer) vs. SBERT controls — Mia/Ava 90 ep")
    ax.legend(
        fontsize=7,
        ncol=3,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.92,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out = os.path.join(SAVE_DIR, "A2_sbert_controls_grouped_mia_ava_90ep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_controls_summary():
    """Prefer 90-ep Mia/Ava controls aggregate; never silently use legacy analysis_files copy."""
    ctrl_path = first_existing(
        [
            os.path.join(REPO, "control_analysis_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "analysis_files", "control_analysis_summary_mia_ava_90ep.json"),
            os.path.join(REPO, "control_analysis_summary.json"),
        ]
    )
    if ctrl_path is None:
        print(
            "\n--- Section E skipped: no control summary at repo root "
            "(expected control_analysis_summary_mia_ava_90ep.json or control_analysis_summary.json). "
            "Not using analysis_files/control_analysis_summary.json (450-ep legacy).\n"
        )
        return
    print(f"Controls summary: {ctrl_path}")
    with open(ctrl_path) as f:
        controls_data = json.load(f)

    short_tags = [r["short"] for r in rows]
    by_short = {e["model"]: e for e in controls_data}
    print(f"Loaded {len(controls_data)} entries in file; Mia/Ava short tags:")
    for tag in short_tags:
        ent = by_short.get(tag)
        if ent is None:
            print(f"  (missing) {tag}")
        else:
            print(f"  {tag}  L{ent['layer_A']}→L{ent['layer_B']}")


def main():
    print(
        f"MIA_AVA postfix: {MIA_AVA_RESULTS_POSTFIX!r} → affine tag {AFFINE_TAG_SUFFIX!r}, "
        f"seeds {SEED_BRACKET}"
    )
    print(f"Built {len(rows)} model rows (short → affine file id)")
    for r in rows:
        print(f"  {r['short']!r}  →  {r['affine_id']!r}")

    plot_affine_heatmaps()
    plot_sbert_controls()
    print_controls_summary()


if __name__ == "__main__":
    main()
