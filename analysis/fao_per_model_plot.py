#!/usr/bin/env python3
"""
Per-model-pair Pareto-gap visualizations.

For each model pair, produces a 2-panel figure:
  (1) Normalized efficiency scatter: achieved/max ratio per agreed episode.
  (2) Disagreement bar: count of agreed vs. disagreed episodes.

Reuses scenario loading, episode parsing, and metadata from fao_baseline.py.
Run from the same project root as fao_baseline.py:
    python analysis/plot_pareto_gaps.py
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from the existing analysis script.
# Assumes this file lives next to fao_baseline.py in analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fao_analysis import (  # noqa: E402
    load_scenarios,
    compute_scenario_metadata,
    load_llm_episodes,
    parse_episode,
    score_allocation,
    extract_model_name,
    MODEL_DISPLAY,
    MODEL_COLORS,
    PROJECT_ROOT,
)

OUTPUT_DIR = PROJECT_ROOT / 'summary_plots_deal_or_no_deal' / 'per_pair_pareto_gaps'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def _display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def collect_episode_outcomes(episodes, scenarios, scenario_meta):
    """For each model pair, build a list of per-episode outcome dicts.

    Each dict has:
        scenario_idx, status ('agreed' | 'disagreed' | 'parse_fail'),
        joint_score, max_joint, efficiency, gap, is_pareto_optimal
    """
    pair_outcomes = defaultdict(list)

    for pair_name, pair_episodes in episodes.items():
        for sc_idx, dialog in pair_episodes:
            sc = scenarios[sc_idx]
            meta = scenario_meta[sc_idx]
            max_joint = meta['max_joint']

            alloc_a, atype, _ = parse_episode(dialog, sc)

            if atype == 'parse_failure':
                pair_outcomes[pair_name].append({
                    'scenario_idx': sc_idx,
                    'status': 'parse_fail',
                    'joint_score': None,
                    'max_joint': max_joint,
                    'efficiency': None,
                    'gap': None,
                    'is_pareto_optimal': False,
                })
                continue

            if atype in ('leave', 'implicit'):
                sa, sb = score_allocation(alloc_a, sc)
                joint = sa + sb
                gap = max_joint - joint
                eff = joint / max_joint if max_joint > 0 else 0.0
                # Pareto-optimal iff allocation is in this scenario's Pareto set
                is_pareto = any(
                    all(alloc_a[t] == pa[t] for t in sc.item_types)
                    for pa, _, _ in meta['pareto']
                )
                pair_outcomes[pair_name].append({
                    'scenario_idx': sc_idx,
                    'status': 'agreed',
                    'joint_score': joint,
                    'max_joint': max_joint,
                    'efficiency': eff,
                    'gap': gap,
                    'is_pareto_optimal': is_pareto,
                })
            else:
                pair_outcomes[pair_name].append({
                    'scenario_idx': sc_idx,
                    'status': 'disagreed',
                    'joint_score': 0,
                    'max_joint': max_joint,
                    'efficiency': 0.0,
                    'gap': max_joint,
                    'is_pareto_optimal': False,
                })

    return pair_outcomes


def _jittered(values, scale=0.08, seed=None):
    """Add small uniform jitter for visibility of overlapping integer points."""
    rng = np.random.default_rng(seed)
    return np.array(values) + rng.uniform(-scale, scale, size=len(values))


def plot_pair_gaps(pair_name, outcomes, output_path):
    model = extract_model_name(pair_name)
    color = MODEL_COLORS.get(model, '#444444')
    display = _display_name(model)

    agreed = [o for o in outcomes if o['status'] == 'agreed']
    disagreed = [o for o in outcomes if o['status'] == 'disagreed']
    parse_fail = [o for o in outcomes if o['status'] == 'parse_fail']

    n_total = len(outcomes)
    n_agreed = len(agreed)
    n_disagreed = len(disagreed)
    n_parse_fail = len(parse_fail)
    n_pareto = sum(1 for o in agreed if o['is_pareto_optimal'])

    mean_gap = np.mean([o['gap'] for o in agreed]) if agreed else float('nan')
    mean_eff = np.mean([o['efficiency'] for o in agreed]) if agreed else float('nan')

    # 2-panel layout: efficiency scatter, outcome bar. Bar panel narrower.
    fig, (ax_eff, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={'width_ratios': [1, 0.4]},
    )

    lim_max = max((o['max_joint'] for o in outcomes), default=20) + 1

    # --- Panel 1: normalized efficiency scatter ---
    if agreed:
        max_joints = [o['max_joint'] for o in agreed]
        effs = [o['efficiency'] for o in agreed]
        is_opt = np.array([o['is_pareto_optimal'] for o in agreed])

        x = _jittered(max_joints, seed=2)
        y = _jittered(effs, scale=0.012, seed=3)

        if is_opt.any():
            ax_eff.scatter(x[is_opt], y[is_opt], c=color, s=55, alpha=0.75,
                           edgecolors='black', linewidths=0.4,
                           label=f'Pareto-optimal (n={int(is_opt.sum())})', zorder=3)
        if (~is_opt).any():
            ax_eff.scatter(x[~is_opt], y[~is_opt], facecolors='none',
                           edgecolors=color, s=55, linewidths=1.3,
                           label=f'Agreed but suboptimal (n={int((~is_opt).sum())})', zorder=2)

    ax_eff.axhline(1.0, color='k', linestyle='--', alpha=0.35, lw=1, label='Pareto-optimal (eff = 1)')
    if agreed:
        ax_eff.axhline(mean_eff, color=color, linestyle=':', alpha=0.7, lw=1.2,
                       label=f'Mean efficiency = {mean_eff:.2f}')
    ax_eff.set_xlim(0, lim_max)
    ax_eff.set_ylim(-0.02, 1.08)
    ax_eff.set_xlabel('Max joint score for scenario', fontsize=11)
    ax_eff.set_ylabel('Efficiency (achieved / max)', fontsize=11)
    ax_eff.set_title('Normalized efficiency by scenario difficulty', fontsize=12)
    ax_eff.grid(True, alpha=0.25)
    ax_eff.legend(loc='lower right', fontsize=8)

    # --- Panel 3: episode outcome bar ---
    counts = [n_agreed, n_disagreed, n_parse_fail]
    labels = ['Agreed', 'Disagreed', 'Parse fail']
    colors_bar = [color, '#888888', '#cccccc']
    bars = ax_bar.bar(labels, counts, color=colors_bar, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{count}', ha='center', va='bottom', fontsize=10)
    ax_bar.set_ylabel('Episode count', fontsize=11)
    ax_bar.set_title('Outcome breakdown', fontsize=12)
    ax_bar.set_ylim(0, max(counts) * 1.18 if counts and max(counts) > 0 else 1)
    ax_bar.grid(True, alpha=0.25, axis='y')
    ax_bar.tick_params(axis='x', labelsize=10)

    # --- Suptitle with summary stats ---
    agree_rate = n_agreed / n_total if n_total else 0
    pareto_rate = n_pareto / n_agreed if n_agreed else 0
    fig.suptitle(
        f'{display}   '
        f'episodes={n_total}   agreement={agree_rate:.0%}   '
        f'mean efficiency (agreed)={mean_eff:.2f}   '
        f'mean gap (agreed)={mean_gap:.2f}   '
        f'Pareto-optimal rate (agreed)={pareto_rate:.0%}',
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading scenarios...")
    scenarios = load_scenarios()
    log.info(f"  {len(scenarios)} scenarios")

    log.info("Computing scenario metadata (Pareto frontiers)...")
    scenario_meta = [compute_scenario_metadata(sc) for sc in scenarios]

    log.info("Loading LLM episodes...")
    episodes = load_llm_episodes(scenarios)
    log.info(f"  {len(episodes)} model pairs")

    log.info("Computing per-episode outcomes...")
    pair_outcomes = collect_episode_outcomes(episodes, scenarios, scenario_meta)

    log.info("Generating per-pair gap plots...")
    for pair_name, outcomes in sorted(pair_outcomes.items()):
        model = extract_model_name(pair_name)
        safe = model.replace('/', '_').replace(' ', '_')
        out_path = OUTPUT_DIR / f'{safe}.png'
        plot_pair_gaps(pair_name, outcomes, out_path)
        log.info(f"  {_display_name(model):30s} -> {out_path.name}")

    log.info(f"All plots saved to {OUTPUT_DIR}")
    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()