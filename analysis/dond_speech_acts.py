#!/usr/bin/env python3
"""
Keyword/pattern-based speech act tagger for Deal-or-No-Deal negotiation dialogs.

Tags each turn with one or more negotiation speech acts, aggregates by model,
and correlates with outcomes.
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import urllib.request

from fao_analysis import (
    load_scenarios, RESULT_DIRS, _load_csv_dialog, _get_speaker, _get_utterance,
    parse_episode, score_allocation, compute_scenario_metadata, extract_model_name,
    parse_items_from_text, PROJECT_ROOT, Scenario, MODEL_DISPLAY,
    _parse_turn_allocation, HUMAN_DATA_URL, find_effective_end,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SPEECH_ACTS = [
    'propose', 'counter_propose', 'accept', 'reject',
    'inquire', 'reveal_preference', 'concede', 'insist',
    'leave', 'small_talk',
]

ACCEPT_PATTERNS = [
    'okay', 'deal', 'agreed', 'sounds good', 'i accept',
    'that works', "let's do it", 'fine by me',
]

REJECT_PATTERNS = [
    "i can't accept", 'no deal', "that doesn't work",
    'i reject', 'not acceptable', 'i disagree',
]

INQUIRE_PATTERNS = [
    'what do you', 'which items', 'what would you',
    'what about', 'how about',
]

PREFERENCE_PATTERNS = [
    'i really need', 'i value', 'important to me',
    'worth more to me', "i don't need", "i don't care about",
]


def _allocs_equal(a: Dict[str, int], b: Dict[str, int]) -> bool:
    if a is None or b is None:
        return False
    return all(a.get(k, 0) == b.get(k, 0) for k in set(a) | set(b))


def _speaker_score(alloc_a: Dict[str, int], scenario: Scenario, speaker_is_a: bool) -> float:
    if speaker_is_a:
        return sum(alloc_a[t] * scenario.values_a[t] for t in scenario.item_types)
    else:
        return sum((scenario.item_counts[t] - alloc_a[t]) * scenario.values_b[t]
                   for t in scenario.item_types)


def tag_speech_acts(turns: List[Dict], scenario: Scenario) -> List[Dict]:
    """Returns list of dicts: {turn, speaker, text, acts, parsed_alloc, speaker_score}"""
    last_alloc_by_speaker = {}
    last_score_by_speaker = {}
    last_alloc_by_other = {}
    results = []

    for t in turns:
        turn_idx = t['turn']
        speaker = t['speaker']
        text = t['text']
        text_lower = text.lower()
        speaker_is_a = speaker is not None and 'Mia' in speaker
        other_speaker = None
        for s in last_alloc_by_speaker:
            if s != speaker:
                other_speaker = s
                break

        acts = []
        parsed_alloc = _parse_turn_allocation(text, scenario, speaker_is_a)
        sp_score = None

        has_leave = bool(re.search(r'\bLEAVE\b', text, re.IGNORECASE))

        if parsed_alloc is not None:
            sp_score = _speaker_score(parsed_alloc, scenario, speaker_is_a)

            others_last = last_alloc_by_speaker.get(other_speaker) if other_speaker else None

            if others_last is not None:
                if _allocs_equal(parsed_alloc, others_last):
                    acts.append('accept')
                else:
                    acts.append('counter_propose')
            else:
                acts.append('propose')

            own_last = last_alloc_by_speaker.get(speaker)
            own_last_score = last_score_by_speaker.get(speaker)

            if own_last is not None and own_last_score is not None:
                if sp_score < own_last_score:
                    acts.append('concede')
                elif _allocs_equal(parsed_alloc, own_last):
                    if others_last is not None and not _allocs_equal(others_last, own_last):
                        acts.append('insist')

            if has_leave:
                acts.append('leave')

            last_alloc_by_speaker[speaker] = parsed_alloc
            last_score_by_speaker[speaker] = sp_score
            if other_speaker:
                last_alloc_by_other[other_speaker] = parsed_alloc
            for s in last_alloc_by_speaker:
                if s != speaker:
                    last_alloc_by_other[s] = parsed_alloc

        else:
            if any(p in text_lower for p in ACCEPT_PATTERNS):
                acts.append('accept')
            elif any(p in text_lower for p in REJECT_PATTERNS):
                acts.append('reject')

            if '?' in text and any(p in text_lower for p in INQUIRE_PATTERNS):
                acts.append('inquire')

            if any(p in text_lower for p in PREFERENCE_PATTERNS):
                acts.append('reveal_preference')

            if has_leave:
                acts.append('leave')

            if not acts:
                acts.append('small_talk')

        results.append({
            'turn': turn_idx,
            'speaker': speaker,
            'text': text,
            'acts': acts,
            'parsed_alloc': parsed_alloc,
            'speaker_score': sp_score,
        })

    return results


def load_all_episodes(scenarios):
    """Walk RESULT_DIRS and yield (model, scenario_idx, turns_list) for each episode."""
    all_results = []
    for result_dir in RESULT_DIRS:
        if not result_dir.exists():
            log.warning(f"Result dir not found: {result_dir}")
            continue
        for pair_dir in sorted(result_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            model = extract_model_name(pair_dir.name)
            for csv_file in sorted(pair_dir.glob('*.csv')):
                m = re.match(r'(\d+)_temp', csv_file.name)
                if not m:
                    continue
                sc_idx = int(m.group(1))
                if sc_idx >= len(scenarios):
                    continue
                try:
                    dialog_turns = _load_csv_dialog(csv_file)
                except Exception as e:
                    log.warning(f"Failed to load {csv_file}: {e}")
                    continue

                sc = scenarios[sc_idx]
                eff_end = find_effective_end(dialog_turns, sc)
                effective_turns = dialog_turns[:eff_end]

                turns = []
                for i, raw_turn in enumerate(effective_turns):
                    speaker = _get_speaker(raw_turn)
                    text = _get_utterance(raw_turn)
                    turns.append({'turn': i, 'speaker': speaker, 'text': text})

                tagged = tag_speech_acts(turns, sc)

                alloc_a, atype, _ = parse_episode(dialog_turns, sc)
                agreed = atype in ('leave', 'implicit')
                joint_score = sum(score_allocation(alloc_a, sc)) if alloc_a else 0.0

                all_results.append({
                    'model': model,
                    'scenario_idx': sc_idx,
                    'tagged_turns': tagged,
                    'agreed': agreed,
                    'joint_score': joint_score,
                })
    return all_results


def load_human_episodes(scenarios):
    """Load Lewis et al. 2017 human dialogs, parse turns, and tag speech acts."""
    text = urllib.request.urlopen(HUMAN_DATA_URL, timeout=30).read().decode('utf-8')
    lines = text.strip().splitlines()

    def parse_line(line):
        def grab(tag):
            m = re.search(rf'<{tag}>(.*?)</{tag}>', line)
            return m.group(1).strip() if m else ''
        def to_cv(s):
            nums = list(map(int, s.split()))
            return nums[0::2], nums[1::2]
        counts, vals = to_cv(grab('input'))
        _, partner_vals = to_cv(grab('partner_input'))
        dialog_text = grab('dialogue')
        output_text = grab('output')
        return counts, vals, partner_vals, dialog_text, output_text

    sc_by_key = {}
    for i, sc in enumerate(scenarios):
        counts = tuple(sc.item_counts[t] for t in sc.item_types)
        vals_a = tuple(sc.values_a[t] for t in sc.item_types)
        vals_b = tuple(sc.values_b[t] for t in sc.item_types)
        sc_by_key[(counts, vals_a, vals_b)] = i

    HUMAN_ITEMS = ['book', 'hat', 'ball']

    unique = [parse_line(l) for l in lines[::2]]

    results = []
    for counts, vals, partner_vals, dialog_text, output_text in unique:
        key = (tuple(counts), tuple(vals), tuple(partner_vals))
        sc_idx = sc_by_key.get(key)
        if sc_idx is None:
            continue

        raw_turns = [t.strip() for t in dialog_text.split('<eos>') if t.strip()]
        turns = []
        for i, raw in enumerate(raw_turns):
            if '<selection>' in raw:
                continue
            if raw.startswith('YOU:'):
                speaker = 'Mia Sanders'
                text = raw[4:].strip()
            elif raw.startswith('THEM:'):
                speaker = 'Ava Thompson'
                text = raw[5:].strip()
            else:
                speaker = 'Unknown'
                text = raw
            turns.append({'turn': i, 'speaker': speaker, 'text': text})

        if not turns:
            continue

        # Human text uses "book", "hat", "ball" — build a Scenario with those
        # names so parse_items_from_text can find them in the dialog.
        human_sc = Scenario(
            env_id=f'human_{sc_idx}',
            scenario_text='',
            item_types=HUMAN_ITEMS,
            item_counts={t: c for t, c in zip(HUMAN_ITEMS, counts)},
            values_a={t: v for t, v in zip(HUMAN_ITEMS, vals)},
            values_b={t: v for t, v in zip(HUMAN_ITEMS, partner_vals)},
        )
        tagged = tag_speech_acts(turns, human_sc)

        agreed = '<disagree>' not in output_text and '<no_agreement>' not in output_text and '<disconnect>' not in output_text
        joint_score = 0.0
        if agreed:
            parts = output_text.strip().split()
            alloc_vals = []
            for p in parts:
                if '=' in p:
                    _, v = p.split('=')
                    alloc_vals.append(int(v))
            if len(alloc_vals) >= 6:
                you_alloc = alloc_vals[:3]
                them_alloc = alloc_vals[3:]
                sa = sum(a * v for a, v in zip(you_alloc, vals))
                sb = sum(a * v for a, v in zip(them_alloc, partner_vals))
                joint_score = sa + sb

        results.append({
            'model': 'Human (Lewis et al.)',
            'scenario_idx': sc_idx,
            'tagged_turns': tagged,
            'agreed': agreed,
            'joint_score': joint_score,
        })

    log.info(f"Human baseline: {len(results)} matched episodes, "
             f"{sum(len(ep['tagged_turns']) for ep in results)} turns")
    return results


def aggregate_by_model(all_results):
    """Returns DataFrame: model | total_turns | propose_frac | counter_propose_frac | ..."""
    model_data = defaultdict(lambda: {'total_turns': 0, **{a: 0 for a in SPEECH_ACTS}})

    for ep in all_results:
        model = ep['model']
        for t in ep['tagged_turns']:
            model_data[model]['total_turns'] += 1
            for act in t['acts']:
                if act in model_data[model]:
                    model_data[model][act] += 1

    rows = []
    for model, counts in sorted(model_data.items()):
        total = counts['total_turns']
        row = {'model': model, 'total_turns': total}
        for act in SPEECH_ACTS:
            row[f'{act}_frac'] = counts[act] / total if total > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_transitions(all_results):
    """Returns DataFrame of bigram counts: from_act | to_act | count"""
    trans = defaultdict(int)
    for ep in all_results:
        tagged = ep['tagged_turns']
        for i in range(len(tagged) - 1):
            for a1 in tagged[i]['acts']:
                for a2 in tagged[i + 1]['acts']:
                    trans[(a1, a2)] += 1

    rows = []
    for (from_act, to_act), count in sorted(trans.items(), key=lambda x: -x[1]):
        rows.append({'from_act': from_act, 'to_act': to_act, 'count': count})
    return pd.DataFrame(rows)


def correlate_with_outcomes(all_results):
    """Returns DataFrame: model | scenario_idx | n_proposes | n_concessions | n_accepts |
                          n_rejects | n_inquiries | agreed | joint_score"""
    rows = []
    for ep in all_results:
        counts = defaultdict(int)
        for t in ep['tagged_turns']:
            for act in t['acts']:
                counts[act] += 1
        rows.append({
            'model': ep['model'],
            'scenario_idx': ep['scenario_idx'],
            'n_proposes': counts.get('propose', 0) + counts.get('counter_propose', 0),
            'n_concessions': counts.get('concede', 0),
            'n_accepts': counts.get('accept', 0),
            'n_rejects': counts.get('reject', 0),
            'n_inquiries': counts.get('inquire', 0),
            'agreed': int(ep['agreed']),
            'joint_score': ep['joint_score'],
        })
    return pd.DataFrame(rows)


def _display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def plot_speech_act_frequency(summary_df, output_path):
    """Grouped bar chart. X-axis = models, bars = act types, y-axis = fraction of turns."""
    act_cols = [c for c in summary_df.columns if c.endswith('_frac')]
    act_names = [c.replace('_frac', '') for c in act_cols]
    models = summary_df['model'].tolist()
    display_names = [_display_name(m) for m in models]

    n_models = len(models)
    n_acts = len(act_names)
    x = np.arange(n_models)
    width = 0.8 / n_acts
    cmap = plt.cm.tab10(np.linspace(0, 1, n_acts))

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6))
    for i, (col, act_name) in enumerate(zip(act_cols, act_names)):
        vals = summary_df[col].values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=act_name, color=cmap[i], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Fraction of turns')
    ax.set_title('Speech Act Frequency by Model')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_transitions(trans_df, output_path, title=None):
    """Matrix heatmap showing P(next_act | current_act)."""
    acts_present = sorted(set(trans_df['from_act']) | set(trans_df['to_act']))
    n = len(acts_present)
    act_to_idx = {a: i for i, a in enumerate(acts_present)}
    mat = np.zeros((n, n))

    for _, row in trans_df.iterrows():
        i = act_to_idx.get(row['from_act'])
        j = act_to_idx.get(row['to_act'])
        if i is not None and j is not None:
            mat[i, j] = row['count']

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prob_mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    im = ax.imshow(prob_mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=prob_mat.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(acts_present, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(acts_present, fontsize=7)
    ax.set_xlabel('Next act')
    ax.set_ylabel('Current act')
    ax.set_title(title or 'Speech Act Transition Probabilities P(next | current)')

    for i in range(n):
        for j in range(n):
            val = prob_mat[i, j]
            if val > 0.005:
                color = 'white' if val > 0.5 * prob_mat.max() else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.7, label='P(next | current)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def write_report(output_path, summary_df, trans_df, outcome_df, all_results):
    """Write a human-readable summary report."""
    total_episodes = len(all_results)
    total_turns = sum(len(ep['tagged_turns']) for ep in all_results)
    models = sorted(set(ep['model'] for ep in all_results))

    with open(output_path, 'w') as f:
        f.write("SPEECH ACT ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total turns tagged: {total_turns}\n")
        f.write(f"Models: {len(models)}\n\n")

        f.write("SPEECH ACT FREQUENCIES BY MODEL\n")
        f.write("-" * 60 + "\n")
        act_cols = [c for c in summary_df.columns if c.endswith('_frac')]
        header = f"{'Model':<30} {'Turns':>6}"
        for col in act_cols:
            header += f" {col.replace('_frac', ''):>10}"
        f.write(header + "\n")
        for _, row in summary_df.iterrows():
            line = f"  {_display_name(row['model']):<28} {int(row['total_turns']):>6}"
            for col in act_cols:
                line += f" {row[col]:>10.3f}"
            f.write(line + "\n")
        f.write("\n")

        f.write("TOP TRANSITION BIGRAMS\n")
        f.write("-" * 60 + "\n")
        for _, row in trans_df.head(20).iterrows():
            f.write(f"  {row['from_act']:<20} -> {row['to_act']:<20} {int(row['count']):>6}\n")
        f.write("\n")

        f.write("OUTCOME CORRELATIONS (mean per model)\n")
        f.write("-" * 60 + "\n")
        if not outcome_df.empty:
            model_means = outcome_df.groupby('model').agg({
                'n_proposes': 'mean',
                'n_concessions': 'mean',
                'n_accepts': 'mean',
                'n_rejects': 'mean',
                'n_inquiries': 'mean',
                'agreed': 'mean',
                'joint_score': 'mean',
            }).reset_index()
            header = (f"{'Model':<30} {'proposes':>8} {'conces.':>8} {'accepts':>8} "
                      f"{'rejects':>8} {'inquire':>8} {'agree%':>8} {'joint':>8}")
            f.write(header + "\n")
            for _, row in model_means.iterrows():
                line = (f"  {_display_name(row['model']):<28} {row['n_proposes']:>8.1f} "
                        f"{row['n_concessions']:>8.1f} {row['n_accepts']:>8.1f} "
                        f"{row['n_rejects']:>8.1f} {row['n_inquiries']:>8.1f} "
                        f"{row['agreed']*100:>7.1f}% {row['joint_score']:>8.1f}")
                f.write(line + "\n")
        f.write("\n")

    log.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Speech act tagger for Deal-or-No-Deal dialogs.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. Default: PROJECT_ROOT/summary_plots_deal_or_no_deal/speech_acts')
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'summary_plots_deal_or_no_deal' / 'speech_acts'
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading scenarios...")
    scenarios = load_scenarios()
    log.info(f"Loaded {len(scenarios)} scenarios")

    log.info("Loading and tagging LLM episodes...")
    all_results = load_all_episodes(scenarios)
    log.info(f"Tagged {len(all_results)} LLM episodes, "
             f"{sum(len(ep['tagged_turns']) for ep in all_results)} total turns")

    log.info("Loading and tagging human episodes...")
    try:
        human_results = load_human_episodes(scenarios)
        all_results.extend(human_results)
    except Exception as e:
        log.warning(f"Failed to load human baseline: {e}")

    if not all_results:
        log.error("No episodes loaded. Check RESULT_DIRS.")
        sys.exit(1)

    log.info("Aggregating by model...")
    summary_df = aggregate_by_model(all_results)

    log.info("Correlating with outcomes...")
    outcome_df = correlate_with_outcomes(all_results)

    log.info("Writing per-turn tags CSV...")
    tag_rows = []
    for ep in all_results:
        for t in ep['tagged_turns']:
            tag_rows.append({
                'model': ep['model'],
                'scenario_idx': ep['scenario_idx'],
                'turn': t['turn'],
                'speaker': t['speaker'],
                'text': t['text'][:100],
                'acts': ','.join(t['acts']),
                'parsed_alloc': json.dumps(t['parsed_alloc']) if t['parsed_alloc'] else '',
                'speaker_score': t['speaker_score'] if t['speaker_score'] is not None else '',
            })
    pd.DataFrame(tag_rows).to_csv(output_dir / 'speech_act_tags_all.csv', index=False)

    summary_df.to_csv(output_dir / 'speech_act_summary_by_model.csv', index=False)
    outcome_df.to_csv(output_dir / 'speech_act_outcome_correlations.csv', index=False)

    log.info("Generating frequency plot...")
    plot_speech_act_frequency(summary_df, output_dir / 'speech_act_frequency.png')

    log.info("Generating per-source transition heatmaps...")
    transition_sources = [
        ('Human (Lewis et al.)', 'human'),
        ('gemini-3.1-pro-preview', 'gemini-3.1-pro-preview'),
        ('Qwen_Qwen3-14B', 'Qwen_Qwen3-14B'),
    ]
    for model_key, file_label in transition_sources:
        filtered = [ep for ep in all_results if ep['model'] == model_key]
        if not filtered:
            log.warning(f"No episodes for {model_key}, skipping transition plot")
            continue
        trans_df = compute_transitions(filtered)
        safe_label = file_label.replace('/', '_')
        trans_df.to_csv(output_dir / f'speech_act_transitions_{safe_label}.csv', index=False)
        plot_transitions(
            trans_df,
            output_dir / f'speech_act_transitions_{safe_label}.png',
            title=f'Speech Act Transitions: {_display_name(model_key)}',
        )

    log.info("Writing report...")
    global_trans_df = compute_transitions(all_results)
    write_report(output_dir / 'speech_act_report.txt', summary_df, global_trans_df, outcome_df, all_results)

    log.info(f"All outputs saved to {output_dir}")
    print(f"\nDone. Outputs in {output_dir}")


if __name__ == '__main__':
    main()
