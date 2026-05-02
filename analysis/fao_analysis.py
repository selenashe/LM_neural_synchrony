#!/usr/bin/env python3
"""
FAO (First-Acceptable-Offer) Baseline Analysis for Deal-or-No-Deal negotiations.

Determines whether LLM agent pairs achieve outcomes exceeding what
non-strategic threshold-acceptance agents can achieve. Produces a headline
plot of LLM pair outcomes against the FAO-vs-FAO frontier in
(agreement rate, joint score) space.
"""

import json
import re
import csv
import os
import ast
import sys
import itertools
import logging
import urllib.request
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = PROJECT_ROOT / 'sotopia_utils' / 'sotopia_data' / 'envs_deal_or_no_deal.json'
COMBOS_PATH = PROJECT_ROOT / 'sotopia_utils' / 'sotopia_data' / 'env_agent_combos_deal_or_no_deal.json'

RESULT_DIRS = [
    PROJECT_ROOT / 'sotopia_results_deal_or_no_deal' / 'dialogs',
    PROJECT_ROOT / 'sotopia_results_deal_or_no_deal_gemini' / 'dialogs',
]

OUTPUT_DIR = PROJECT_ROOT / 'summary_plots_deal_or_no_deal'

HUMAN_DATA_URL = ("https://raw.githubusercontent.com/dcarlyle/end-to-end-negotiator"
                  "/master/src/data/negotiate/test.txt")

MAX_TURNS_FAO = 8  # each agent gets 8 turns (16 total, matching LLM simulation)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ============================================================================
# Component 1: Scenario Parser & Allocation Enumerator
# ============================================================================

@dataclass
class Scenario:
    env_id: str
    scenario_text: str
    item_types: List[str]        # singular names, ordered
    item_counts: Dict[str, int]  # type -> total count
    values_a: Dict[str, int]     # type -> per-item value for agent A (Mia)
    values_b: Dict[str, int]     # type -> per-item value for agent B (Ava)


def _singularize_single(word: str) -> str:
    """Singularize a single word."""
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 2:
        base = word[:-2]
        if base.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return base
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def _singularize(phrase: str) -> str:
    """Singularize a (possibly multi-word) item name.

    For 'X of Y' items, the head noun is before 'of' (e.g. 'bags of soil' -> 'bag of soil').
    For other multi-word items, the head noun is the last word (e.g. 'dryer sheet boxes' -> 'dryer sheet box').
    """
    phrase = phrase.strip()
    if ' of ' in phrase:
        before, after = phrase.split(' of ', 1)
        return _singularize(before) + ' of ' + after
    parts = phrase.split()
    if len(parts) > 1:
        parts[-1] = _singularize_single(parts[-1])
        return ' '.join(parts)
    return _singularize_single(phrase)


def _pluralize_single(word: str) -> str:
    """Pluralize a single word."""
    if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    if word.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    return word + 's'


def _pluralize(phrase: str) -> str:
    """Pluralize a (possibly multi-word) item name."""
    phrase = phrase.strip()
    if ' of ' in phrase:
        before, after = phrase.split(' of ', 1)
        return _pluralize(before) + ' of ' + after
    parts = phrase.split()
    if len(parts) > 1:
        parts[-1] = _pluralize_single(parts[-1])
        return ' '.join(parts)
    return _pluralize_single(phrase)


def _parse_items_from_scenario(text: str) -> List[Tuple[str, int]]:
    """Extract (singular_name, count) from scenario text like '2 mugs, 3 plates, and 1 bowl'."""
    m = re.search(r'(?:which include|including)\s+(.*)', text)
    items_text = m.group(1) if m else text
    # Split on comma+optional-and to isolate each "N item_name" chunk
    parts = re.split(r',\s*(?:and\s+)?', items_text)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        mm = re.match(r'(\d+)\s+(.*)', part)
        if mm:
            count = int(mm.group(1))
            raw_name = mm.group(2).strip().rstrip('.')
            result.append((_singularize(raw_name), count))
    return result


def _extract_item_names_from_values(goal_text: str) -> List[str]:
    """Extract canonical singular item names from value function text."""
    m = re.search(r'<extra_info>(.*?)</extra_info>', goal_text, re.DOTALL)
    if not m:
        raise ValueError(f"No <extra_info> in: {goal_text[:120]}")
    info = m.group(1)
    names = []
    for match in re.finditer(r'(?:each|the)\s+(.*?)\s+is\s+worth\s+\d+\s+point', info, re.IGNORECASE):
        name = match.group(1).strip()
        if name not in names:
            names.append(name)
    return names


def _parse_values(goal_text: str, item_types: List[str]) -> Dict[str, int]:
    """Extract per-item point values from agent goal text."""
    m = re.search(r'<extra_info>(.*?)</extra_info>', goal_text, re.DOTALL)
    if not m:
        raise ValueError(f"No <extra_info> in: {goal_text[:120]}")
    info = m.group(1)

    values = {}
    for t in item_types:
        for form in [t, _pluralize(t)]:
            pattern = rf'(?:each\s+|the\s+)?{re.escape(form)}\s+is\s+worth\s+(\d+)\s+point'
            match = re.search(pattern, info, re.IGNORECASE)
            if match:
                values[t] = int(match.group(1))
                break
        if t not in values:
            raise ValueError(f"No value for '{t}' in: {info}")
    return values


def _find_item_count(item_name: str, scenario_text: str) -> int:
    """Find the count of an item in the scenario text by trying plural/singular forms."""
    for form in [_pluralize(item_name), item_name]:
        m = re.search(rf'(\d+)\s+{re.escape(form)}', scenario_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot find count for '{item_name}' in: {scenario_text}")


def load_scenarios() -> List[Scenario]:
    with open(SCENARIOS_PATH) as f:
        envs = json.load(f)
    with open(COMBOS_PATH) as f:
        combos = json.load(f)

    scenarios = []
    for combo in combos:
        env_id = combo['env_id']
        env = envs[env_id]

        # Extract canonical item names from value functions (already singular)
        item_types = _extract_item_names_from_values(env['agent_goals'][0])

        # Find counts from scenario text
        item_counts = {}
        for t in item_types:
            item_counts[t] = _find_item_count(t, env['scenario'])

        values_a = _parse_values(env['agent_goals'][0], item_types)
        values_b = _parse_values(env['agent_goals'][1], item_types)
        scenarios.append(Scenario(env_id, env['scenario'], item_types, item_counts, values_a, values_b))
    return scenarios


def enumerate_allocations(sc: Scenario) -> List[Dict[str, int]]:
    """All feasible alloc_a dicts (items assigned to agent A; B gets remainder)."""
    ranges = [range(sc.item_counts[t] + 1) for t in sc.item_types]
    return [{t: c for t, c in zip(sc.item_types, combo)}
            for combo in itertools.product(*ranges)]


def score_allocation(alloc_a: Dict[str, int], sc: Scenario) -> Tuple[float, float]:
    sa = sum(alloc_a[t] * sc.values_a[t] for t in sc.item_types)
    sb = sum((sc.item_counts[t] - alloc_a[t]) * sc.values_b[t] for t in sc.item_types)
    return sa, sb


def compute_scenario_metadata(sc: Scenario) -> dict:
    allocations = enumerate_allocations(sc)
    scored = [(a, *score_allocation(a, sc)) for a in allocations]
    max_joint = max(sa + sb for _, sa, sb in scored)

    pareto = []
    for alloc, sa, sb in scored:
        if not any((sa2 >= sa and sb2 >= sb and (sa2 > sa or sb2 > sb))
                   for _, sa2, sb2 in scored):
            pareto.append((alloc, sa, sb))

    nash = max(((sa, sb) for _, sa, sb in scored if sa >= 0 and sb >= 0),
               key=lambda x: x[0] * x[1], default=(0, 0))

    return {
        'scored': scored,
        'max_joint': max_joint,
        'pareto': pareto,
        'nash': nash,
        'max_a': max(sa for _, sa, _ in scored),
        'max_b': max(sb for _, _, sb in scored),
    }


# ============================================================================
# Component 2 & 3: FAO Agent and FAO-vs-FAO Simulator
# ============================================================================

def fao_opening(sc: Scenario, agent_idx: int, strategy: str = 'greedy') -> Dict[str, int]:
    """Compute FAO opening proposal as alloc_a dict."""
    if strategy == 'demand_all':
        if agent_idx == 0:
            return {t: sc.item_counts[t] for t in sc.item_types}
        else:
            return {t: 0 for t in sc.item_types}

    # Greedy: demand items valued most until score reaches 10
    values = sc.values_a if agent_idx == 0 else sc.values_b
    items_flat = []
    for t in sc.item_types:
        for _ in range(sc.item_counts[t]):
            items_flat.append((t, values[t]))
    items_flat.sort(key=lambda x: -x[1])

    agent_takes = {t: 0 for t in sc.item_types}
    running = 0
    for item_type, val in items_flat:
        if val == 0:
            continue
        if running + val <= 10:
            agent_takes[item_type] += 1
            running += val

    if agent_idx == 0:
        return agent_takes
    else:
        return {t: sc.item_counts[t] - agent_takes[t] for t in sc.item_types}


def simulate_fao_pair(sc, tau_a, tau_b, max_turns=MAX_TURNS_FAO, strategy='greedy'):
    """Returns (agreed, score_a, score_b, turns_used)."""
    opening_a = fao_opening(sc, 0, strategy)
    opening_b = fao_opening(sc, 1, strategy)

    last_proposal = None
    for turn in range(max_turns * 2):
        if turn % 2 == 0:  # A's turn
            if turn > 0 and last_proposal is not None:
                a_score = sum(last_proposal[t] * sc.values_a[t] for t in sc.item_types)
                if a_score >= tau_a:
                    return True, *score_allocation(last_proposal, sc), turn + 1
            last_proposal = opening_a
        else:  # B's turn
            b_score = sum((sc.item_counts[t] - last_proposal[t]) * sc.values_b[t]
                          for t in sc.item_types)
            if b_score >= tau_b:
                return True, *score_allocation(last_proposal, sc), turn + 1
            last_proposal = opening_b

    return False, 0, 0, max_turns * 2


def run_fao_sweep(scenarios, max_turns=MAX_TURNS_FAO, strategy='greedy'):
    results = []
    for tau in range(11):
        agreed, total_sa, total_sb = 0, 0.0, 0.0
        n = len(scenarios)
        for sc in scenarios:
            ok, sa, sb, _ = simulate_fao_pair(sc, tau, tau, max_turns, strategy)
            if ok:
                agreed += 1
                total_sa += sa
                total_sb += sb
        results.append({
            'tau': tau,
            'agreement_rate': agreed / n,
            'mean_score_a': total_sa / n,
            'mean_score_b': total_sb / n,
            'mean_joint_score': (total_sa + total_sb) / n,
            'n_agreed': agreed,
            'n_total': n,
        })
    return results


# ============================================================================
# Component 5: LLM Transcript Parser
# ============================================================================

_NUM_WORDS = {
    'zero': 0, 'no': 0, 'none': 0,
    'a': 1, 'an': 1, 'one': 1,
    'two': 2, 'both': 2,
    'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7,
}


def _parse_num(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s in _NUM_WORDS:
        return _NUM_WORDS[s]
    try:
        return int(s)
    except ValueError:
        return None


def _build_item_regex(sc: Scenario) -> str:
    """Build regex alternation matching any item form, including abbreviated forms."""
    forms = set()
    for t in sc.item_types:
        forms.add(re.escape(_pluralize(t)))
        forms.add(re.escape(t))
        parts = t.split()
        if len(parts) > 1:
            # Add suffix abbreviations: "drive" for "USB drive"
            for i in range(1, len(parts)):
                suffix = ' '.join(parts[i:])
                forms.add(re.escape(suffix))
                forms.add(re.escape(_pluralize(suffix)))
            # Add prefix abbreviations: "USB" for "USB drive"
            for j in range(1, len(parts)):
                prefix = ' '.join(parts[:j])
                forms.add(re.escape(prefix))
                forms.add(re.escape(_pluralize(prefix)))
    forms_list = sorted(forms, key=len, reverse=True)
    return '(' + '|'.join(forms_list) + ')'


def _match_item_type(name: str, item_types: List[str]) -> Optional[str]:
    name_s = _singularize(name.strip().lower())
    for t in item_types:
        tl = t.lower()
        if name_s == tl or name.strip().lower() == tl or name.strip().lower() == _pluralize(tl):
            return t
    for t in item_types:
        tl = t.lower()
        if tl in name_s or name_s in tl:
            return t
    return None


def parse_items_from_text(text: str, sc: Scenario) -> Optional[Dict[str, int]]:
    """Extract {item_type: count} from free text using the scenario's known items."""
    text_lower = text.lower()
    item_re = _build_item_regex(sc)
    num_re = r'(\d+|one|two|three|four|five|six|seven|both|all|no|none)'

    result = {}

    # Pass 1a: "NUM ITEM" (direct, no intervening adjective — safe for multi-word items)
    pattern1a = num_re + r'\s+' + item_re
    for m in re.finditer(pattern1a, text_lower, re.IGNORECASE):
        num_str, item_name = m.group(1), m.group(2)
        n = _parse_num(num_str)
        t = _match_item_type(item_name, sc.item_types)
        if n is not None and t is not None:
            if num_str in ('all', 'both'):
                n = sc.item_counts[t]
            result[t] = n

    # Pass 1b: "NUM adj ITEM" (with one adjective, for items not found in 1a)
    if len(result) < len(sc.item_types):
        pattern1b = num_re + r'\s+\w+\s+' + item_re
        for m in re.finditer(pattern1b, text_lower, re.IGNORECASE):
            num_str, item_name = m.group(1), m.group(2)
            t = _match_item_type(item_name, sc.item_types)
            if t is not None and t not in result:
                n = _parse_num(num_str)
                if n is not None:
                    if num_str in ('all', 'both'):
                        n = sc.item_counts[t]
                    result[t] = n

    # Pass 2: "the/a/an [adj] ITEM" for items still not found
    for t in sc.item_types:
        if t in result:
            continue
        singular_forms = [t]
        parts = t.split()
        if len(parts) > 1:
            for i in range(1, len(parts)):
                singular_forms.append(' '.join(parts[i:]))
            for j in range(1, len(parts)):
                singular_forms.append(' '.join(parts[:j]))
        forms_with_plural = []
        for sf in singular_forms:
            forms_with_plural.append((_pluralize(sf), True))
            forms_with_plural.append((sf, False))
        for form, is_plural in forms_with_plural:
            pat = r'(?:a|an|the)\s+(?:\w+\s+)?' + re.escape(form)
            if re.search(pat, text_lower, re.IGNORECASE):
                result[t] = sc.item_counts[t] if is_plural else 1
                break

    return result if result else None


def _get_speaker(turn: str) -> Optional[str]:
    m = re.match(r'Turn \d+:\s*(.*?)\s+said:', turn)
    return m.group(1) if m else None


def _get_utterance(turn: str) -> str:
    m = re.match(r'Turn \d+:\s*.*?\s+said:(.*)', turn, re.DOTALL)
    return m.group(1).strip() if m else turn


def _split_speaker_partner(text: str, sc: Scenario):
    """Split text into (speaker_items, partner_items) dicts.

    Returns (speaker_items, partner_items) where partner_items may be None.
    """
    # Strategy 1: semicolon split "I take X; you take Y"
    parts = re.split(r'\s*;\s*', text)
    if len(parts) >= 2:
        s = parse_items_from_text(parts[0], sc)
        p = parse_items_from_text(parts[1], sc)
        if s:
            return s, p

    # Strategy 2: "you take/get/have/keep" keyword split
    m = re.search(r',?\s*you\s+(?:take|get|have|keep|can\s+have|receive|will\s+get)\s+',
                  text, re.IGNORECASE)
    if m:
        before = text[:m.start()]
        after = text[m.end():]
        s = parse_items_from_text(before, sc)
        p = parse_items_from_text(after, sc)
        if s:
            return s, p

    # Strategy 3: whole text is speaker's items
    s = parse_items_from_text(text, sc)
    return s, None


def _build_alloc_a(speaker_items, partner_items, sc, speaker_is_a):
    """Build alloc_a dict from speaker/partner item dicts.

    If the explicit partner allocation causes over-allocation, falls back to
    inferring partner gets the remainder after the speaker takes their items.
    """
    if speaker_items is None:
        return None

    alloc_speaker = {t: speaker_items.get(t, 0) for t in sc.item_types}

    # Check speaker's allocation is valid first
    for t in sc.item_types:
        if alloc_speaker[t] < 0 or alloc_speaker[t] > sc.item_counts[t]:
            return None

    if partner_items:
        alloc_partner = {t: partner_items.get(t, 0) for t in sc.item_types}
        # Validate total adds up; if over-allocated, fall back to remainder
        valid = all(alloc_speaker[t] + alloc_partner[t] <= sc.item_counts[t]
                    for t in sc.item_types)
        if not valid:
            alloc_partner = {t: sc.item_counts[t] - alloc_speaker[t] for t in sc.item_types}
    else:
        alloc_partner = {t: sc.item_counts[t] - alloc_speaker[t] for t in sc.item_types}

    alloc_a = alloc_speaker if speaker_is_a else alloc_partner
    return alloc_a


def _parse_turn_allocation(utt: str, sc: Scenario, speaker_is_a: bool) -> Optional[Dict[str, int]]:
    """Parse a structured allocation from a single utterance."""
    clean = re.sub(r'\b(LEAVE|leave|LEFT|left)\b', '', utt).strip()
    clean = re.sub(r'[.!?,;]+$', '', clean).strip()
    clean = clean.strip('"\'()').strip()

    speaker_items, partner_items = _split_speaker_partner(clean, sc)
    return _build_alloc_a(speaker_items, partner_items, sc, speaker_is_a)


def parse_episode(dialog_turns: List[str], sc: Scenario):
    """Parse episode outcome.

    Returns: (alloc_a, agreement_type, details)
        alloc_a:        dict or None
        agreement_type: 'leave' | 'implicit' | 'disagreement' | 'parse_failure'
        details:        description string
    """
    if not dialog_turns:
        return None, 'disagreement', 'empty dialogue'

    # Check for LEAVE signal in last two turns
    leave_idx = None
    for idx in (len(dialog_turns) - 1, len(dialog_turns) - 2):
        if idx < 0:
            continue
        if re.search(r'\b(leave|left)\b', dialog_turns[idx], re.IGNORECASE):
            leave_idx = idx
            break

    if leave_idx is not None:
        turn = dialog_turns[leave_idx]
        speaker = _get_speaker(turn)
        utt = _get_utterance(turn)
        speaker_is_a = speaker is not None and 'Mia' in speaker

        alloc_a = _parse_turn_allocation(utt, sc, speaker_is_a)
        if alloc_a is not None:
            return alloc_a, 'leave', f'LEAVE by {speaker}'

        # Try the turn before LEAVE (allocation might be in previous turn)
        if leave_idx > 0:
            prev = dialog_turns[leave_idx - 1]
            prev_speaker = _get_speaker(prev)
            prev_utt = _get_utterance(prev)
            prev_is_a = prev_speaker is not None and 'Mia' in prev_speaker
            alloc_a = _parse_turn_allocation(prev_utt, sc, prev_is_a)
            if alloc_a is not None:
                return alloc_a, 'leave', f'allocation from turn before LEAVE by {prev_speaker}'

        return None, 'parse_failure', f'LEAVE by {speaker} unparseable: {utt[:150]}'

    # No LEAVE — check for implicit agreement in final turns
    if len(dialog_turns) >= 2:
        alloc_a = _check_implicit_agreement(dialog_turns, sc)
        if alloc_a is not None:
            return alloc_a, 'implicit', 'compatible proposals in final turns'

    return None, 'disagreement', f'{len(dialog_turns)} turns, no agreement signal'


def _check_implicit_agreement(turns, sc):
    """Check if last two turns from different speakers propose the same allocation."""
    last = turns[-1]
    prev = turns[-2]

    last_speaker = _get_speaker(last)
    prev_speaker = _get_speaker(prev)

    if last_speaker == prev_speaker:
        return None

    last_is_a = last_speaker is not None and 'Mia' in last_speaker
    prev_is_a = prev_speaker is not None and 'Mia' in prev_speaker

    alloc_last = _parse_turn_allocation(_get_utterance(last), sc, last_is_a)
    alloc_prev = _parse_turn_allocation(_get_utterance(prev), sc, prev_is_a)

    if alloc_last is None or alloc_prev is None:
        return None

    if all(alloc_last[t] == alloc_prev[t] for t in sc.item_types):
        return alloc_last

    return None


def find_effective_end(dialog_turns, sc):
    """Return the index (exclusive) of the effective end of a dialog.

    For LEAVE episodes: the turn containing LEAVE (+ 1).
    For implicit agreement: the second turn of the first matching pair (+ 1).
    For disagreements / no match: len(dialog_turns).
    """
    for idx in range(len(dialog_turns)):
        if re.search(r'\b(leave|left)\b', dialog_turns[idx], re.IGNORECASE):
            return idx + 1

    for idx in range(1, len(dialog_turns)):
        cur_speaker = _get_speaker(dialog_turns[idx])
        prev_speaker = _get_speaker(dialog_turns[idx - 1])
        if cur_speaker == prev_speaker:
            continue
        cur_is_a = cur_speaker is not None and 'Mia' in cur_speaker
        prev_is_a = prev_speaker is not None and 'Mia' in prev_speaker
        alloc_cur = _parse_turn_allocation(_get_utterance(dialog_turns[idx]), sc, cur_is_a)
        alloc_prev = _parse_turn_allocation(_get_utterance(dialog_turns[idx - 1]), sc, prev_is_a)
        if alloc_cur is not None and alloc_prev is not None:
            if all(alloc_cur[t] == alloc_prev[t] for t in sc.item_types):
                return idx + 1

    return len(dialog_turns)


# ============================================================================
# Data Loading
# ============================================================================

def _load_csv_dialog(path: Path) -> List[str]:
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        row = next(reader)
    dialog_str = row[1]
    try:
        turns = ast.literal_eval(dialog_str)
    except (ValueError, SyntaxError):
        turns = json.loads(dialog_str)
    if isinstance(turns, str):
        turns = [turns]
    return turns


def extract_model_name(pair_dir_name: str) -> str:
    m = re.match(r'(.+?)_None_\d+_.+?_None_\d+_deal_or_no_deal', pair_dir_name)
    return m.group(1) if m else pair_dir_name


def load_llm_episodes(scenarios):
    """Load all LLM episodes.

    Returns: dict  model_pair_dirname -> list of (scenario_idx, dialog_turns)
    """
    episodes = defaultdict(list)
    for result_dir in RESULT_DIRS:
        if not result_dir.exists():
            log.warning(f"Result dir not found: {result_dir}")
            continue
        for pair_dir in sorted(result_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            for csv_file in sorted(pair_dir.glob('*.csv')):
                m = re.match(r'(\d+)_temp', csv_file.name)
                if not m:
                    continue
                sc_idx = int(m.group(1))
                if sc_idx >= len(scenarios):
                    log.warning(f"Scenario index {sc_idx} out of range: {csv_file}")
                    continue
                try:
                    dialog_turns = _load_csv_dialog(csv_file)
                    episodes[pair_dir.name].append((sc_idx, dialog_turns))
                except Exception as e:
                    log.warning(f"Failed to load {csv_file}: {e}")
    return episodes


def load_human_episodes(scenarios):
    """Load human dialogues from Lewis et al. 2017 test set, matched to our scenarios.

    When multiple human dialogues exist for the same scenario, averages their
    outcomes so each scenario contributes one episode — matching the LLM setup.

    Returns list of dicts with keys: scenario_idx, agreed, score_a, score_b, joint.
    """
    text = urllib.request.urlopen(HUMAN_DATA_URL, timeout=30).read().decode("utf-8")
    lines = text.strip().splitlines()

    def parse_line(line):
        def grab(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", line)
            return m.group(1).strip() if m else ""
        def to_cv(s):
            nums = list(map(int, s.split()))
            return nums[0::2], nums[1::2]
        counts, vals = to_cv(grab("input"))
        _, partner_vals = to_cv(grab("partner_input"))
        return counts, vals, partner_vals, grab("output")

    # Data has paired lines (same dialogue, flipped perspective); take even lines
    unique = [parse_line(l) for l in lines[::2]]

    # Build scenario key lookup: (counts, vals_a, vals_b) -> scenario index
    sc_by_key = {}
    for i, sc in enumerate(scenarios):
        counts = [sc.item_counts[t] for t in sc.item_types]
        vals_a = [sc.values_a[t] for t in sc.item_types]
        vals_b = [sc.values_b[t] for t in sc.item_types]
        sc_by_key[(tuple(counts), tuple(vals_a), tuple(vals_b))] = i

    # Collect all human dialogues per matched scenario
    by_scenario = defaultdict(list)
    n_raw_matched = 0
    for counts, vals, partner_vals, output in unique:
        key = (tuple(counts), tuple(vals), tuple(partner_vals))
        sc_idx = sc_by_key.get(key)
        if sc_idx is None:
            continue
        n_raw_matched += 1

        if '<disagree>' in output or '<no_agreement>' in output or '<disconnect>' in output:
            by_scenario[sc_idx].append((False, 0, 0))
            continue

        parts = output.strip().split()
        alloc_vals = []
        for p in parts:
            if '=' not in p:
                continue
            _, v = p.split('=')
            alloc_vals.append(int(v))
        you_alloc = alloc_vals[:3]
        them_alloc = alloc_vals[3:]

        sa = sum(a * v for a, v in zip(you_alloc, vals))
        sb = sum(a * v for a, v in zip(them_alloc, partner_vals))
        by_scenario[sc_idx].append((True, sa, sb))

    # Average within each scenario to produce one episode per scenario
    episodes = []
    for sc_idx, dialogs in sorted(by_scenario.items()):
        n = len(dialogs)
        n_agreed = sum(1 for agreed, _, _ in dialogs if agreed)
        avg_sa = sum(sa for _, sa, _ in dialogs) / n
        avg_sb = sum(sb for _, _, sb in dialogs) / n
        episodes.append({
            'scenario_idx': sc_idx,
            'agreement_rate': n_agreed / n,
            'score_a': avg_sa,
            'score_b': avg_sb,
            'joint': avg_sa + avg_sb,
            'n_dialogs': n,
        })

    log.info(f"Human baseline: {n_raw_matched} raw dialogues across "
             f"{len(by_scenario)} matched scenarios -> {len(episodes)} episodes")
    return episodes


def compute_human_metrics(human_episodes, scenarios, scenario_meta):
    """Compute metrics for human baseline in the same format as LLM metrics.

    Each episode is one scenario (already averaged if multiple human dialogues
    existed for that scenario).
    """
    n_total = len(human_episodes)
    total_agreement = 0.0
    total_sa = 0.0
    total_sb = 0.0
    total_pareto_eff = 0.0
    n_with_agreement = 0
    n_pareto_optimal = 0
    scores_list = []
    n_raw_dialogs = 0
    total_sa_agreed, total_sb_agreed = 0.0, 0.0
    score_ratios = []
    score_diffs = []
    nash_distances = []
    n_ir_violation = 0

    for e in human_episodes:
        sc_idx = e['scenario_idx']
        meta = scenario_meta[sc_idx]
        sa, sb = e['score_a'], e['score_b']
        ar = e['agreement_rate']
        n_raw_dialogs += e['n_dialogs']

        total_agreement += ar
        total_sa += sa
        total_sb += sb
        scores_list.append((sa, sb, sc_idx))

        if ar > 0:
            n_with_agreement += 1
            total_sa_agreed += sa
            total_sb_agreed += sb
            joint = sa + sb
            max_joint = meta['max_joint']
            agreed_joint = joint / ar if ar > 0 else 0
            total_pareto_eff += agreed_joint / max_joint if max_joint > 0 else 0

            is_pareto = any(abs(p_sa - sa) < 0.01 and abs(p_sb - sb) < 0.01
                           for _, p_sa, p_sb in meta['pareto'])
            if is_pareto:
                n_pareto_optimal += 1

            hi, lo = max(sa, sb), min(sa, sb)
            score_ratios.append(lo / hi if hi > 0 else 0)
            score_diffs.append(abs(sa - sb))
            nash_sa, nash_sb = meta['nash']
            nash_distances.append(((sa - nash_sa) ** 2 + (sb - nash_sb) ** 2) ** 0.5)
            if sa == 0 or sb == 0:
                n_ir_violation += 1

    mean_ar = total_agreement / n_total if n_total > 0 else 0

    return {
        'model': 'Human (Lewis et al.)',
        'n_episodes': n_total,
        'n_valid': n_total,
        'n_agreed': n_with_agreement,
        'n_raw_dialogs': n_raw_dialogs,
        'n_parse_fail': 0,
        'n_implicit': 0,
        'agreement_rate': mean_ar,
        'mean_score_a': total_sa / n_total if n_total > 0 else 0,
        'mean_score_b': total_sb / n_total if n_total > 0 else 0,
        'mean_joint_score': (total_sa + total_sb) / n_total if n_total > 0 else 0,
        'mean_joint_score_agreed': (total_sa_agreed + total_sb_agreed) / n_with_agreement if n_with_agreement > 0 else 0,
        'mean_pareto_efficiency': total_pareto_eff / n_with_agreement if n_with_agreement > 0 else 0,
        'pareto_optimal_rate': n_pareto_optimal / n_with_agreement if n_with_agreement > 0 else 0,
        'parse_failure_rate': 0,
        'mean_turns': 0,
        'n_walked_away': 0,
        'n_timed_out': 0,
        'score_ratio_mean': float(np.mean(score_ratios)) if score_ratios else 0,
        'score_abs_diff_mean': float(np.mean(score_diffs)) if score_diffs else 0,
        'nash_distance_mean': float(np.mean(nash_distances)) if nash_distances else 0,
        'ir_violation_rate': n_ir_violation / n_with_agreement if n_with_agreement > 0 else 0,
        'scores': scores_list,
    }


# ============================================================================
# Component 6: Outcome Metrics
# ============================================================================

def compute_llm_metrics(episodes, scenarios, scenario_meta):
    """Compute outcome metrics for each LLM pair.

    Returns: (metrics_dict, parse_failures_list)
    """
    results = {}
    parse_failures = []

    for pair_name, pair_episodes in episodes.items():
        model = extract_model_name(pair_name)
        n_total = len(pair_episodes)

        n_agreed = 0
        n_parse_fail = 0
        n_implicit = 0
        total_sa, total_sb = 0.0, 0.0
        total_pareto_eff = 0.0
        n_pareto_optimal = 0
        scores_list = []
        total_turns = 0
        total_sa_agreed, total_sb_agreed = 0.0, 0.0
        n_walked_away = 0
        n_timed_out = 0
        score_ratios = []
        score_diffs = []
        nash_distances = []
        n_ir_violation = 0

        for sc_idx, dialog in pair_episodes:
            sc = scenarios[sc_idx]
            meta = scenario_meta[sc_idx]

            alloc_a, atype, details = parse_episode(dialog, sc)
            total_turns += len(dialog)

            if atype == 'parse_failure':
                n_parse_fail += 1
                parse_failures.append({
                    'model': model,
                    'scenario_idx': sc_idx,
                    'env_id': sc.env_id,
                    'details': details,
                    'last_turn': dialog[-1][:200] if dialog else '',
                })
                continue

            if atype in ('leave', 'implicit'):
                n_agreed += 1
                if atype == 'implicit':
                    n_implicit += 1
                sa, sb = score_allocation(alloc_a, sc)
                total_sa += sa
                total_sb += sb
                total_sa_agreed += sa
                total_sb_agreed += sb
                scores_list.append((sa, sb, sc_idx))

                joint = sa + sb
                max_joint = meta['max_joint']
                total_pareto_eff += joint / max_joint if max_joint > 0 else 0

                is_pareto = any(
                    all(alloc_a[t] == pa[t] for t in sc.item_types)
                    for pa, _, _ in meta['pareto']
                )
                if is_pareto:
                    n_pareto_optimal += 1

                hi, lo = max(sa, sb), min(sa, sb)
                score_ratios.append(lo / hi if hi > 0 else 0)
                score_diffs.append(abs(sa - sb))
                nash_sa, nash_sb = meta['nash']
                nash_distances.append(((sa - nash_sa) ** 2 + (sb - nash_sb) ** 2) ** 0.5)
                if sa == 0 or sb == 0:
                    n_ir_violation += 1
            else:
                scores_list.append((0, 0, sc_idx))
                if len(dialog) >= 16:
                    n_timed_out += 1
                else:
                    n_walked_away += 1

        n_valid = n_total - n_parse_fail

        results[pair_name] = {
            'model': model,
            'n_episodes': n_total,
            'n_valid': n_valid,
            'n_agreed': n_agreed,
            'n_parse_fail': n_parse_fail,
            'n_implicit': n_implicit,
            'agreement_rate': n_agreed / n_valid if n_valid > 0 else 0,
            'mean_score_a': total_sa / n_valid if n_valid > 0 else 0,
            'mean_score_b': total_sb / n_valid if n_valid > 0 else 0,
            'mean_joint_score': (total_sa + total_sb) / n_valid if n_valid > 0 else 0,
            'mean_joint_score_agreed': (total_sa_agreed + total_sb_agreed) / n_agreed if n_agreed > 0 else 0,
            'mean_pareto_efficiency': total_pareto_eff / n_agreed if n_agreed > 0 else 0,
            'pareto_optimal_rate': n_pareto_optimal / n_agreed if n_agreed > 0 else 0,
            'parse_failure_rate': n_parse_fail / n_total if n_total > 0 else 0,
            'mean_turns': total_turns / n_valid if n_valid > 0 else 0,
            'n_walked_away': n_walked_away,
            'n_timed_out': n_timed_out,
            'score_ratio_mean': float(np.mean(score_ratios)) if score_ratios else 0,
            'score_abs_diff_mean': float(np.mean(score_diffs)) if score_diffs else 0,
            'nash_distance_mean': float(np.mean(nash_distances)) if nash_distances else 0,
            'ir_violation_rate': n_ir_violation / n_agreed if n_agreed > 0 else 0,
            'scores': scores_list,
        }

    return results, parse_failures


def compute_fao_relative_metrics(llm_metrics, fao_sweep):
    """Add surplus-over-FAO and above-frontier metrics."""
    fao_ar = np.array([r['agreement_rate'] for r in fao_sweep])
    fao_js = np.array([r['mean_joint_score'] for r in fao_sweep])

    order = np.argsort(fao_ar)
    fao_ar_sorted = fao_ar[order]
    fao_js_sorted = fao_js[order]

    # Build upper envelope for the frontier
    # Walk from low agreement rate to high, keeping running max of joint score
    frontier_ar = [fao_ar_sorted[0]]
    frontier_js = [fao_js_sorted[0]]
    for i in range(1, len(fao_ar_sorted)):
        if fao_js_sorted[i] >= frontier_js[-1] or fao_ar_sorted[i] > frontier_ar[-1]:
            frontier_ar.append(fao_ar_sorted[i])
            frontier_js.append(max(fao_js_sorted[i], frontier_js[-1]))

    frontier_ar = np.array(frontier_ar)
    frontier_js = np.array(frontier_js)

    for m in llm_metrics.values():
        ar = m['agreement_rate']
        js = m['mean_joint_score']
        fao_js_at_ar = float(np.interp(ar, frontier_ar, frontier_js))
        m['fao_js_at_matched_ar'] = fao_js_at_ar
        m['surplus_over_fao'] = js - fao_js_at_ar
        m['above_fao_frontier'] = js > fao_js_at_ar + 0.01

    return llm_metrics


# ============================================================================
# Component 7 & 8: Plotting
# ============================================================================

MODEL_DISPLAY = {
    'Meta-Llama-3-8B-Instruct': 'Llama-3-8B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B-v0.2',
    'Mistral-7B-Instruct-v0.3': 'Mistral-7B-v0.3',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'allenai_OLMo-3-7B-Instruct': 'OLMo-3-7B',
    'DeepSeek-R1-Distill-Llama-8B': 'DeepSeek-R1-8B',
    'Qwen3-8B': 'Qwen3-8B',
    'Qwen_Qwen3-14B': 'Qwen3-14B',
    'gemini-3.1-pro-preview': 'Gemini-3.1-Pro',
    'Human (Lewis et al.)': 'Human',
}

MODEL_COLORS = {
    'Meta-Llama-3-8B-Instruct': '#e41a1c',
    'Mistral-7B-Instruct-v0.2': '#377eb8',
    'Mistral-7B-Instruct-v0.3': '#4daf4a',
    'Qwen2.5-7B-Instruct': '#984ea3',
    'allenai_OLMo-3-7B-Instruct': '#ff7f00',
    'DeepSeek-R1-Distill-Llama-8B': '#a65628',
    'Qwen3-8B': '#f781bf',
    'Qwen_Qwen3-14B': '#666666',
    'gemini-3.1-pro-preview': '#e7298a',
    'Human (Lewis et al.)': '#2ca02c',
}


def _display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def plot_headline(fao_greedy, llm_metrics, output_path,
                  human_metrics=None):
    fig, ax = plt.subplots(figsize=(11, 7.5))

    # FAO frontier (greedy)
    ar_g = [r['agreement_rate'] for r in fao_greedy]
    js_g = [r['mean_joint_score'] for r in fao_greedy]
    ax.plot(ar_g, js_g, 'k-o', ms=5, lw=2, label='FAO frontier (greedy)', zorder=3)
    for r in fao_greedy:
        if r['tau'] in (0, 2, 4, 5, 6, 7, 8, 9, 10):
            ax.annotate(f"τ={r['tau']}", (r['agreement_rate'], r['mean_joint_score']),
                        textcoords='offset points', xytext=(6, 4), fontsize=7, color='gray')

    # Human baseline
    if human_metrics:
        hm = human_metrics
        ax.scatter(hm['agreement_rate'], hm['mean_joint_score'],
                   c='#2ca02c', marker='*', s=350, zorder=6,
                   edgecolors='black', linewidths=0.8, label='Human (Lewis et al.)')
        n_raw = hm.get('n_raw_dialogs', hm['n_episodes'])
        ax.annotate(f"Human ({hm['n_episodes']} scenarios, {n_raw} dialogs)",
                    (hm['agreement_rate'], hm['mean_joint_score']),
                    textcoords='offset points', xytext=(12, -5),
                    fontsize=9, fontweight='bold', color='#2ca02c')

    # LLM pairs
    for pair_name, m in llm_metrics.items():
        model = m['model']
        color = MODEL_COLORS.get(model, '#333333')
        marker = '^' if m.get('above_fao_frontier') else 'o'
        ax.scatter(m['agreement_rate'], m['mean_joint_score'],
                   c=color, marker=marker, s=140, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.annotate(_display_name(model),
                    (m['agreement_rate'], m['mean_joint_score']),
                    textcoords='offset points', xytext=(10, -5),
                    fontsize=8, fontweight='bold', color=color)

    ax.set_xlabel('Agreement Rate', fontsize=13)
    ax.set_ylabel('Mean Joint Score (across all episodes, 0 for disagreement)', fontsize=13)
    ax.set_title('LLM Negotiation Outcomes vs. FAO Baseline', fontsize=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_pareto_efficiency(llm_metrics, output_path, human_metrics=None):
    fig, ax = plt.subplots(figsize=(11, 7.5))

    if human_metrics:
        hm = human_metrics
        ax.scatter(hm['agreement_rate'], hm['mean_pareto_efficiency'],
                   c='#2ca02c', marker='*', s=350, zorder=6,
                   edgecolors='black', linewidths=0.8)
        n_raw = hm.get('n_raw_dialogs', hm['n_episodes'])
        ax.annotate(f"Human ({hm['n_episodes']} sc., {n_raw} dlg.)",
                    (hm['agreement_rate'], hm['mean_pareto_efficiency']),
                    textcoords='offset points', xytext=(12, -5),
                    fontsize=9, fontweight='bold', color='#2ca02c')

    for m in llm_metrics.values():
        model = m['model']
        color = MODEL_COLORS.get(model, '#333333')
        ax.scatter(m['agreement_rate'], m['mean_pareto_efficiency'],
                   c=color, s=140, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(_display_name(model),
                    (m['agreement_rate'], m['mean_pareto_efficiency']),
                    textcoords='offset points', xytext=(10, -5),
                    fontsize=8, fontweight='bold', color=color)

    ax.set_xlabel('Agreement Rate', fontsize=13)
    ax.set_ylabel('Mean Pareto Efficiency (agreed episodes only)', fontsize=13)
    ax.set_title('Pareto Efficiency vs. Agreement Rate', fontsize=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Perfect efficiency')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {output_path}")


def plot_per_pair_distributions(llm_metrics, scenarios, scenario_meta, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for m in llm_metrics.values():
        model = m['model']
        scores = m['scores']
        if not scores:
            continue

        fig, ax = plt.subplots(figsize=(7, 7))

        sa_vals = [s[0] for s in scores]
        sb_vals = [s[1] for s in scores]

        # Jitter zero-score points slightly for visibility
        sa_plot = [s + np.random.uniform(-0.15, 0.15) if s == 0 and sb_vals[i] == 0
                   else s for i, s in enumerate(sa_vals)]
        sb_plot = [s + np.random.uniform(-0.15, 0.15) if s == 0 and sa_vals[i] == 0
                   else s for i, s in enumerate(sb_vals)]

        ax.scatter(sa_plot, sb_plot, alpha=0.5, s=40, c='steelblue',
                   edgecolors='navy', linewidths=0.3)

        # Overlay Pareto frontier points across seen scenarios
        pareto_points = set()
        for sc_idx in set(s[2] for s in scores):
            for _, sa, sb in scenario_meta[sc_idx]['pareto']:
                pareto_points.add((sa, sb))
        if pareto_points:
            pp = sorted(pareto_points)
            ax.scatter([p[0] for p in pp], [p[1] for p in pp],
                       marker='x', c='red', s=15, alpha=0.15, label='Pareto (all scenarios)')

        ax.set_xlabel('Agent A Score (Mia)', fontsize=11)
        ax.set_ylabel('Agent B Score (Ava)', fontsize=11)
        ax.set_title(f'{_display_name(model)}\n'
                     f'n={m["n_episodes"]}  agree={m["agreement_rate"]:.0%}  '
                     f'joint={m["mean_joint_score"]:.1f}', fontsize=12)
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-0.5, 11)
        ax.plot([0, 10], [10, 0], 'k--', alpha=0.15)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')

        fig.tight_layout()
        safe_name = model.replace('/', '_').replace(' ', '_')
        fig.savefig(output_dir / f'{safe_name}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

    log.info(f"Saved per-pair distributions to {output_dir}")


def load_human_turn_counts(scenarios):
    """Load per-dialog turn counts from Lewis et al. 2017 human data."""
    text = urllib.request.urlopen(HUMAN_DATA_URL, timeout=30).read().decode("utf-8")
    lines = text.strip().splitlines()

    sc_by_key = {}
    for i, sc in enumerate(scenarios):
        counts = [sc.item_counts[t] for t in sc.item_types]
        vals_a = [sc.values_a[t] for t in sc.item_types]
        vals_b = [sc.values_b[t] for t in sc.item_types]
        sc_by_key[(tuple(counts), tuple(vals_a), tuple(vals_b))] = i

    turn_counts = []
    for line in lines[::2]:
        m_input = re.search(r'<input>(.*?)</input>', line)
        m_partner = re.search(r'<partner_input>(.*?)</partner_input>', line)
        m_dialog = re.search(r'<dialogue>(.*?)</dialogue>', line)
        if not (m_input and m_partner and m_dialog):
            continue

        nums = list(map(int, m_input.group(1).split()))
        counts_list, vals = nums[0::2], nums[1::2]
        partner_nums = list(map(int, m_partner.group(1).split()))
        partner_vals = partner_nums[1::2]

        key = (tuple(counts_list), tuple(vals), tuple(partner_vals))
        if key not in sc_by_key:
            continue

        raw_turns = [t.strip() for t in m_dialog.group(1).split('<eos>') if t.strip()]
        n_turns = sum(1 for t in raw_turns if '<selection>' not in t)
        turn_counts.append(n_turns)

    return turn_counts


def plot_turn_distribution(episodes, scenarios, output_path, human_turn_counts=None):
    """Subplot grid: one histogram per model pair + human. Uses effective turn counts."""
    turn_counts_by_model = defaultdict(list)
    for pair_name, pair_episodes in episodes.items():
        model = extract_model_name(pair_name)
        for sc_idx, dialog in pair_episodes:
            eff = find_effective_end(dialog, scenarios[sc_idx])
            turn_counts_by_model[model].append(eff)

    if human_turn_counts:
        turn_counts_by_model['Human (Lewis et al.)'] = human_turn_counts

    models = sorted(turn_counts_by_model.keys())
    n = len(models)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    all_turn_vals = []
    for tc in turn_counts_by_model.values():
        all_turn_vals.extend(tc)
    max_turns = max(all_turn_vals) if all_turn_vals else 16
    bins = range(1, max_turns + 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        tc = turn_counts_by_model[model]
        color = MODEL_COLORS.get(model, '#5dade2')
        ax.hist(tc, bins=bins, color=color, edgecolor='black', linewidth=0.5,
                align='left', rwidth=0.85)
        ax.set_xlabel('Number of turns', fontsize=9)
        ax.set_ylabel('Episodes', fontsize=9)
        display = _display_name(model)
        mean_t = np.mean(tc)
        ax.set_title(f'{display}  (N={len(tc)}, mean={mean_t:.1f})', fontsize=10)
        ax.tick_params(labelsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Turn Distribution by Model', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load & validate scenarios ---
    log.info("Loading scenarios...")
    scenarios = load_scenarios()
    log.info(f"Loaded {len(scenarios)} scenarios")

    for sc in scenarios:
        ta = sum(sc.item_counts[t] * sc.values_a[t] for t in sc.item_types)
        tb = sum(sc.item_counts[t] * sc.values_b[t] for t in sc.item_types)
        assert ta == 10, f"{sc.env_id}: agent A values sum to {ta}, expected 10"
        assert tb == 10, f"{sc.env_id}: agent B values sum to {tb}, expected 10"

    # --- 2. Scenario metadata (allocations, Pareto frontiers) ---
    log.info("Computing scenario metadata...")
    scenario_meta = [compute_scenario_metadata(sc) for sc in scenarios]

    # --- 3. FAO sweeps ---
    log.info("Running FAO-vs-FAO sweep (greedy opening)...")
    fao_greedy = run_fao_sweep(scenarios, strategy='greedy')
    pd.DataFrame(fao_greedy).to_csv(OUTPUT_DIR / 'fao_frontier_greedy.csv', index=False)

    print("\n=== FAO Sweep (Greedy Opening) ===")
    print(f"{'tau':>4s}  {'agree':>6s}  {'joint':>6s}  {'score_A':>8s}  {'score_B':>8s}")
    for r in fao_greedy:
        print(f"  {r['tau']:2d}    {r['agreement_rate']:.3f}   {r['mean_joint_score']:5.2f}"
              f"     {r['mean_score_a']:5.2f}     {r['mean_score_b']:5.2f}")

    # --- 4. Load & parse LLM episodes ---
    log.info("Loading LLM episodes...")
    episodes = load_llm_episodes(scenarios)
    log.info(f"Loaded episodes for {len(episodes)} model pairs:")
    for pair, eps in sorted(episodes.items()):
        log.info(f"  {extract_model_name(pair):40s} {len(eps):3d} episodes")

    # --- 4b. Load human baseline from Lewis et al. 2017 test set ---
    log.info("Loading human baseline dialogues...")
    human_metrics = None
    try:
        human_episodes = load_human_episodes(scenarios)
        human_metrics = compute_human_metrics(human_episodes, scenarios, scenario_meta)
        human_metrics = compute_fao_relative_metrics(
            {'human': human_metrics}, fao_greedy)['human']
    except Exception as e:
        log.warning(f"Failed to load human baseline: {e}")

    # --- 5. Compute LLM metrics ---
    log.info("Computing LLM outcome metrics...")
    llm_metrics, parse_failures = compute_llm_metrics(episodes, scenarios, scenario_meta)
    llm_metrics = compute_fao_relative_metrics(llm_metrics, fao_greedy)

    # Save CSVs
    rows = []
    for pair_name, m in llm_metrics.items():
        rows.append({k: v for k, v in m.items() if k != 'scores'})
    if human_metrics:
        rows.append({k: v for k, v in human_metrics.items() if k != 'scores'})
    llm_df = pd.DataFrame(rows).sort_values('surplus_over_fao', ascending=False)
    llm_df.to_csv(OUTPUT_DIR / 'llm_pair_outcomes.csv', index=False)

    if parse_failures:
        pd.DataFrame(parse_failures).to_csv(OUTPUT_DIR / 'parse_failure_log.csv', index=False)

    # Print summary table
    print("\n=== Outcomes (sorted by surplus over FAO) ===")
    print(f"{'Model':>35s}  {'agree':>6s}  {'joint':>6s}  {'jnt/deal':>8s}  {'surplus':>8s}  "
          f"{'P.eff':>6s}  {'fair':>5s}  {'Nash':>5s}  {'turns':>5s}  "
          f"{'walk':>4s}  {'t/o':>4s}  {'IR%':>5s}  {'n':>4s}  {'fail':>5s}")

    all_metrics = list(llm_metrics.items())
    if human_metrics:
        all_metrics.append(('human', human_metrics))

    for _, m in sorted(all_metrics, key=lambda x: -x[1]['surplus_over_fao']):
        print(f"  {_display_name(m['model']):33s}  {m['agreement_rate']:.3f}  "
              f"{m['mean_joint_score']:5.2f}  {m['mean_joint_score_agreed']:>7.2f}  "
              f"{m['surplus_over_fao']:+6.2f}  "
              f"{m['mean_pareto_efficiency']:.3f}  "
              f"{m['score_ratio_mean']:.2f}  "
              f"{m['nash_distance_mean']:.2f}  "
              f"{m['mean_turns']:4.1f}  "
              f"{m['n_walked_away']:4d}  {m['n_timed_out']:4d}  "
              f"{m['ir_violation_rate']*100:4.1f}  "
              f"{m['n_valid']:4d}  {m['n_parse_fail']:5d}")

    if parse_failures:
        print(f"\n  Total parse failures: {len(parse_failures)}")

    # --- 6. Plots ---
    log.info("Generating plots...")
    plot_headline(fao_greedy, llm_metrics,
                  OUTPUT_DIR / 'negotiation_fao_frontier.png',
                  human_metrics=human_metrics)
    plot_pareto_efficiency(llm_metrics, OUTPUT_DIR / 'pareto_efficiency_frontier.png',
                           human_metrics=human_metrics)
    plot_per_pair_distributions(llm_metrics, scenarios, scenario_meta,
                                OUTPUT_DIR / 'per_pair_score_distributions')

    human_turn_counts = None
    try:
        human_turn_counts = load_human_turn_counts(scenarios)
    except Exception as e:
        log.warning(f"Failed to load human turn counts: {e}")
    plot_turn_distribution(episodes, scenarios, OUTPUT_DIR / 'turn_distribution.png',
                           human_turn_counts=human_turn_counts)

    log.info(f"All outputs saved to {OUTPUT_DIR}")
    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
