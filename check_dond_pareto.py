#!/usr/bin/env python3
"""
Pareto-efficiency check for deal-or-no-deal (item division) episodes in the
Gemini Mia/Ava run. Reads dialog CSVs, parses scenario inventory + per-agent
valuations from the intro, and compares the agreed final split to the full
Pareto frontier (exhaustive enumeration; item counts are small).

Episode 5: no dialog CSV in repo. Episode 53: no agreement / truncated run.
Episode 88: no explicit mutual acceptance on the last turn; we report two
candidate splits from late turns.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent
DIALOG_DIR = (
    REPO
    / "sotopia_results_gemini"
    / "dialogs"
    / "gemini-3.1-pro-preview_None_0_gemini-3.1-pro-preview_None_0"
)
COMBOS_PATH = REPO / "sotopia_utils" / "sotopia_data" / "env_agent_combos_fixed_two_agents.json"
ENVS_PATH = REPO / "sotopia_utils" / "sotopia_data" / "envs.json"

# Episodes whose env source is deal-or-no-deal (from env_agent_combos_fixed_two_agents + envs.json)
DOND_EPISODES = [5, 10, 25, 26, 31, 42, 44, 53, 83, 88]

# Canonical item keys (singular)
NORM = {
    "apple": "apple",
    "apples": "apple",
    "banana": "banana",
    "bananas": "banana",
    "orange": "orange",
    "oranges": "orange",
    "book": "book",
    "books": "book",
    "hat": "hat",
    "hats": "hat",
    "ball": "ball",
    "balls": "ball",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_inventory_from_scenario(scenario: str) -> Counter:
    inv = Counter()
    # "3 apples, 2 bananas, and 1 orange" / "4 books, 6 hats, and 2 balls"
    pat = re.compile(
        r"(\d+)\s+(apples?|bananas?|oranges?|books?|hats?|balls?)\b",
        re.I,
    )
    for n, w in pat.findall(scenario):
        key = NORM.get(w.lower())
        if key:
            inv[key] += int(n)
    return inv


def parse_valuations_from_intro(intro: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Mia = agent 1, Ava = agent 2. Returns (mia_pts_per_item, ava_pts_per_item)."""
    mia_block = re.search(
        r"Mia Sanders's goal:.*?(?=Ava Thompson's goal:)", intro, re.DOTALL
    )
    ava_block = re.search(r"Ava Thompson's goal:.*", intro, re.DOTALL)
    if not mia_block or not ava_block:
        raise ValueError("Could not split goal blocks")

    def parse_block(block: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        lower = block.lower()
        # (regex, kind): kind is "pair" (item, pts) or fixed item for single capture
        tagged: List[Tuple[str, str]] = [
            (r"each\s+(apple|banana|orange|book|hat|ball)\s+as\s+(\d+)", "pair"),
            (r"each\s+(apple|banana|orange|book|hat|ball)[^0-9]*(\d+)", "pair"),
            (r"the orange as\s+(\d+)", "orange"),
            (r"the orange equals\s+(\d+)", "orange"),
            (r"1 (apple|banana) equals\s+(\d+)", "pair"),
            (r"(apple|banana|orange|book|hat|ball)s?\s*(?:is|are|=)\s*(?:worth\s*)?(\d+)", "pair"),
            (r"(apple|banana|orange|book|hat|ball)s?\s*at\s*(\d+)\s*points?", "pair"),
            (r"(apple|banana|orange|book|hat|ball)s?\s*give\s*you\s*(\d+)", "pair"),
            (r"(books?|hats?|balls?)\s*give\s*you\s*(\d+)\s*points?\s*each", "pair"),
            (r"you value the (books?|hats?|balls?)\s*at\s*(\d+)\s*points?\s*each", "pair"),
            (r"value the (books?|hats?|balls?)\s*at\s*(\d+)\s*points?\s*each", "pair"),
            (r"the (hats|balls|books)\s*are\s*worth\s*(\d+)\s*points?\s*each", "pair"),
            (r"the (book|ball|hat)\s*is\s*worth\s*(\d+)\s*points?", "pair"),
            (r"a (hat|ball|book)\s*is\s*worth\s*(\d+)\s*points?", "pair"),
            (r"balls? are worth\s*(\d+)\s*points?\s*each", "ball"),
            (r"ball gives you\s*(\d+)", "ball"),
            (r"the ball gives you\s*(\d+)", "ball"),
        ]
        for pat, kind in tagged:
            for m in re.finditer(pat, lower):
                if kind == "pair":
                    w, pts = m.group(1), int(m.group(2))
                    key = NORM.get(w.lower())
                elif kind == "orange":
                    key, pts = "orange", int(m.group(1))
                elif kind == "ball":
                    key, pts = "ball", int(m.group(1))
                else:
                    continue
                if key in ("apple", "banana", "orange", "book", "hat", "ball"):
                    out[key] = pts
        return out

    return parse_block(mia_block.group(0)), parse_block(ava_block.group(0))


def read_dialog_csv(ep: int) -> Tuple[str, List[str]]:
    path = DIALOG_DIR / f"{ep}_temp0.7_seed0.csv"
    if not path.is_file():
        return "", []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return "", []
    intro = rows[1][0] if rows[1] else ""
    dialog_raw = rows[1][1] if len(rows[1]) > 1 else "[]"
    try:
        turns = ast.literal_eval(dialog_raw)
    except (SyntaxError, ValueError):
        turns = []
    return intro, turns


def score(inv: Counter, per_item: Dict[str, int]) -> int:
    return sum(per_item.get(k, 0) * inv[k] for k in inv)


def expand_inventory(inv: Counter) -> List[str]:
    items: List[str] = []
    for k, c in sorted(inv.items()):
        for _ in range(c):
            items.append(k)
    return items


def pareto_frontier(
    items: List[str], p_mia: Dict[str, int], p_ava: Dict[str, int]
) -> Tuple[List[Tuple[int, int]], int]:
    """Return sorted unique (mia_score, ava_score) on Pareto frontier + count of all splits."""
    all_pts: List[Tuple[int, int]] = []
    for mask in product([0, 1], repeat=len(items)):
        mia_inv = Counter()
        ava_inv = Counter()
        for it, assign_mia in zip(items, mask):
            if assign_mia:
                mia_inv[it] += 1
            else:
                ava_inv[it] += 1
        s1 = score(mia_inv, p_mia)
        s2 = score(ava_inv, p_ava)
        all_pts.append((s1, s2))
    pts_set = list(set(all_pts))

    def dominated(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return (b[0] >= a[0] and b[1] > a[1]) or (b[0] > a[0] and b[1] >= a[1])

    frontier = [p for p in pts_set if not any(dominated(p, q) for q in pts_set if q != p)]
    frontier.sort()
    return frontier, len(all_pts)


def on_frontier(point: Tuple[int, int], frontier: List[Tuple[int, int]]) -> bool:
    return point in set(frontier)


def main() -> None:
    combos = load_json(COMBOS_PATH)
    envs = load_json(ENVS_PATH)

    def source(ep: int) -> str:
        eid = combos[ep]["env_id"]
        return envs[eid].get("source", "")

    # Final splits manually verified from last agreement turns in each *_temp0.7_seed0.csv
    # (Mia inventory, Ava inventory) as Counter of canonical keys.
    hand_split: Dict[int, Tuple[Optional[Counter], Optional[Counter], str]] = {
        10: (
            Counter({"book": 4, "hat": 2}),
            Counter({"hat": 4, "ball": 2}),
            "Turn 4: Mia takes books + two hats; Ava takes four hats + balls.",
        ),
        25: (
            Counter({"book": 3, "ball": 1}),
            Counter({"hat": 2}),
            "Turn 2: Mia books+ball; Ava both hats.",
        ),
        26: (
            Counter({"book": 1, "ball": 1}),
            Counter({"hat": 3, "ball": 1}),
            "Turn 6: Mia book + one ball; Ava all hats + other ball.",
        ),
        31: (
            Counter({"book": 3}),
            Counter({"hat": 2, "ball": 1}),
            "Turn 5: Mia all books; Ava hats + ball.",
        ),
        42: (
            Counter({"apple": 1, "banana": 4, "orange": 1}),
            Counter({"orange": 2}),
            "Turn 5: Ava two oranges; Mia apple + four bananas + remaining orange.",
        ),
        44: (
            Counter({"banana": 2, "orange": 1}),
            Counter({"apple": 3}),
            "Turn 2: Mia bananas+orange; Ava all apples.",
        ),
        83: (
            Counter({"hat": 1, "ball": 2}),
            Counter({"book": 1, "hat": 2}),
            "Turn 9: Ava book + two hats; Mia one hat + both balls.",
        ),
    }

    print("Pareto check: deal-or-no-deal episodes (Gemini dialog CSVs)\n")
    for ep in DOND_EPISODES:
        if source(ep) != "deal-or-no-deal":
            print(f"Episode {ep}: SKIP (source={source(ep)!r})")
            continue

        intro, turns = read_dialog_csv(ep)
        if not intro:
            print(f"Episode {ep}: NO CSV — cannot evaluate.")
            continue

        scenario_m = re.search(r"Scenario:\s*(.+?)\n", intro, re.DOTALL)
        scenario = scenario_m.group(1).strip() if scenario_m else ""
        inv = parse_inventory_from_scenario(scenario)
        p_mia, p_ava = parse_valuations_from_intro(intro)
        for d in (p_mia, p_ava):
            for k in list(d.keys()):
                if k not in inv:
                    del d[k]
        items = expand_inventory(inv)
        frontier, n_splits = pareto_frontier(items, p_mia, p_ava)

        print("=" * 72)
        print(f"Episode {ep}")
        print(f"  scenario: {scenario[:100]}{'…' if len(scenario) > 100 else ''}")
        print(f"  inventory: {dict(inv)}")
        print(f"  Mia $/item: {p_mia}")
        print(f"  Ava $/item: {p_ava}")
        print(f"  exhaustive assignments: {n_splits}  |  Pareto points: {len(frontier)}")

        if ep == 53:
            print("  outcome: NO AGREEMENT (single truncated turn). Not Pareto-evaluable.")
            continue

        if ep == 88:
            # Two late proposals without clear acceptance
            cand_a_mia = Counter({"orange": 1})
            cand_a_ava = Counter({"apple": 3, "banana": 2})
            note_a = "Turn 13 (Ava offer): Mia orange; Ava 3 apples + 2 bananas."
            cand_b_mia = Counter({"orange": 1, "banana": 1})
            cand_b_ava = Counter({"apple": 3, "banana": 1})
            note_b = "Turn 14 (Mia counter): Mia orange+1 banana; Ava 3 apples+1 banana."
            for label, cm, ca, note in [
                ("A", cand_a_mia, cand_a_ava, note_a),
                ("B", cand_b_mia, cand_b_ava, note_b),
            ]:
                sm = score(cm, p_mia)
                sa = score(ca, p_ava)
                ok = on_frontier((sm, sa), frontier)
                print(f"  candidate {label}: Mia {dict(cm)} | Ava {dict(ca)}")
                print(f"    scores (Mia, Ava)=({sm}, {sa})  Pareto-optimal? {ok}")
                print(f"    {note}")
            print("  outcome: NEGOTIATION OPEN (no terminal deal in log).")
            continue

        if ep not in hand_split:
            if ep == 5:
                print("  outcome: missing dialog CSV in checkout.")
            continue

        cm, ca, note = hand_split[ep]
        assert cm is not None and ca is not None
        if cm + ca != inv:
            print(f"  WARNING: split sum {dict(cm+ca)} != inventory {dict(inv)}")
        sm, sa = score(cm, p_mia), score(ca, p_ava)
        ok = on_frontier((sm, sa), frontier)
        print(f"  agreed split: Mia {dict(cm)} | Ava {dict(ca)}")
        print(f"  scores (Mia, Ava)=({sm}, {sa})  Pareto-optimal? {ok}")
        print(f"  source_turns: {note}")

    print("\nDone.")


if __name__ == "__main__":
    main()
