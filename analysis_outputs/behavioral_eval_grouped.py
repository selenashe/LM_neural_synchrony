import json
from collections import defaultdict

CATEGORY_GROUPS = {
    "Negotiation (Craigslist)": [f"craigslist_bargains_{str(i).zfill(5)}" for i in range(10)] + ["sell_item"],
    "Item Division": ["divide_things", "divide_items", "divide_fruits"],
    "Mutual Friend Discovery": [f"mutual_friend_{str(i).zfill(5)}" for i in range(10)],
    "Donation/Charity": ["ask_for_donantion", "charity_donation", "donate_charity", "donate_funds",
                         "donate_money", "donate_to_cause", "matching_donation", "small_donation"],
    "Conflict Resolution": ["share_blanket", "music_preference", "disagree_on_movie", "dinner_decision",
                            "movie_to_watch", "sleep_arrangement", "share_stuff", "take_turns",
                            "compliment_conflict", "split_classes", "fire_trash", "tree_trimming",
                            "facetime_etiquettes", "interrupted_speech", "talk_loudly", "yell"],
    "Relationship/Social": ["borrow_money", "flirt_with_someone", "go_on_a_date", "new_friends",
                            "secret_feeling", "rekindle_relationship", "distance_friend",
                            "apology_and_acceptance", "emotional_support", "helping_hand",
                            "drink_less", "lying_affair", "confess_mistake", "reveal_secret",
                            "ask_gift_preference"],
    "Strategic/Game": ["prison_dilemma", "reveal_answer", "game_winning", "play_rights"],
    "Persuasion": ["join_trip", "play_hooky", "drive_sportscar", "break_bad_luck",
                   "correct_misinformation", "food_refusal", "free_stuff", "movie_scene",
                   "political_views", "revenge_plot", "unwelcome_guest"],
    "Professional": ["financial_report", "financial_support", "life_dilemma"],
}

def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)

def group_analysis(data, model_name):
    by_code = defaultdict(list)
    for key, ep in data.items():
        ts = ep.get("task_success", {})
        analysis = ep.get("analysis", {})
        codename = ts.get("codename", "unknown")
        by_code[codename].append({
            "success": ts.get("success", False),
            "a1_met": ts.get("agent_1_goal_met", False),
            "a2_met": ts.get("agent_2_goal_met", False),
            "a1_track": analysis.get("agent_1_goal_tracking", "unknown"),
            "a2_track": analysis.get("agent_2_goal_tracking", "unknown"),
        })

    print(f"\n{'='*70}")
    print(f"  GROUPED TASK ANALYSIS: {model_name}")
    print(f"{'='*70}")
    print(f"  {'Group':<35s} {'N':>3s} {'Succ%':>7s} {'A1Met%':>7s} {'A2Met%':>7s} {'A1Str%':>7s} {'A2Str%':>7s}")
    print(f"  {'-'*75}")

    assigned = set()
    group_stats = {}
    for group_name, codenames in sorted(CATEGORY_GROUPS.items()):
        eps = []
        for cn in codenames:
            eps.extend(by_code.get(cn, []))
            assigned.add(cn)
        if not eps:
            continue
        n = len(eps)
        succ = sum(1 for e in eps if e["success"])
        a1m = sum(1 for e in eps if e["a1_met"])
        a2m = sum(1 for e in eps if e["a2_met"])
        a1s = sum(1 for e in eps if e["a1_track"] == "strong")
        a2s = sum(1 for e in eps if e["a2_track"] == "strong")
        print(f"  {group_name:<35s} {n:>3d} {100*succ/n:>6.1f}% {100*a1m/n:>6.1f}% {100*a2m/n:>6.1f}% {100*a1s/n:>6.1f}% {100*a2s/n:>6.1f}%")
        group_stats[group_name] = {"n": n, "succ": succ, "a1m": a1m, "a2m": a2m, "a1s": a1s, "a2s": a2s}

    # Catch unassigned
    unassigned = []
    for cn in by_code:
        if cn not in assigned:
            unassigned.extend(by_code[cn])
    if unassigned:
        n = len(unassigned)
        succ = sum(1 for e in unassigned if e["success"])
        print(f"  {'Other/Unclassified':<35s} {n:>3d} {100*succ/n:>6.1f}%")

    return group_stats

def compare_grouped(g_stats, l_stats):
    print(f"\n{'='*70}")
    print(f"  GROUPED COMPARISON: GEMINI vs LLAMA-7B")
    print(f"{'='*70}")
    print(f"  {'Group':<30s} {'Gem N':>5s} {'Gem%':>6s} {'Lla N':>5s} {'Lla%':>6s} {'Delta':>6s}")
    print(f"  {'-'*60}")
    all_groups = sorted(set(list(g_stats.keys()) + list(l_stats.keys())))
    for g in all_groups:
        gs = g_stats.get(g, {"n": 0, "succ": 0})
        ls = l_stats.get(g, {"n": 0, "succ": 0})
        g_pct = 100*gs["succ"]/gs["n"] if gs["n"] > 0 else 0
        l_pct = 100*ls["succ"]/ls["n"] if ls["n"] > 0 else 0
        delta = g_pct - l_pct
        sign = "+" if delta > 0 else ""
        print(f"  {g:<30s} {gs['n']:>5d} {g_pct:>5.1f}% {ls['n']:>5d} {l_pct:>5.1f}% {sign}{delta:>4.1f}%")


if __name__ == "__main__":
    gemini_data = load_data("analysis_outputs/gemini_goal_tracking_by_file.json")
    llama_data = load_data("analysis_outputs/goal_tracking_llama7b_llama7b_by_file.json")

    g_stats = group_analysis(gemini_data, "Gemini 3.1 Pro Preview")
    l_stats = group_analysis(llama_data, "Llama-7B × Llama-7B")
    compare_grouped(g_stats, l_stats)
