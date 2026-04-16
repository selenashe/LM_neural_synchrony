import json
from collections import defaultdict, Counter

def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)

def analyze_model(data, model_name):
    results = {
        "model": model_name,
        "total_episodes": 0,
        "task_success": {"success": 0, "fail": 0},
        "agent1_goal": {"met": 0, "not_met": 0},
        "agent2_goal": {"met": 0, "not_met": 0},
        "goal_tracking": {"agent1": Counter(), "agent2": Counter()},
        "by_category": defaultdict(lambda: {
            "count": 0, "success": 0,
            "a1_met": 0, "a2_met": 0,
            "a1_tracking": Counter(), "a2_tracking": Counter()
        }),
        "goal_tracking_vs_success": {
            "both_strong_success": 0, "both_strong_total": 0,
            "any_poor_success": 0, "any_poor_total": 0,
            "any_mixed_success": 0, "any_mixed_total": 0,
        },
        "unnatural_behavior_counts": Counter(),
        "issue_episodes": 0,
    }

    for key, episode in data.items():
        ts = episode.get("task_success", {})
        analysis = episode.get("analysis", {})
        codename = ts.get("codename", "unknown")
        results["total_episodes"] += 1

        success = ts.get("success", False)
        a1_met = ts.get("agent_1_goal_met", False)
        a2_met = ts.get("agent_2_goal_met", False)
        a1_track = analysis.get("agent_1_goal_tracking", "unknown")
        a2_track = analysis.get("agent_2_goal_tracking", "unknown")

        if success:
            results["task_success"]["success"] += 1
        else:
            results["task_success"]["fail"] += 1

        if a1_met:
            results["agent1_goal"]["met"] += 1
        else:
            results["agent1_goal"]["not_met"] += 1

        if a2_met:
            results["agent2_goal"]["met"] += 1
        else:
            results["agent2_goal"]["not_met"] += 1

        results["goal_tracking"]["agent1"][a1_track] += 1
        results["goal_tracking"]["agent2"][a2_track] += 1

        cat = results["by_category"][codename]
        cat["count"] += 1
        if success:
            cat["success"] += 1
        if a1_met:
            cat["a1_met"] += 1
        if a2_met:
            cat["a2_met"] += 1
        cat["a1_tracking"][a1_track] += 1
        cat["a2_tracking"][a2_track] += 1

        # Goal tracking vs success relationship
        if a1_track == "strong" and a2_track == "strong":
            results["goal_tracking_vs_success"]["both_strong_total"] += 1
            if success:
                results["goal_tracking_vs_success"]["both_strong_success"] += 1
        if a1_track == "poor" or a2_track == "poor":
            results["goal_tracking_vs_success"]["any_poor_total"] += 1
            if success:
                results["goal_tracking_vs_success"]["any_poor_success"] += 1
        if a1_track == "mixed" or a2_track == "mixed":
            results["goal_tracking_vs_success"]["any_mixed_total"] += 1
            if success:
                results["goal_tracking_vs_success"]["any_mixed_success"] += 1

        if analysis.get("issues"):
            results["issue_episodes"] += 1

        for behavior in analysis.get("unnatural_behaviors", []):
            b_lower = behavior.lower()
            if "repetit" in b_lower:
                results["unnatural_behavior_counts"]["repetition"] += 1
            elif "cut" in b_lower or "truncat" in b_lower:
                results["unnatural_behavior_counts"]["truncation/cut-off"] += 1
            elif "hallucin" in b_lower:
                results["unnatural_behavior_counts"]["hallucination"] += 1
            elif "meta" in b_lower or "break" in b_lower:
                results["unnatural_behavior_counts"]["breaking character/meta"] += 1
            elif "leak" in b_lower or "private" in b_lower or "point" in b_lower:
                results["unnatural_behavior_counts"]["leaking private info"] += 1
            else:
                results["unnatural_behavior_counts"]["other"] += 1

    return results


def print_report(r):
    n = r["total_episodes"]
    print(f"\n{'='*70}")
    print(f"  BEHAVIORAL EVALUATION: {r['model']}")
    print(f"{'='*70}")
    print(f"  Total episodes: {n}")

    # Task success
    s = r["task_success"]["success"]
    print(f"\n--- TASK SUCCESS ---")
    print(f"  Overall success rate:     {s}/{n} ({100*s/n:.1f}%)")
    print(f"  Agent 1 goal met:         {r['agent1_goal']['met']}/{n} ({100*r['agent1_goal']['met']/n:.1f}%)")
    print(f"  Agent 2 goal met:         {r['agent2_goal']['met']}/{n} ({100*r['agent2_goal']['met']/n:.1f}%)")
    both_met = sum(1 for key, ep in [] for _ in [])  # placeholder
    # Compute both met from raw
    print(f"  Episodes w/ issues:       {r['issue_episodes']}/{n} ({100*r['issue_episodes']/n:.1f}%)")

    # Goal tracking
    print(f"\n--- GOAL TRACKING QUALITY ---")
    for agent_key, label in [("agent1", "Agent 1"), ("agent2", "Agent 2")]:
        c = r["goal_tracking"][agent_key]
        total = sum(c.values())
        print(f"  {label}:")
        for level in ["strong", "mixed", "poor"]:
            cnt = c.get(level, 0)
            print(f"    {level:8s}: {cnt}/{total} ({100*cnt/total:.1f}%)")

    # Goal tracking vs success
    print(f"\n--- GOAL TRACKING → TASK SUCCESS ---")
    gvs = r["goal_tracking_vs_success"]
    if gvs["both_strong_total"] > 0:
        print(f"  Both agents strong:  {gvs['both_strong_success']}/{gvs['both_strong_total']} success ({100*gvs['both_strong_success']/gvs['both_strong_total']:.1f}%)")
    if gvs["any_mixed_total"] > 0:
        print(f"  Any agent mixed:     {gvs['any_mixed_success']}/{gvs['any_mixed_total']} success ({100*gvs['any_mixed_success']/gvs['any_mixed_total']:.1f}%)")
    if gvs["any_poor_total"] > 0:
        print(f"  Any agent poor:      {gvs['any_poor_success']}/{gvs['any_poor_total']} success ({100*gvs['any_poor_success']/gvs['any_poor_total']:.1f}%)")

    # Unnatural behaviors
    print(f"\n--- UNNATURAL BEHAVIOR CATEGORIES ---")
    for beh, cnt in r["unnatural_behavior_counts"].most_common():
        print(f"  {beh:30s}: {cnt} episodes")

    # By category
    print(f"\n--- BREAKDOWN BY TASK CATEGORY ---")
    print(f"  {'Category':<30s} {'N':>3s} {'Success':>8s} {'A1 Met':>8s} {'A2 Met':>8s} {'A1 Track':>12s} {'A2 Track':>12s}")
    print(f"  {'-'*85}")
    cats = sorted(r["by_category"].items(), key=lambda x: x[0])
    for cat_name, cat_data in cats:
        c = cat_data["count"]
        s_rate = f"{cat_data['success']}/{c}"
        a1_rate = f"{cat_data['a1_met']}/{c}"
        a2_rate = f"{cat_data['a2_met']}/{c}"
        a1_t = cat_data["a1_tracking"]
        a2_t = cat_data["a2_tracking"]
        a1_str = f"S{a1_t.get('strong',0)}M{a1_t.get('mixed',0)}P{a1_t.get('poor',0)}"
        a2_str = f"S{a2_t.get('strong',0)}M{a2_t.get('mixed',0)}P{a2_t.get('poor',0)}"
        print(f"  {cat_name:<30s} {c:>3d} {s_rate:>8s} {a1_rate:>8s} {a2_rate:>8s} {a1_str:>12s} {a2_str:>12s}")


def compare_models(gemini_r, llama_r):
    print(f"\n{'='*70}")
    print(f"  COMPARATIVE SUMMARY: GEMINI vs LLAMA-7B")
    print(f"{'='*70}")
    
    gn = gemini_r["total_episodes"]
    ln = llama_r["total_episodes"]
    
    gs = gemini_r["task_success"]["success"]
    ls = llama_r["task_success"]["success"]
    
    print(f"\n{'Metric':<40s} {'Gemini':>12s} {'Llama-7B':>12s}")
    print(f"{'-'*65}")
    print(f"{'Total episodes':<40s} {gn:>12d} {ln:>12d}")
    print(f"{'Task success rate':<40s} {100*gs/gn:>11.1f}% {100*ls/ln:>11.1f}%")
    print(f"{'Agent 1 goal met rate':<40s} {100*gemini_r['agent1_goal']['met']/gn:>11.1f}% {100*llama_r['agent1_goal']['met']/ln:>11.1f}%")
    print(f"{'Agent 2 goal met rate':<40s} {100*gemini_r['agent2_goal']['met']/gn:>11.1f}% {100*llama_r['agent2_goal']['met']/ln:>11.1f}%")
    
    g_strong1 = gemini_r["goal_tracking"]["agent1"].get("strong", 0)
    l_strong1 = llama_r["goal_tracking"]["agent1"].get("strong", 0)
    g_strong2 = gemini_r["goal_tracking"]["agent2"].get("strong", 0)
    l_strong2 = llama_r["goal_tracking"]["agent2"].get("strong", 0)
    g_poor1 = gemini_r["goal_tracking"]["agent1"].get("poor", 0)
    l_poor1 = llama_r["goal_tracking"]["agent1"].get("poor", 0)
    g_poor2 = gemini_r["goal_tracking"]["agent2"].get("poor", 0)
    l_poor2 = llama_r["goal_tracking"]["agent2"].get("poor", 0)
    
    print(f"{'Agent 1 strong tracking':<40s} {100*g_strong1/gn:>11.1f}% {100*l_strong1/ln:>11.1f}%")
    print(f"{'Agent 2 strong tracking':<40s} {100*g_strong2/gn:>11.1f}% {100*l_strong2/ln:>11.1f}%")
    print(f"{'Agent 1 poor tracking':<40s} {100*g_poor1/gn:>11.1f}% {100*l_poor1/ln:>11.1f}%")
    print(f"{'Agent 2 poor tracking':<40s} {100*g_poor2/gn:>11.1f}% {100*l_poor2/ln:>11.1f}%")
    print(f"{'Episodes w/ issues':<40s} {100*gemini_r['issue_episodes']/gn:>11.1f}% {100*llama_r['issue_episodes']/ln:>11.1f}%")
    
    # Both strong → success comparison
    gvs_g = gemini_r["goal_tracking_vs_success"]
    gvs_l = llama_r["goal_tracking_vs_success"]
    
    print(f"\n--- Goal Tracking → Success Rate Comparison ---")
    for label, key in [("Both strong", "both_strong"), ("Any mixed", "any_mixed"), ("Any poor", "any_poor")]:
        g_tot = gvs_g[f"{key}_total"]
        l_tot = gvs_l[f"{key}_total"]
        g_suc = gvs_g[f"{key}_success"]
        l_suc = gvs_l[f"{key}_success"]
        g_pct = f"{100*g_suc/g_tot:.1f}%" if g_tot > 0 else "N/A"
        l_pct = f"{100*l_suc/l_tot:.1f}%" if l_tot > 0 else "N/A"
        print(f"  {label:<20s}: Gemini {g_suc}/{g_tot} ({g_pct}), Llama {l_suc}/{l_tot} ({l_pct})")

    # Category comparison
    all_cats = sorted(set(list(gemini_r["by_category"].keys()) + list(llama_r["by_category"].keys())))
    print(f"\n--- Task Success by Category (Gemini vs Llama) ---")
    print(f"  {'Category':<30s} {'Gemini':>12s} {'Llama-7B':>12s}")
    print(f"  {'-'*55}")
    for cat in all_cats:
        g_cat = gemini_r["by_category"].get(cat, {"count": 0, "success": 0})
        l_cat = llama_r["by_category"].get(cat, {"count": 0, "success": 0})
        gc = g_cat["count"]
        lc = l_cat["count"]
        g_str = f"{g_cat['success']}/{gc}" if gc > 0 else "---"
        l_str = f"{l_cat['success']}/{lc}" if lc > 0 else "---"
        print(f"  {cat:<30s} {g_str:>12s} {l_str:>12s}")


if __name__ == "__main__":
    gemini_data = load_data("analysis_outputs/gemini_goal_tracking_by_file.json")
    llama_data = load_data("analysis_outputs/goal_tracking_llama7b_llama7b_by_file.json")
    
    gemini_results = analyze_model(gemini_data, "Gemini 3.1 Pro Preview")
    llama_results = analyze_model(llama_data, "Llama-7B × Llama-7B")
    
    print_report(gemini_results)
    print_report(llama_results)
    compare_models(gemini_results, llama_results)
