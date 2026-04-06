"""
Classify all 450 Sotopia combo entries on a multi-dimensional alignment scale
using Gemini via Vertex AI.  Each combo is evaluated holistically using the
scenario, both agents' goals, and both agents' full profiles.

Outputs goal_alignment_labels.json keyed by combo index (0-449).

Usage:
    python goal_alignment_labels.py [--model vertex_ai/gemini-3.1-pro-preview] [--output goal_alignment_labels.json]
    python goal_alignment_labels.py --resume   # pick up where you left off

Requires gcloud auth for Vertex AI.
"""

import os
import re
import json
import argparse
import time
from litellm import completion

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENVS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/envs.json")
COMBOS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/env_agent_combos.json")
AGENTS_PATH = os.path.join(SCRIPT_DIR, "sotopia_utils/sotopia_data/agents.json")

RELATIONSHIP_LABELS = {
    0: "strangers",
    1: "know each other by name",
    2: "acquaintances",
    3: "friends",
    4: "romantic partners",
    5: "family members",
}

SYSTEM_PROMPT = """\
You are an expert at analyzing social interactions. You will be given a scenario, \
two agents' profiles (personality, values, secrets), and their private goals from \
a social simulation. Your task is to assess the interaction holistically.

Respond with ONLY a JSON object (no markdown fencing) with these fields:

- "alignment_score": float from -1.0 to 1.0
    -1.0 = directly opposed (zero-sum, one wins only if the other loses)
     0.0 = orthogonal/independent (goals don't interact)
     1.0 = fully aligned (both benefit from the same outcome)

- "goal_structure": one of:
    "zero_sum"               — one agent's gain is the other's loss
    "coordination"           — both benefit from reaching the same outcome
    "mixed_motive"           — elements of both cooperation and competition
    "persuasion"             — one agent tries to change the other's behavior or belief
    "information_asymmetry"  — one agent has info the other wants, or vice versa

- "value_compatibility": one of:
    "compatible"   — agents' personalities/values support cooperation in this scenario
    "neutral"      — agents' values are unrelated to the interaction
    "conflicting"  — agents' values create tension given the scenario

- "power_dynamic": one of:
    "symmetric"    — agents are on roughly equal footing
    "asymmetric"   — one agent has structural leverage (e.g. lender vs borrower, seller vs buyer, authority vs subordinate)

- "stakes": one of:
    "low"    — casual social interaction with minor consequences
    "medium" — moderate consequences (reputation, moderate money, social standing)
    "high"   — significant financial, relational, or moral consequences

- "reasoning": 1-2 sentence explanation (keep under 80 words)"""

USER_TEMPLATE = """\
Scenario: {scenario}
Relationship: {relationship}

--- Agent 1 ---
Name: {a1_name}
Age: {a1_age} | Occupation: {a1_occupation} | Gender: {a1_gender}
Personality: {a1_personality}
Values: {a1_values} | Decision style: {a1_decision}
Secret: {a1_secret}
Goal: {a1_goal}

--- Agent 2 ---
Name: {a2_name}
Age: {a2_age} | Occupation: {a2_occupation} | Gender: {a2_gender}
Personality: {a2_personality}
Values: {a2_values} | Decision style: {a2_decision}
Secret: {a2_secret}
Goal: {a2_goal}"""

VALID_GOAL_STRUCTURES = {"zero_sum", "coordination", "mixed_motive", "persuasion", "information_asymmetry"}
VALID_VALUE_COMPAT = {"compatible", "neutral", "conflicting"}
VALID_POWER = {"symmetric", "asymmetric"}
VALID_STAKES = {"low", "medium", "high"}


def clean_tags(text: str) -> str:
    """Strip XML-like viewer tags but keep extra_info content."""
    text = re.sub(r'<p viewer=[^>]*>', '', text)
    text = re.sub(r'</p>', '', text)
    text = re.sub(r'<clarification_hint>', '', text)
    text = re.sub(r'</clarification_hint>', '', text)
    text = re.sub(r'<strategy_hint>', '', text)
    text = re.sub(r'</strategy_hint>', '', text)
    text = re.sub(r'<extra_info>', '[Extra info: ', text)
    text = re.sub(r'</extra_info>', ']', text)
    return text.strip()


def format_agent(profile: dict) -> dict:
    """Extract prompt-relevant fields from an agent profile."""
    return {
        "name": f"{profile['first_name']} {profile['last_name']}",
        "age": profile["age"],
        "occupation": profile["occupation"],
        "gender": profile["gender"],
        "personality": profile["personality_and_values"],
        "values": ", ".join(profile.get("schwartz_personal_values", [])),
        "decision": profile.get("decision_making_style", "Unknown"),
        "secret": profile.get("secret", "None"),
    }


def classify_combo(model, scenario, relationship, agent1, agent2, goal_1, goal_2, max_retries=3):
    a1 = format_agent(agent1)
    a2 = format_agent(agent2)

    user_msg = USER_TEMPLATE.format(
        scenario=clean_tags(scenario),
        relationship=relationship,
        a1_name=a1["name"], a1_age=a1["age"], a1_occupation=a1["occupation"],
        a1_gender=a1["gender"], a1_personality=a1["personality"],
        a1_values=a1["values"], a1_decision=a1["decision"],
        a1_secret=a1["secret"], a1_goal=clean_tags(goal_1),
        a2_name=a2["name"], a2_age=a2["age"], a2_occupation=a2["occupation"],
        a2_gender=a2["gender"], a2_personality=a2["personality"],
        a2_values=a2["values"], a2_decision=a2["decision"],
        a2_secret=a2["secret"], a2_goal=clean_tags(goal_2),
    )

    prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"
    raw = ""

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                vertex_project="hs-soil-gemini",
                vertex_location="global",
                temperature=0,
                max_tokens=1024,
                timeout=120,
                num_retries=3,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)

            score = float(result["alignment_score"])
            assert -1.0 <= score <= 1.0, f"score {score} out of range"
            assert result["goal_structure"] in VALID_GOAL_STRUCTURES, \
                f"invalid goal_structure: {result['goal_structure']}"
            assert result["value_compatibility"] in VALID_VALUE_COMPAT, \
                f"invalid value_compatibility: {result['value_compatibility']}"
            assert result["power_dynamic"] in VALID_POWER, \
                f"invalid power_dynamic: {result['power_dynamic']}"
            assert result["stakes"] in VALID_STAKES, \
                f"invalid stakes: {result['stakes']}"
            return result

        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1} due to: {e}")
                time.sleep(1)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                print(f"  Raw response: {raw[:500]}")
                return {
                    "alignment_score": None,
                    "goal_structure": None,
                    "value_compatibility": None,
                    "power_dynamic": None,
                    "stakes": None,
                    "reasoning": f"PARSE_ERROR: {e}",
                    "raw_response": raw,
                }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  API error, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Label goal alignment for all Sotopia combos")
    parser.add_argument("--model", default="vertex_ai/gemini-3.1-pro-preview", help="LiteLLM model string")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "goal_alignment_labels.json"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-labeled combos")
    args = parser.parse_args()

    with open(ENVS_PATH) as f:
        envs = json.load(f)
    with open(COMBOS_PATH) as f:
        combos = json.load(f)
    with open(AGENTS_PATH) as f:
        agents = json.load(f)

    print(f"Loaded {len(combos)} combos, {len(envs)} environments, {len(agents)} agent profiles")

    labels = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            labels = json.load(f)
        print(f"Resuming: {len(labels)} combos already labeled")

    for i, combo in enumerate(combos):
        combo_key = str(i)
        if combo_key in labels:
            continue

        env = envs[combo["env_id"]]
        agent1 = agents[combo["agent_ids"][0]]
        agent2 = agents[combo["agent_ids"][1]]
        relationship = RELATIONSHIP_LABELS.get(env.get("relationship", 0), "unknown")

        print(f"[{i + 1}/{len(combos)}] combo {i} | env={combo['env_id'][:12]}... | "
              f"source={env['source']} | {agent1['first_name']} vs {agent2['first_name']}")

        result = classify_combo(
            args.model,
            env["scenario"],
            relationship,
            agent1, agent2,
            env["agent_goals"][0],
            env["agent_goals"][1],
        )

        labels[combo_key] = {
            "env_id": combo["env_id"],
            "source": env["source"],
            "agent_1": f"{agent1['first_name']} {agent1['last_name']}",
            "agent_2": f"{agent2['first_name']} {agent2['last_name']}",
            "alignment_score": result["alignment_score"],
            "goal_structure": result["goal_structure"],
            "value_compatibility": result["value_compatibility"],
            "power_dynamic": result["power_dynamic"],
            "stakes": result["stakes"],
            "reasoning": result["reasoning"],
        }

        if (i + 1) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump(labels, f, indent=2)
            print(f"  Checkpoint saved ({len(labels)} labeled)")

    with open(args.output, "w") as f:
        json.dump(labels, f, indent=2)

    n_labeled = sum(1 for v in labels.values() if v["alignment_score"] is not None)
    print(f"\nDone. {n_labeled}/{len(combos)} successfully labeled.")
    print(f"Output: {args.output}")

    from collections import Counter
    scores = [v["alignment_score"] for v in labels.values() if v["alignment_score"] is not None]
    for dim in ["goal_structure", "value_compatibility", "power_dynamic", "stakes"]:
        dist = Counter(v[dim] for v in labels.values() if v[dim])
        print(f"\n{dim}: {dict(dist)}")
    if scores:
        print(f"\nAlignment score: mean={sum(scores)/len(scores):.3f}, "
              f"min={min(scores):.2f}, max={max(scores):.2f}")


if __name__ == "__main__":
    main()
