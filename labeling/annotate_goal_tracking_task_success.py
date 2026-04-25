"""
Annotate goal-tracking JSON exports with Gemini (Vertex) judged task success per episode.

Episode index is parsed from each JSON key's CSV basename (e.g. ``42_temp0.7_seed0.csv`` -> 42)
and looked up in ``experiment_config.COMBOS_PATH`` (default:
``sotopia_utils/sotopia_data/env_agent_combos_fixed_two_agents.json``). That row supplies
``env_id`` and ``agent_ids``; scenario and goals come from ``envs.json``; agent profiles from
``agents.json``. Transcripts are read from disk when possible:

- Gemini keys (``.../prompt_records/N_temp....csv``): under ``--gemini-results-root``.
- Llama keys (``N_temp....csv`` only): under ``--llama-prompt-root``.

If the CSV is missing, ``transcript_preview`` from the goal-tracking JSON is used.

Each scenario object gains a ``task_success`` field::

    "task_success": {
        "success": true | false | null,
        "agent_1_goal_met": ...,
        "agent_2_goal_met": ...,
        "reasoning": "...",
        "episode_index": ...,
        "env_id": "...",
        "codename": "...",
        "transcript_source": "..."
    }

Vertex / LiteLLM call shape matches ``analyze_prompt_goal_tracking_gemini.py`` (same
``vertex_project``, ``vertex_location``, ``temperature=0``, repair path on bad JSON).

Dependencies (install in the environment you use to run this repo)::

    pip install 'litellm[google]' tenacity

Requires Application Default Credentials with access to the configured Vertex project
(``gcloud auth application-default login``).

Usage:
    python annotate_goal_tracking_task_success.py
    python annotate_goal_tracking_task_success.py --limit 3
    python annotate_goal_tracking_task_success.py --resume
    python annotate_goal_tracking_task_success.py --gemini-goal-json path/to/a.json --llama-goal-json path/to/b.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from litellm import completion

from analyze_prompt_goal_tracking_gemini import (
    DEFAULT_MODEL,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    _extract_first_json_object,
    _repair_json_with_model,
    extract_transcript_from_prompts,
    read_prompt_csv,
)
from experiment_config import COMBOS_PATH, ENVS_PATH, REPO_ROOT
from goal_alignment_labels import RELATIONSHIP_LABELS, clean_tags, format_agent

REPO_ROOT_PATH = Path(REPO_ROOT)
AGENTS_PATH = REPO_ROOT_PATH / "sotopia_utils" / "sotopia_data" / "agents.json"

TASK_SUCCESS_SYSTEM = """\
You are evaluating a finished two-agent dialogue from the Sotopia social simulation.

You are given the official scenario, each agent's private goal (from the environment), \
short agent profiles, and the dialogue transcript. Decide whether the **interaction** \
achieves a successful outcome relative to those goals.

Guidelines:
- If goals align or are cooperative, success means both make reasonable progress or reach a good joint outcome.
- If goals conflict (e.g. one wants money, the other to refuse), success is **not** both fully achieving incompatible ends; \
  judge whether the outcome is a plausible, stable resolution that reasonably reflects each side's goal (e.g. a clear loan decision, \
  a fair division, discovery of mutual friend, etc.).
- If the transcript is empty, incoherent, or mostly cut off, treat success as false.

Respond with ONLY a JSON object (no markdown fencing) with these fields:
- "task_success": boolean — your overall judgment for this episode.
- "agent_1_goal_met": boolean — whether agent 1's stated goal is substantially achieved.
- "agent_2_goal_met": boolean — whether agent 2's stated goal is substantially achieved.
- "reasoning": string, 2-5 sentences explaining the judgment with reference to the dialogue."""


def episode_index_from_key(key: str) -> int | None:
    name = key.rsplit("/", 1)[-1]
    m = re.match(r"^(\d+)_temp", name)
    return int(m.group(1)) if m else None


def load_json_dict(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_user_message(
    episode_index: int,
    env_id: str,
    codename: str,
    scenario: str,
    relationship: int,
    goal_1: str,
    goal_2: str,
    agent1: dict,
    agent2: dict,
    transcript: str,
) -> str:
    rel = RELATIONSHIP_LABELS.get(relationship, str(relationship))
    a1 = format_agent(agent1)
    a2 = format_agent(agent2)
    return f"""\
Episode index: {episode_index}
Environment id: {env_id}
Codename: {codename}
Relationship: {rel}

--- Scenario ---
{clean_tags(scenario)}

--- Agent 1 ({a1["name"]}) ---
Occupation: {a1["occupation"]} | Personality: {a1["personality"][:800]}
Private goal: {clean_tags(goal_1)}

--- Agent 2 ({a2["name"]}) ---
Occupation: {a2["occupation"]} | Personality: {a2["personality"][:800]}
Private goal: {clean_tags(goal_2)}

--- Dialogue transcript (extracted from logged prompts) ---
{transcript[:50000]}
"""


def call_task_success(
    model: str,
    user_msg: str,
    max_retries: int = 3,
) -> dict:
    raw = ""
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": TASK_SUCCESS_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                vertex_project=VERTEX_PROJECT,
                vertex_location=VERTEX_LOCATION,
                temperature=0,
                max_tokens=1024,
                timeout=120,
                num_retries=3,
            )
            raw = response.choices[0].message.content.strip()
            raw_json = _extract_first_json_object(raw)
            try:
                out = json.loads(raw_json)
            except json.JSONDecodeError:
                repaired = _repair_json_with_model(model, raw)
                if repaired is None:
                    raise
                out = repaired
                out["_repaired_json"] = True
            if "task_success" not in out:
                raise KeyError("missing task_success")
            return out
        except Exception as error:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  retry in {wait}s: {error}")
                time.sleep(wait)
            else:
                return {
                    "task_success": None,
                    "agent_1_goal_met": None,
                    "agent_2_goal_met": None,
                    "reasoning": f"ERROR: {error}",
                    "raw_response": raw[:4000],
                }
    return {}


def resolve_csv_path(key: str, gemini_results_root: Path, llama_prompt_root: Path) -> Path | None:
    if "prompt_records/" in key:
        return (gemini_results_root / key).resolve()
    return (llama_prompt_root / Path(key).name).resolve()


def annotate_file(
    goal_tracking_path: Path,
    gemini_results_root: Path,
    llama_prompt_root: Path,
    combos: list,
    envs: dict,
    agents: dict,
    model: str,
    resume: bool,
    limit: int | None,
) -> None:
    data = load_json_dict(goal_tracking_path)
    keys = list(data.keys())
    processed = 0

    for key in keys:
        if limit is not None and processed >= limit:
            break
        entry = data[key]
        if not isinstance(entry, dict):
            continue
        if resume:
            ts = entry.get("task_success")
            if isinstance(ts, dict) and isinstance(ts.get("success"), bool):
                continue

        ep = episode_index_from_key(key)
        if ep is None:
            print(f"[skip bad key] {key}")
            continue
        if ep < 0 or ep >= len(combos):
            print(f"[skip out of range] {key} -> {ep}")
            continue

        combo = combos[ep]
        env_id = combo["env_id"]
        env = envs.get(env_id)
        if not env:
            print(f"[skip missing env] {key} env_id={env_id}")
            continue
        aid1, aid2 = combo["agent_ids"][0], combo["agent_ids"][1]
        ag1, ag2 = agents.get(aid1), agents.get(aid2)
        if not ag1 or not ag2:
            print(f"[skip missing agent] {key}")
            continue

        csv_path = resolve_csv_path(key, gemini_results_root, llama_prompt_root)
        if csv_path is None or not csv_path.is_file():
            transcript = (entry.get("transcript_preview") or "")[:50000]
            source = "transcript_preview_only"
        else:
            rows = read_prompt_csv(csv_path)
            transcript = extract_transcript_from_prompts(rows) or (
                entry.get("transcript_preview") or ""
            )
            source = str(csv_path.relative_to(REPO_ROOT_PATH))

        goals = env.get("agent_goals") or ["", ""]
        g1 = goals[0] if len(goals) > 0 else ""
        g2 = goals[1] if len(goals) > 1 else ""

        user_msg = build_user_message(
            ep,
            env_id,
            str(env.get("codename", "")),
            str(env.get("scenario", "")),
            int(env.get("relationship", 0)),
            g1,
            g2,
            ag1,
            ag2,
            transcript,
        )

        print(f"[{processed + 1}] {key} (episode {ep}, {env.get('codename')})")
        result = call_task_success(model, user_msg)
        ts_val = result.get("task_success")
        data[key]["task_success"] = {
            "success": ts_val if isinstance(ts_val, bool) else None,
            "agent_1_goal_met": result.get("agent_1_goal_met"),
            "agent_2_goal_met": result.get("agent_2_goal_met"),
            "reasoning": result.get("reasoning"),
            "episode_index": ep,
            "env_id": env_id,
            "codename": env.get("codename"),
            "transcript_source": source,
        }
        if "_repaired_json" in result:
            data[key]["task_success"]["_repaired_json"] = True
        if "raw_response" in result:
            data[key]["task_success"]["raw_response"] = result["raw_response"]

        save_json(goal_tracking_path, data)
        processed += 1

    print(f"Updated {processed} entries in {goal_tracking_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gemini-goal-json",
        default=str(REPO_ROOT_PATH / "analysis_outputs" / "gemini_goal_tracking_by_file.json"),
    )
    parser.add_argument(
        "--llama-goal-json",
        default=str(REPO_ROOT_PATH / "analysis_outputs" / "goal_tracking_llama7b_llama7b_by_file.json"),
    )
    parser.add_argument(
        "--gemini-results-root",
        default=str(REPO_ROOT_PATH / "sotopia_results_gemini"),
    )
    parser.add_argument(
        "--llama-prompt-root",
        default=str(REPO_ROOT_PATH / "prompt_records_llama7b_llama7b"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max annotations per file")
    args = parser.parse_args()

    with Path(COMBOS_PATH).open(encoding="utf-8") as f:
        combos = json.load(f)
    with Path(ENVS_PATH).open(encoding="utf-8") as f:
        envs = json.load(f)
    with AGENTS_PATH.open(encoding="utf-8") as f:
        agents = json.load(f)

    gemini_root = Path(args.gemini_results_root)
    llama_root = Path(args.llama_prompt_root)

    for path_str in (args.gemini_goal_json, args.llama_goal_json):
        p = Path(path_str)
        if not p.is_file():
            raise SystemExit(f"Not found: {p}")
        annotate_file(
            p,
            gemini_root,
            llama_root,
            combos,
            envs,
            agents,
            args.model,
            args.resume,
            args.limit,
        )


if __name__ == "__main__":
    main()
