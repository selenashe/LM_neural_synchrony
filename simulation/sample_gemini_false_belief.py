#!/usr/bin/env python3
"""
Run false-belief Sotopia simulations with two Gemini agents on Vertex AI.

Two gemini-3.1-pro-preview agents converse for a free-determined number of
turns (cutoff at 15). No per-turn token limit is enforced — the model's own
output limit applies.

Uses the same episode loop, CSV layout, and prompt-record format as
sample_gemini_agent.py, targeting the false_belief_100 data files.

Requires the same gcloud / Vertex auth as sample_gemini_agent.py.

Local run (all 600 episodes):
  cd /juice6/u/jshe/nlp/LM_neural_synchrony
  conda activate neural_sync
  export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
  export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
  export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief_100
  export SOTOPIA_RUN_LABEL=false_belief_fixed_two_agents
  python simulation/sample_gemini_false_belief.py

Smoke test:
  python simulation/sample_gemini_false_belief.py --max_episodes 2
"""

from __future__ import annotations

import argparse
import csv
import os
from copy import copy

from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiment_config import EPISODES_NUM, STAGE_A_SEEDS, RESULTS_DIR
from simulation.gemini_sot_env import (
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_MODEL,
    DEFAULT_VERTEX_PROJECT,
    GeminiSotopiaEnv,
)
from core.utils import REPO_ROOT, create_folder_if_not_there, set_seed


def main():
    parser = argparse.ArgumentParser(
        description="False-belief Sotopia episodes with two Gemini/Vertex agents"
    )
    parser.add_argument(
        "--model_1",
        type=str,
        default="gemini-3.1-pro-preview",
    )
    parser.add_argument(
        "--model_2",
        type=str,
        default="gemini-3.1-pro-preview",
    )
    parser.add_argument("--affected_type", type=str, default="None")
    parser.add_argument("--value", type=float, default=0)
    parser.add_argument(
        "--run_label",
        type=str,
        default=os.environ.get("SOTOPIA_RUN_LABEL", ""),
        help="Optional suffix for output paths (default: SOTOPIA_RUN_LABEL env).",
    )
    parser.add_argument(
        "--vertex_model",
        type=str,
        default=DEFAULT_VERTEX_MODEL,
        help="LiteLLM model id for both agents (default: vertex_ai/gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--vertex_model_2",
        type=str,
        default="",
        help="Optional second Vertex model for agent B (default: same as --vertex_model)",
    )
    parser.add_argument("--vertex_project", type=str, default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--vertex_location", type=str, default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Cap episodes (for smoke tests). Default: all episodes from combos file.",
    )
    args = parser.parse_args()

    if args.value == 0.0:
        args.value = 0

    model_1_name = f"{args.model_1}_{args.affected_type}_{args.value}"
    model_2_name = f"{args.model_2}_{args.affected_type}_{args.value}"

    _run = (args.run_label or os.environ.get("SOTOPIA_RUN_LABEL", "") or "").strip()
    _leaf = f"{model_1_name}_{model_2_name}" + (f"_{_run}" if _run else "")

    dialog_base = os.path.join(RESULTS_DIR, "dialogs", _leaf)
    save_dir_base = os.path.join(RESULTS_DIR, _leaf)
    create_folder_if_not_there(dialog_base)

    episodes_num = EPISODES_NUM if args.max_episodes is None else min(EPISODES_NUM, args.max_episodes)
    seeds = STAGE_A_SEEDS
    temp = args.temperature

    vm2 = args.vertex_model_2.strip() or None

    for seed in seeds:
        print(save_dir_base, seed)
        set_seed(seed)
        verbose = False

        env = GeminiSotopiaEnv(
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            vertex_model_1=args.vertex_model,
            vertex_model_2=vm2,
            vertex_project=args.vertex_project,
            vertex_location=args.vertex_location,
            max_turns=args.max_turns,
            saving_dir_base=save_dir_base,
            save_states=False,
            save_states_only_dialog=False,
            save_states_dialog_imperson=False,
            save_states_without_goal=False,
            probing_goal=False,
            probing_self_goal=False,
            max_new_tokens=None,
            verbose=verbose,
            temperature=temp,
            seed=seed,
        )

        turns = []
        for episode_id in tqdm(range(episodes_num)):
            set_seed(seed)
            out_csv = f"{dialog_base}/{episode_id}_temp{temp}_seed{seed}.csv"
            if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                print(f"Episode {episode_id} already exists, skipping.")
                continue
            with open(out_csv, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Intro", "Dialog", "Intended Dialog"])

            done = False
            env.reset(env_id=episode_id)

            cur_turn = 0
            while not done:
                done = env.step()
                cur_turn = copy(env.cur_turn)

            turns.append(cur_turn)

            intro, dialog, intended_dialog = env.complete_intro, env.cur_dialog, env.intended_dialog
            with open(out_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([intro, dialog, intended_dialog])

            create_folder_if_not_there(f"{dialog_base}/prompt_records")
            pr_path = f"{dialog_base}/prompt_records/{episode_id}_temp{temp}_seed{seed}.csv"
            with open(pr_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["save_states"])
                for i in range(cur_turn):
                    writer.writerow([env.prompts[i]])

        print("turns of each episode:", turns)
        del env


if __name__ == "__main__":
    main()
