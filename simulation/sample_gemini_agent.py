#!/usr/bin/env python3
"""
Run Sotopia simulations with Gemini on Vertex AI (same episode loop as sample_normal_agent.py).

Uses private per-agent intros, alternating turns, max_turns / temperature / max_new_tokens
aligned with sample_normal_agent.py. Writes the same dialog CSV + prompt_records layout
under sotopia_results_gemini/dialogs/ (no hidden-state .npy files).
Defaults to the fixed two-agent combos file:
  sotopia_utils/sotopia_data/env_agent_combos_fixed_two_agents.json

Vertex / LiteLLM settings match goal_alignment_labels.classify_combo:
  vertex_ai/gemini-3.1-pro-preview, project hs-soil-gemini, location global, timeout 120.

Requires gcloud / Vertex auth appropriate for LiteLLM (same as goal_alignment_labels.py).

Smoke test (few episodes):
  cd /juice6/u/jshe/nlp/LM_neural_synchrony
  conda activate neural_sync
  export SOTOPIA_RUN_LABEL=mia_ava_fixed_two_agents  # optional
  python sample_gemini_agent.py --model_1 gemini-3.1-pro-preview --model_2 gemini-3.1-pro-preview \\
    --max_episodes 2

Outputs (same naming as sample_normal_agent.py, under sotopia_results_gemini):
  sotopia_results_gemini/dialogs/<model_1>_<None>_0_<model_2>_<None>_0[_<run_label>]/
    <episode_id>_temp0.7_seed<seed>.csv
    prompt_records/<episode_id>_temp0.7_seed<seed>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from copy import copy

from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiment_config import EPISODES_NUM, STAGE_A_SEEDS
from simulation.gemini_sot_env import (
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_MODEL,
    DEFAULT_VERTEX_PROJECT,
    GeminiSotopiaEnv,
)
from core.utils import REPO_ROOT, create_folder_if_not_there, set_seed

DEFAULT_GEMINI_ENV_AGENT_COMBOS_PATH = os.path.join(
    REPO_ROOT, "sotopia_utils", "sotopia_data", "env_agent_combos_fixed_two_agents.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Sotopia episodes with Gemini/Vertex (mirrors sample_normal_agent.py outputs)"
    )
    parser.add_argument(
        "--model_1",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Agent A label used in output folder naming (generation uses --vertex_model).",
    )
    parser.add_argument(
        "--model_2",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Agent B label used in output folder naming (generation uses --vertex_model_2 or --vertex_model).",
    )
    parser.add_argument("--affected_type", type=str, default="None")
    parser.add_argument("--value", type=float, default=0)
    parser.add_argument(
        "--run_label",
        type=str,
        default=os.environ.get("SOTOPIA_RUN_LABEL", ""),
        help="Optional suffix for sotopia_results_gemini paths (default: SOTOPIA_RUN_LABEL env).",
    )
    parser.add_argument(
        "--vertex_model",
        type=str,
        default=DEFAULT_VERTEX_MODEL,
        help="LiteLLM model id (default: vertex_ai/gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--vertex_model_2",
        type=str,
        default="",
        help="Optional second Vertex model for agent B (default: same as --vertex_model)",
    )
    parser.add_argument("--vertex_project", type=str, default=DEFAULT_VERTEX_PROJECT)
    parser.add_argument("--vertex_location", type=str, default=DEFAULT_VERTEX_LOCATION)
    parser.add_argument(
        "--env_agent_combos_path",
        type=str,
        default=DEFAULT_GEMINI_ENV_AGENT_COMBOS_PATH,
        help="Path to env-agent combos JSON (default: fixed_two_agents list).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_turns", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="If set, cap episodes at this count (for smoke tests). Default: EPISODES_NUM from experiment_config.",
    )
    args = parser.parse_args()

    if args.value == 0.0:
        args.value = 0

    model_1_name = f"{args.model_1}_{args.affected_type}_{args.value}"
    model_2_name = f"{args.model_2}_{args.affected_type}_{args.value}"

    _run = (args.run_label or os.environ.get("SOTOPIA_RUN_LABEL", "") or "").strip()
    _leaf = f"{model_1_name}_{model_2_name}" + (f"_{_run}" if _run else "")

    results_gemini_root = os.path.join(REPO_ROOT, "sotopia_results_gemini")
    save_dir_base = os.path.join(results_gemini_root, _leaf)
    dialog_base = os.path.join(results_gemini_root, "dialogs", _leaf)
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
            env_agent_combos_path=args.env_agent_combos_path,
            max_turns=args.max_turns,
            saving_dir_base=save_dir_base,
            save_states=False,
            save_states_only_dialog=False,
            save_states_dialog_imperson=False,
            save_states_without_goal=False,
            probing_goal=False,
            probing_self_goal=False,
            max_new_tokens=args.max_new_tokens,
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
