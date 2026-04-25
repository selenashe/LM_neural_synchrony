import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import csv
from simulation.sot_env import SotopiaEnv
from core.utils import *
from experiment_config import EPISODES_NUM, STAGE_A_SEEDS, RESULTS_DIR
from tqdm import tqdm
from copy import copy
import argparse

parser = argparse.ArgumentParser(description='Run Sotopia environment with specified affected types array')
parser.add_argument('--model_1', type=str, default="Mistral-7B-Instruct-v0.3")
parser.add_argument('--model_2', type=str, default="Mistral-7B-Instruct-v0.3")
parser.add_argument('--affected_type', type=str, default="None")
parser.add_argument('--value', type=float, default=0)
parser.add_argument(
    '--run_label',
    type=str,
    default=os.environ.get('SOTOPIA_RUN_LABEL', ''),
    help='Optional suffix for sotopia_results paths (default: SOTOPIA_RUN_LABEL env if set) so runs do not overwrite prior outputs.',
)
args = parser.parse_args()

THINKING_MODELS = {"Qwen_Qwen3-14B", "Qwen3-8B", "DeepSeek-R1-Distill-Llama-8B"}
_is_thinking = args.model_1 in THINKING_MODELS or args.model_2 in THINKING_MODELS
max_new_tokens = 4096 if _is_thinking else 300
max_turns = 16
episodes_num = EPISODES_NUM
temp = 0.7
seeds = STAGE_A_SEEDS

if args.value == 0.0:
    args.value = 0

model_1_name = f"{args.model_1}_{args.affected_type}_{args.value}"
model_2_name = f"{args.model_2}_{args.affected_type}_{args.value}"

_run = (args.run_label or os.environ.get('SOTOPIA_RUN_LABEL', '') or '').strip()
_leaf = f"{model_1_name}_{model_2_name}" + (f"_{_run}" if _run else '')

save_dir_base = os.path.join(RESULTS_DIR, _leaf)
dialog_base = os.path.join(RESULTS_DIR, "dialogs", _leaf)
# Do not mkdir(dialog_base) here: if SotopiaEnv fails (OOM, bad checkpoint), Slurm
# would leave an empty dialogs/<pair>/ tombstone. Create the folder only when we
# are about to write the first episode CSV (after env is known-good).

for seed in seeds:
    print(save_dir_base, seed)
    set_seed(seed)
    verbose = False
    turns = []

    env = SotopiaEnv(model_1_name=model_1_name, 
                        model_2_name=model_2_name, 
                        max_turns=max_turns, 
                        saving_dir_base=save_dir_base, 
                        save_states=False,
                        save_states_only_dialog=False,
                        save_states_dialog_imperson=False,
                        save_states_without_goal=False,
                        probing_goal=False,
                        probing_self_goal=False,
                        max_new_tokens=max_new_tokens,
                        verbose=verbose,
                        temperature=temp,
                        seed=seed)
    
    for episode_id in tqdm(range(episodes_num)):
        set_seed(seed)
        if os.path.exists(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv'):
            if os.path.getsize(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv') > 0:
                print(f"Episode {episode_id} already exists, skipping.")
                continue
        create_folder_if_not_there(dialog_base)
        with open(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Intro', 'Dialog', 'Intended Dialog'])

        done = False
        env.reset(env_id=episode_id)
        
        cur_turn = 0
        while not done:
            done = env.step()
            cur_turn = copy(env.cur_turn)

        turns.append(cur_turn)

        intro, dialog, intended_dialog = env.complete_intro, env.cur_dialog, env.intended_dialog
        with open(f'{dialog_base}/{episode_id}_temp0.7_seed{seed}.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([intro, dialog, intended_dialog])

        if any(t for t in env.thinking_traces):
            traces_dir = f"{dialog_base}/thinking_traces"
            create_folder_if_not_there(traces_dir)
            with open(f'{traces_dir}/{episode_id}_temp0.7_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Turn', 'Thinking_Trace'])
                for turn_idx, trace in enumerate(env.thinking_traces):
                    writer.writerow([turn_idx, trace])

        create_folder_if_not_there(f"{dialog_base}/prompt_records")
        with open(f'{dialog_base}/prompt_records/{episode_id}_temp0.7_seed{seed}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['save_states',
                            ])
            for i in range(cur_turn):
                writer.writerow([
                    env.prompts[i],
                ])

    print('turns of each episode:', turns)
    del env
                

