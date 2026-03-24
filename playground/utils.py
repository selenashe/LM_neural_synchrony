import os

# Parent of playground/ = repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import random
import json

def enh_print(x, color="green"):
    if color == "green":
        print(f"\033[92m{x}\033[0m")
    elif color == "red":
        print(f"\033[91m{x}\033[0m")
    elif color == "yellow":
        print(f"\033[93m{x}\033[0m")
    else:
        print(x)

def create_folder_if_not_there(path): 
    if not os.path.exists(path):
      os.makedirs(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_short_names_and_identifiers(models):

    def shorten(x):
        first = False
        if x[0] == "M":
            first = True
        a_ver = "Mis" + x.split('Mistral-7B-Instruct-')[1].split('Llama')[0].split('_')[0]
        a_mode = ''.join(x.split('Mistral-7B-Instruct-')[1].split('Llama')[0].split('interv')[0].split('_')[1:])
        b_ver = "Lla" + x.split('Llama-')[1].split('-')[0]
        b_mode = ''.join(x.split('Llama-')[1].split('Mistral')[0].split('interv')[0].split('_')[1:])
        if b_mode == "None0":
            b_mode = ""
        if a_mode == "None0":
            a_mode = ""
        if "Llama-3.2-3B-Instruct" in x:
            b_ver += "-3B"
        if first:
            return a_ver + a_mode + b_ver + b_mode
        else:
            return b_ver + b_mode + a_ver + a_mode

    short_names = []
    models_identifier = ""
    for m in models:
        if "Mistral" in m and "Llama" in m:
            model_short_name = shorten(m)
            if "diff_backbones" not in models_identifier:
                models_identifier += "diff_backbones"
        elif "Qwen" not in m:
            if "Mistral" in m:
                a_ver = "Mis" + m.split('Mistral-7B-Instruct-')[1].split('_')[0]
                a_mode = ''.join(m.split('Mistral-7B-Instruct-')[1].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                b_ver = "Mis" + m.split('Mistral-7B-Instruct-')[2].split('_')[0]
                b_mode = ''.join(m.split('Mistral-7B-Instruct-')[2].split('interv')[0].split('_')[1:])
                if b_mode == "None0":
                    b_mode = ""
                model_short_name = a_ver + a_mode + b_ver + b_mode
            if "Llama" in m:
                a_ver = "Lla" + m.split('Llama-')[1].split('-')[0]
                a_mode = ''.join(m.split('Llama-')[1].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                b_ver = "Lla" + m.split('Llama-')[2].split('-')[0]
                b_mode = ''.join(m.split('Llama-')[2].split('interv')[0].split('_')[1:])
                if b_mode == "None0":
                    b_mode = ""
                model_short_name = a_ver + a_mode + b_ver + b_mode
        else:
            if "Mistral" in m:
                first = False
                if m[0] == "Q":
                    first = True
                a_ver = "Mis" + m.split('Mistral-7B-Instruct-')[1].split('Qwen')[0].split('_')[0]
                a_mode = ''.join(m.split('Mistral-7B-Instruct-')[1].split('Qwen')[0].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                model_short_name = "Q3" + a_ver + a_mode if first else a_ver + a_mode + "Q3"
            if "Llama" in m:
                first = False
                if m[0] == "Q":
                    first = True
                a_ver = "Lla" + m.split('Llama-')[1].split('-')[0]
                a_mode = ''.join(m.split('Llama-')[1].split('Qwen')[0].split('interv')[0].split('_')[1:])
                if a_mode == "None0":
                    a_mode = ""
                model_short_name = "Q3" + a_ver + a_mode if first else a_ver + a_mode + "Q3"
        model_short_name = model_short_name.replace('None0', '')
        short_names.append(model_short_name)

    if len(models) == 1:
        models_identifier = short_names[0]
    # print(models, short_names)
    return short_names, models_identifier


def _discover_sotopia_reference_pair_path():
    sr = os.path.join(REPO_ROOT, "sotopia_results")
    if not os.path.isdir(sr):
        raise FileNotFoundError(f"Missing {sr}; run Stage A simulation first.")

    env_name = os.environ.get("SOTOPIA_REFERENCE_RUN")
    default = "Llama-3-8B-Instruct_None_0_Llama-2-7B-Chat_None_0"
    for name in filter(None, [env_name, default]):
        p = os.path.join(sr, name)
        if os.path.isdir(p):
            return p

    candidates = sorted(
        d
        for d in os.listdir(sr)
        if os.path.isdir(os.path.join(sr, d)) and d != "dialogs"
    )
    if not candidates:
        raise FileNotFoundError(f"No model pair directories under {sr}")
    chosen = os.path.join(sr, candidates[0])
    print(f"get_all_modes: using discovered pair dir {candidates[0]} (set SOTOPIA_REFERENCE_RUN to pin)")
    return chosen


def get_all_modes():
    pair_path = _discover_sotopia_reference_pair_path()
    modes = []
    for subdir in os.listdir(pair_path):
        if ".npy" in subdir:
            continue
        if not os.path.isdir(os.path.join(pair_path, subdir)):
            continue
        modes.append(subdir)

    episodes_id = {}
    for mode in modes:
        spath = os.path.join(pair_path, mode)
        all_items = os.listdir(spath)
        subfolders_id = [
            int(item.split("_")[-1])
            for item in all_items
            if os.path.isdir(os.path.join(spath, item))
        ]
        subfolders_id = sorted(subfolders_id)
        episodes_id[mode] = subfolders_id
    return modes, episodes_id

def get_scores(json_file):
    scores = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = data['result']
        d1 = []
        d1_sum = 0
        d1_goal = data["agent_1_evaluation"]["goal"][1]
        for k, v in data["agent_1_evaluation"].items():
            d1.append((k, v[1]))
            d1_sum += v[1]
        
        d2 = []
        d2_sum = 0
        d2_goal = data["agent_2_evaluation"]["goal"][1]
        for k, v in data["agent_2_evaluation"].items():
            d2.append((k, v[1]))
            d2_sum += v[1]
        
        scores["agent_1"] = d1
        scores["agent_2"] = d2

    return scores, d1_sum, d1_goal, d2_sum, d2_goal


def get_all_scores(json_file):
    scores = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
        data = data['result']
        
        d1 = []
        for k, v in data["agent_1_evaluation"].items():
            d1.append((k, v[1]))
        
        d2 = []
        for k, v in data["agent_2_evaluation"].items():
            d2.append((k, v[1]))
        
        scores["agent_1"] = d1
        scores["agent_2"] = d2
    return scores