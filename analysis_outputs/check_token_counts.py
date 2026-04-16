"""Check which contrastive completion pairs have differing token counts."""
from transformers import AutoTokenizer
import json, os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATHS_JSON = os.path.join(REPO_ROOT, "model_paths.json")

with open(MODEL_PATHS_JSON) as f:
    paths = json.load(f)

tok = AutoTokenizer.from_pretrained(paths.get("Mistral-7B-Instruct-v0.2",
                                               "mistralai/Mistral-7B-Instruct-v0.2"),
                                    cache_dir=os.environ.get("LM_HF_CACHE"))

SCENARIO_VOCAB = {
    "key_in_boxes": {"item": "key", "locations": ("red box", "blue box"), "prep": "in the"},
    "wallet_desk_drawer": {"item": "wallet", "locations": ("desk", "drawer"), "prep": "in the"},
    "package_door_porch": {"item": "package", "locations": ("front door", "back porch"), "prep": "at the"},
}

a1, a2 = "Mia", "Ava"

for scenario_name, vocab in SCENARIO_VOCAB.items():
    item = vocab["item"]
    prep = vocab["prep"]
    loc_a, loc_b = vocab["locations"]

    classes = {
        "self_belief_location": [
            f"The {item} is {prep} {loc_a}",
            f"The {item} is {prep} {loc_b}",
        ],
        "self_think_location": [
            f"I think the {item} is {prep} {loc_a}",
            f"I think the {item} is {prep} {loc_b}",
        ],
        "self_believe_location": [
            f"I believe the {item} is {prep} {loc_a}",
            f"I believe the {item} is {prep} {loc_b}",
        ],
        "other_think_location": [
            f"The other person thinks the {item} is {prep} {loc_a}",
            f"The other person thinks the {item} is {prep} {loc_b}",
        ],
        "ava_think_location": [
            f"Ava thinks the {item} is {prep} {loc_a}",
            f"Ava thinks the {item} is {prep} {loc_b}",
        ],
        "mia_think_location": [
            f"Mia thinks the {item} is {prep} {loc_a}",
            f"Mia thinks the {item} is {prep} {loc_b}",
        ],
        "tell_location": [
            f"I should tell them the {item} is {prep} {loc_a}",
            f"I should tell them the {item} is {prep} {loc_b}",
        ],
        "want_check": [
            f"I want them to check the {loc_a}",
            f"I want them to check the {loc_b}",
        ],
        "will_check": [
            f"I will check the {loc_a}",
            f"I will check the {loc_b}",
        ],
        "self_honesty": [
            "I am being honest",
            "I am being deceptive",
        ],
        "other_honesty": [
            "The other person is being honest",
            "The other person is being deceptive",
        ],
        "other_strategy": [
            "The other person is trying to help",
            "The other person is trying to mislead",
        ],
        "ava_strategy": [
            "Ava is trying to help",
            "Ava is trying to mislead",
        ],
        "mia_strategy": [
            "Mia is trying to help",
            "Mia is trying to mislead",
        ],
        "decision_confidence": [
            "I am sure about my choice",
            "I am unsure about my choice",
        ],
    }

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"  item={item}, loc_a={loc_a}, loc_b={loc_b}")
    print(f"{'='*60}")
    
    differ_count = 0
    for cls_name, (comp_a, comp_b) in classes.items():
        toks_a = tok.encode(comp_a, add_special_tokens=False)
        toks_b = tok.encode(comp_b, add_special_tokens=False)
        match = "✓" if len(toks_a) == len(toks_b) else "✗ DIFFER"
        if len(toks_a) != len(toks_b):
            differ_count += 1
        print(f"  {cls_name:<30s}  A:{len(toks_a):2d} vs B:{len(toks_b):2d}  {match}")
        if len(toks_a) != len(toks_b):
            print(f"    A: {comp_a!r}")
            print(f"       tokens: {tok.convert_ids_to_tokens(toks_a)}")
            print(f"    B: {comp_b!r}")
            print(f"       tokens: {tok.convert_ids_to_tokens(toks_b)}")
    print(f"\n  Total differing pairs: {differ_count}/{len(classes)}")
