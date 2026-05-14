# OLMo-2 Checkpoint Evaluation Pipeline — Single-Prompt Completion

## Context

Evaluate false-belief and epistemic vigilance abilities across the OLMo-2-1124-7B post-training trajectory (Base → SFT → DPO → RLVR intermediates → Instruct). Instead of two-agent roleplay, we fix Mia's opening to a deterministic template and have each checkpoint complete only Ava's response. This isolates comprehension ability from turn-dynamics failures that plague small models in multi-turn dialog.

## Checkpoints (10 total, excluding RM)

All under `/juice6/scr6/jshe/hf_models_all_checkpoints/OLMo-2-1124-7B/`:

| Name | Path suffix | Format | Chat template |
|------|-------------|--------|---------------|
| OLMo-2-1124-7B-Base | OLMo-2-1124-7B | safetensors | NO |
| OLMo-2-1124-7B-SFT | OLMo-2-1124-7B-SFT | pytorch bin | YES (`<\|endoftext\|><\|user\|>\n...\n<\|assistant\|>\n`) |
| OLMo-2-1124-7B-DPO | OLMo-2-1124-7B-DPO | pytorch bin | YES |
| OLMo-2-1124-7B-Instruct | OLMo-2-1124-7B-Instruct | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step60 | .../intermediate-checkpoints/step_60 | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step120 | .../intermediate-checkpoints/step_120 | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step180 | .../intermediate-checkpoints/step_180 | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step240 | .../intermediate-checkpoints/step_240 | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step300 | .../intermediate-checkpoints/step_300 | safetensors | YES |
| OLMo-2-1124-7B-RLVR-step360 | .../intermediate-checkpoints/step_360 | safetensors | YES |

Excluded: `OLMo-2-1124-7B-RM` — uses `Olmo2ForSequenceClassification` architecture (outputs scalar rewards, cannot generate text).

## Training stage ordering (developmental trajectory)

```
Base → SFT → DPO → RLVR-step60 → step120 → step180 → step240 → step300 → step360 → Instruct(final)
```

## Scenario data

Reuses existing files (no new env/combo JSONs needed):
- `sotopia_utils/sotopia_data/envs_false_belief_v3.json` — 2400 scenarios (100 scenario_types × 24 conditions)
- `sotopia_utils/sotopia_data/env_agent_combos_false_belief_v3_fixed_two_agents.json` — 2400 combos
- `sotopia_utils/sotopia_data/scenario_vocab_false_belief_v2.json` — maps scenario_type → {item, locations, prep}

### 24 conditions (3 belief × 8 trust manipulations)

- **Belief conditions**: mutual_knowledge, asymmetric_knowledge, conflicting_evidence
- **Trust manipulations**: A1_expert, A2_unreliable, A3_incentivized, A4_malicious, B1_forgetting, B2_intervening, B3_poor_vis, C_cooperate
- **Goal conditions**: adversarial (A1-A4, B1-B3) or cooperate (C only)

## Fixed Mia opening

Template: `"The {item} is {prep} {location}."`

- For adversarial (a_gold=say_false): Mia says the FALSE location (lies)
- For cooperate (a_gold=say_true): Mia says the TRUE location

Mia's stated location is determined by:
1. Look up `true_location` in env `_meta`
2. Look up both `locations` from `scenario_vocab`
3. If `a_gold == "say_false"`: pick the location that is NOT `true_location`
4. If `a_gold == "say_true"`: pick `true_location`

## Prompt format

### Chat-template models (SFT, DPO, Instruct, RLVR steps)

Wrapped via `tokenizer.apply_chat_template()`:

```
<|endoftext|><|user|>
You are Ava Thompson. Respond only with short, direct dialogue.
Scenario: Two people, Mia and Ava, are in a kitchen with a hook and a drawer. Sam placed an apron on one of these two places this morning. Mia's job is to tell Ava where the apron is. Ava will then choose where to look.
Participants: Mia Sanders and Ava Thompson
Ava Thompson's goal: You are Ava. You watched Sam place the apron on the drawer. Mia also watched Sam place the apron on the drawer. You have worked with Mia for years and she has never once misremembered where something was placed. Talk to Mia and decide where to look. Once you have decided, state your final choice and say LEAVE.
Mia Sanders's goal: Unknown
Respond only with dialogue. Do not narrate actions or describe thoughts. Once you have decided which location to check, you should state your final choice and leave the conversation by saying LEAVE. Keep your responses concise.
Turn 0: Mia Sanders said:The apron is on the hook.
Turn 1: Ava Thompson said:
<|assistant|>
```

### Base model (no chat template)

Raw text completion — identical prompt text without the chat wrapper:

```
You are Ava Thompson. Respond only with short, direct dialogue.
Scenario: Two people, Mia and Ava, are in a kitchen with a hook and a drawer. Sam placed an apron on one of these two places this morning. Mia's job is to tell Ava where the apron is. Ava will then choose where to look.
Participants: Mia Sanders and Ava Thompson
Ava Thompson's goal: You are Ava. You watched Sam place the apron on the drawer. Mia also watched Sam place the apron on the drawer. You have worked with Mia for years and she has never once misremembered where something was placed. Talk to Mia and decide where to look. Once you have decided, state your final choice and say LEAVE.
Mia Sanders's goal: Unknown
Respond only with dialogue. Do not narrate actions or describe thoughts. Once you have decided which location to check, you should state your final choice and leave the conversation by saying LEAVE. Keep your responses concise.
Turn 0: Mia Sanders said:The apron is on the hook.
Turn 1: Ava Thompson said:
```

### Why use chat template for post-trained models?

The chat template (`<|endoftext|><|user|>\n...\n<|assistant|>\n`) is the format SFT/DPO/Instruct models were trained on. Without it, the model treats the input as a document to continue rather than an instruction to respond to — it might narrate actions, add more turns, or produce any plausible document continuation instead of Ava's direct response. The base model already serves as the "no template" baseline, so the trajectory Base (no template) → SFT/DPO/Instruct (with template) measures the full package of what each training stage provides.

## Files to create/modify

### 1. `model_paths.json` — Add 10 entries

Map each checkpoint name to its local path.

### 2. `simulation/sample_olmo2_checkpoints.py` — Core simulation script

**Standalone script. Does NOT use SotopiaEnv or LM_nnsight.**

- Loads one checkpoint per invocation via `--checkpoint_name`
- Uses plain `transformers` AutoModelForCausalLM + AutoTokenizer (bfloat16)
- Sets `model.config.use_cache = True` (some OLMo configs set it False for training)
- Generation: `model.generate(max_new_tokens=500, do_sample=True, temperature=0.7)`
- Decodes only generated tokens (strips prompt), takes first line, strips echoed turn prefix
- Writes results in standard CSV format matching `sotopia_results_false_belief_v3_gemini/`
- Includes `prompt_records/` subdirectory
- Resume: skips existing non-empty CSVs
- `--dry_run`: runs 24 episodes (one per condition), prints prompt + response, no files saved

**CLI arguments:**
- `--checkpoint_name` (required)
- `--dry_run` (flag)
- `--max_episodes` (optional int)
- `--temperature` (default: 0.7)
- `--seed` (default: 0)

**Output directory:** `sotopia_results_olmo2_checkpoints_v3/dialogs/{ckpt}_None_0_{ckpt}_None_0_false_belief_v3/`

### 3. `bash/run_olmo2_checkpoint_eval.sh` — SLURM array job

- Array 0-9, one task per checkpoint
- Sphinx partition, 1 GPU 80G, 64G mem, 12h wall time
- Calls `python simulation/sample_olmo2_checkpoints.py --checkpoint_name ${CHECKPOINTS[$i]}`

### 4. `analysis/summarize_olmo2_checkpoint_results.py` — Analysis script

Parses dialog CSVs from all 10 checkpoints. Computes:
- **Behavioral accuracy** per checkpoint × condition
- **Vigilance score**: P(Ava chooses b_gold | adversarial, trust cue present) - P(Ava chooses b_gold | adversarial, no trust cue)
- **Credulity score**: P(Ava follows Mia | trust cue discredits Ava) - P(Ava follows Mia | no trust cue)
- **Null rate**: fraction where no location choice could be parsed (tracks instruction-following emergence)

Plots:
- Developmental trajectory line plot (x=training stage, y=accuracy, lines per condition group)
- Per-checkpoint condition bar chart (like `plot_behavioral_per_pair` in existing analysis)
- Summary CSV export

Location parsing: match location strings from `scenario_vocab` in Ava's response. When both mentioned, take the last one (final stated choice).

## Verification steps

1. `python simulation/sample_olmo2_checkpoints.py --checkpoint_name OLMo-2-1124-7B-Instruct --dry_run` — verify 24 prompts print with sensible responses
2. `--dry_run` for `OLMo-2-1124-7B-Base` — verify raw completion format, check output is parseable
3. Full pipeline on one checkpoint with `--max_episodes 100` to verify CSV format
4. After all 10 complete: `python analysis/summarize_olmo2_checkpoint_results.py`
