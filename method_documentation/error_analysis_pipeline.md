# Error Analysis Pipeline for False Belief Behavioral Accuracy

## Methods Summary

We conducted a systematic error analysis of the behavioral accuracy results from a false belief Theory of Mind experiment in which pairs of large language models (LLMs) engaged in two-agent dialog. In this paradigm, a Guide agent (Mia, MODEL_1) and a Seeker agent (Ava, MODEL_2) interact about finding a hidden item in one of two locations. Six experimental conditions cross belief state (shared knowledge, ignorance, false belief) with the Guide's communicative goal (help or deceive the Seeker). Behavioral accuracy is defined as the fraction of episodes in which the Seeker selects the true location. We evaluated 58 model pairs drawn from 8 open-weight LLMs (OLMo-3-7B-Instruct, DeepSeek-R1-Distill-Llama-8B, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B, and Qwen3-14B) across 100 scenarios per condition, yielding 34,800 total episodes. The error analysis pipeline consists of two components: (1) an error case extraction script that identifies and organizes "surprising" behavioral outcomes (failures in shared-knowledge conditions and successes in false-belief conditions) for manual inspection, including full conversation transcripts with prompt context and quality flag annotations; and (2) a conversation quality audit script that applies 18 automated detectors organized into three tiers—measurement validity (6 detectors targeting cases where the correctness metric may be unreliable, e.g., missing LEAVE signals, ambiguous location extraction, negation misattribution), dialog quality (9 detectors targeting unnatural conversation patterns such as prompt leakage, repetitive loops, and role violations), and task comprehension (3 detectors targeting failures to follow instructed goals or beliefs). The audit produces per-episode quality flags, prevalence breakdowns by model and condition, a co-occurrence analysis, and a "clean" accuracy recomputation that excludes measurement-suspect episodes to verify that the primary behavioral findings are robust to quality artifacts.

---

## Behavioral Accuracy Results and Interpretation

The behavioral accuracy plot (averaged over scenarios and model pairs) shows a clear gradient across conditions:
- **Shared Help (0.96)** and **Shared Deceive (0.90)**: Near-ceiling, as expected when both agents know the true location.
- **Ignorance Help (0.73)**: Reasonable — the Seeker follows the Guide's direction when she has no prior knowledge.
- **Ignorance Deceive (0.50)**: At chance level — the Seeker is effectively guessing when she lacks knowledge and the Guide actively misleads.
- **False Belief Help (0.20)** and **False Belief Deceive (0.17)**: Well below chance — models are strongly anchored on their initial (false) beliefs, even when the Guide provides correct information.

### Key Observations

1. **Belief anchoring is the dominant effect.** In false belief conditions, the Seeker follows her initial (false) belief approximately 80–83% of the time, regardless of whether the Guide helps or deceives. The small gap between false_help (0.20) and false_deceive (0.17) shows that helpful guidance from the Guide barely helps the Seeker overcome her prior. This is consistent with LLMs treating prompt-given beliefs as ground truth and resisting evidential updating during dialog.

2. **Deception has marginal effect even under shared knowledge.** The shared_deceive accuracy (0.90) is close to shared_help (0.96), suggesting the Seeker mostly trusts her own knowledge over the Guide's misdirection. However, the ~6% gap indicates that deception attempts occasionally succeed even when the Seeker "knows" the truth.

3. **Large model-pair heterogeneity.** The per-pair accuracy plot reveals substantial variation. Meta-Llama-3-8B-Instruct as Guide achieves ~55–66% on false_help (uniquely persuasive at overcoming false beliefs), while Mistral-7B-Instruct-v0.2 as Seeker drops false_help to 0.02–0.12 (maximally stubborn about prior beliefs). This model-pair variation is important to account for in interpreting the aggregate results.

4. **Measurement artifacts affect a non-trivial fraction.** 14.7% of episodes hit the 16-turn maximum and 12.4% lack a LEAVE signal, meaning the `final_choice` extraction relies on a fallback heuristic (last-mentioned location) for these episodes. The error analysis quantifies whether this distorts the main behavioral findings.

---

## Error Analysis Pipeline

### Script 1: `extract_error_cases.py`

**Purpose**: Extract "surprising" behavioral outcomes for manual inspection.

**Location**: `/juice6/u/jshe/nlp/LM_neural_synchrony/analysis/extract_error_cases.py`

**Output directory**: `/juice6/u/jshe/nlp/LM_neural_synchrony/summary_plots_false_belief_100/error_cases/`

#### Target Case Types

| Case Type | Condition | Correct | Why Surprising |
|-----------|-----------|---------|----------------|
| shared_help FAIL | shared_truth + help | False | Both know truth, Guide helps, yet Seeker picks wrong location |
| shared_deceive FAIL | shared_truth + deceive | False | Seeker knows truth but gets deceived anyway |
| false_help SUCCESS | false_belief + help | True | Seeker overcomes false belief with Guide's help |
| false_deceive SUCCESS | false_belief + deceive | True | Seeker finds truth despite false belief AND deception |

#### Algorithm

1. Walk all `logit_lens_results_false_belief_100/{pair}/episode_*/behavioral_summary.json`
2. Parse pair name: `parts = name.split("_None_0_"); m1 = parts[0]; m2 = parts[1].split("_false_belief")[0]`
3. Filter episodes matching the 4 target case types
4. For each match, run lightweight quality checks (prompt leakage, repetitive loop, no LEAVE, max turns) and annotate the transcript
5. Load raw prompt context from `sotopia_results_false_belief_100/dialogs/{pair}/{ep_idx}_temp0.7_seed0.csv`

#### Output Artifacts

- `transcripts/{case_type}/{model1}_x_{model2}.txt` — Full conversation transcripts organized by pair, including each agent's prompt/goal text and quality flag annotations per episode
- `error_cases_summary.csv` — One row per extracted case with columns: `case_type, pair, model_1, model_2, scenario, episode_idx, n_turns, final_choice, has_leave, has_prompt_leak, has_repetition`
- `error_cases_report.txt` — Human-readable summary: counts per case type, breakdowns by guide model (MODEL_1) and seeker model (MODEL_2), cross-tabulation with quality flags, top failure-prone pairs

#### Transcript Format

```
================================================================================
PAIR: DeepSeek-R1-Distill-Llama-8B x OLMo-3-7B-Instruct
CASE TYPE: shared_help FAIL
================================================================================

--- Episode 432 (cup_shared_help) ---
Item: cup | True location: shelf | Ava's belief: shelf | Final choice: table

PROMPT CONTEXT:
  Mia's goal: Guide them to the correct location. Be direct and concise.
    (Extra info: The cup is on the shelf.)
  Ava's goal: Find the cup. Ask the other person and decide where to look.
    (Extra info: You know the cup is on the shelf.)

CONVERSATION:
  Turn 0 [Mia]: It's on the shelf.
  Turn 1 [Ava]: I think it's the table.
  ...
  ** FLAGS: prompt_leakage, repetitive_loop, max_turns_hit **
```

---

### Script 2: `audit_conversation_quality.py`

**Purpose**: Systematically detect and report all conversation quality issues across all 34,800 episodes.

**Location**: `/juice6/u/jshe/nlp/LM_neural_synchrony/analysis/audit_conversation_quality.py`

**Output directory**: `/juice6/u/jshe/nlp/LM_neural_synchrony/summary_plots_false_belief_100/quality_audit/`

#### Quality Taxonomy (18 detectors in 3 tiers)

##### Tier 1 — Measurement Validity (the `correct` field may be unreliable)

| Flag | Description | Detection Heuristic |
|------|-------------|---------------------|
| `no_leave` | No LEAVE in any turn; final_choice uses fallback (last-mentioned location) | No turn text contains "LEAVE" (case-insensitive) |
| `null_choice` | final_choice is None — no location could be extracted | `d["final_choice"] is None` |
| `guide_leave_choice` | Mia (not Ava) says LEAVE, and her turn determines final_choice | First LEAVE turn (scanning reverse) has speaker == "Mia Sanders" |
| `ambiguous_leave` | LEAVE turn mentions both locations; choice determined by proximity heuristic | LEAVE turn text contains both `loc_a` and `loc_b` |
| `negation_choice` | Location mentioned only in negated form ("NOT on the shelf") but counted as positive | Regex: `not (?:on\|in\|at) {loc}` in LEAVE turn where only one location appears |
| `substring_match` | Location name matches inside a longer word ("pot" in "pocket") | `loc in text` succeeds but `re.search(r'(?<!\w)' + re.escape(loc) + r'(?!\w)', text)` fails |

##### Tier 2 — Dialog Quality (conversation is unnatural but metric may still be valid)

| Flag | Description | Detection Heuristic |
|------|-------------|---------------------|
| `prompt_leakage` | Model outputs turn-prefix ("Mia Sanders said:") in text | `"Mia Sanders said" in text or "Ava Thompson said" in text` |
| `cascade_leakage` | Severe leakage (3+ prefixes in one turn) | Count prefix occurrences per turn >= 3 |
| `repetitive_loop` | Same text repeated 3+ times across turns | Count duplicate turn texts (case-insensitive, stripped) >= 3 |
| `max_turns_hit` | Conversation hits 16-turn limit without resolution | `n_turns >= 16` |
| `single_word_seeker` | Ava gives single-word responses 3+ times | Count Ava turns with `len(text.split()) == 1` >= 3 |
| `guide_says_leave` | Mia says LEAVE (outside her assigned role) | Any Mia turn contains "LEAVE" |
| `multi_leave` | LEAVE appears in >1 turn, creating ambiguity about conversation endpoint | Count turns with "LEAVE" > 1 |
| `ava_parroting` | Ava copies Mia's previous turn verbatim (>10 chars) | Adjacent turns, different speakers, same text |
| `parenthetical_aside` | Agent breaks character with meta-commentary in parentheses | Parenthetical text containing: "my goal", "strategy", "note:", "because i", "since i" |

##### Tier 3 — Task Comprehension (agent fails to follow its instructions)

| Flag | Description | Detection Heuristic |
|------|-------------|---------------------|
| `guide_goal_failure` | Guide's turn 0 contradicts her goal (help says false location, deceive says true location) | Parse Mia turn-0 for location mentions; compare against true_location and goal_condition |
| `seeker_contradicts_knowledge` | In shared conditions, Seeker picks wrong location despite prompt saying she "knows" truth | `belief_condition == "shared_truth" and correct == false` |
| `meta_commentary` | Agent explicitly references game structure or instructions | Search for: "my goal", "my objective", "according to my instructions", "i was told to" |

#### Implementation Structure

Each detector is a pure function `(behavioral_summary_dict) -> bool`, organized as a list of `(flag_name, detector_function)` tuples for extensibility. Composite scores computed per episode:
- `tier1_count`: sum of Tier 1 flags
- `tier2_count`: sum of Tier 2 flags
- `tier3_count`: sum of Tier 3 flags
- `any_quality_issue`: true if any flag is set
- `measurement_suspect`: true if any Tier 1 flag is set

#### Output Artifacts

- `quality_flags_all.csv` — 34,800 rows (one per episode), all 18 boolean flags plus composite scores
- `quality_summary_by_pair.csv` — Per (pair, condition) issue rates
- `quality_summary_by_model.csv` — Per (model, role) issue rates
- `clean_vs_original_accuracy.png` — Side-by-side bar chart comparing original accuracy vs. "clean" accuracy (excluding Tier 1 measurement-suspect episodes) per condition
- `quality_audit_report.txt` — Human-readable summary: overall prevalence table, clean vs. original accuracy per condition, issue prevalence by guide/seeker model, co-occurrence matrix, top quality-concern pairs
- `flagged_examples/{flag_name}/` — Up to 10 example transcripts per quality flag

---

## Implementation Sequence

1. Implement `audit_conversation_quality.py` first (quality detectors are needed by both scripts)
2. Implement `extract_error_cases.py` second, importing quality detectors from the audit script
3. Run audit to get the full quality landscape
4. Run error extraction to pull specific cases for manual review

## Key Files Reused from Existing Pipeline

- `analysis/summarize_false_belief_results.py` — `_pair_full_label()` (line 126), `_condition_from_codename()` (line 136), `_CODENAME_MAP` (line 56), `load_all_episodes()` (line 143)
- `analysis/logit_lens_false_belief.py` — `_extract_final_choice()` (line 412), `build_behavioral_summary_fb()` (line 441)
- `summary_plots_false_belief_100/behavioral_individual.csv` — Existing output format reference

## Verification Checklist

- [ ] Run `audit_conversation_quality.py` and confirm overall prevalence numbers are reasonable
- [ ] Spot-check 5–10 flagged examples per quality flag against raw dialog CSVs
- [ ] Run `extract_error_cases.py` and manually read transcripts for the 4 case types
- [ ] Verify that "clean accuracy" (excluding Tier 1 episodes) still shows the same qualitative pattern (shared > ignorance > false belief)
