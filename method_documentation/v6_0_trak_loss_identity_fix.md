# V6.0 TRAK — trial-side loss-identity fix

Revision applied on top of the pipeline described in
`v6_0_trak_attribution.md`. The original v6.0 pilot results were computed
with the **old** trial-side loss (single-cued-letter NLL); these changes
make the trial-side gradient an interpretable ∇f comparable to the SFT-side
gradient. Re-running steps 2 + 3 (+ 4) is required to take advantage of
this fix.

---

## TL;DR

| | Old (`v6_0_trak_attribution.md`, Step 2) | New |
|---|---|---|
| Trial loss `f` | `−log P(best_cued_letter_tid | T4 prompt)` at one position | `model(input_ids, labels=labels).loss` (masked-mean CE over a cued-answer assistant span) |
| SFT loss `f` | masked-mean CE over assistant content+trailer | unchanged |
| Are both sides ∇ of the same `f`? | **No** — broke TRAK whitening interpretability | **Yes (modulo two accepted asymmetries below)** |
| Cued-answer span source | N/A — single token | canonical `"({letter}) {option_text}"` by default; `--use_emitted` opts into the leading `"({letter}) ..."` clause of the model's real T4 response |
| Trailer `<|im_end|>\n` in trial loss | N/A | **masked out** (kept in SFT loss as before) |
| `build_t4_prompt` stub | `NotImplementedError`; real logic inline in `main()` | replaced by `build_t4_messages(raw_row, cued_letter, use_emitted)` returning `(prefix_msgs, target_text, target_is_emitted)` |

---

## Why the old loss was wrong

The SFT side computed `f_sft = mean(CE over assistant tokens)` via
`model(input_ids, labels=labels).loss`. The trial side computed
`f_trial = −log p(best_token | prompt)` — NLL of one specific token at
one position. These are **different functions f**, not the same `f`
evaluated at different examples `z`.

TRAK's whitening term `(ΦᵀΦ)⁻¹` only makes the dot product
`φ(trial)ᵀ (ΦᵀΦ)⁻¹ φ(sft)` approximate the leave-one-out influence of
an SFT example on the trial outcome when both gradient sets are `∇f` of
the **same** `f`. With `f` differing, the dot product approximates
nothing with a clean interpretation. Scores were still numerically fine
and rank-orderable — that's the trap: they looked like results.

A second, subtler inconsistency: SFT used `reduction="mean"` over
variable-length response spans (dozens to hundreds of tokens), while the
trial side targeted exactly one token. Even reconciling the output
function would leave a length-dependent scale baked into `Φ` that the
Gram matrix only partially absorbs. (See "Accepted asymmetries" below
for how the new code handles this.)

---

## What changed

### 1. Trial-side `f` now matches SFT-side `f`

The trial-side loss is now masked-mean CE over a *cued-answer assistant
span*:

```python
# In analysis/trak/featurize_trials.py main():
prefix_msgs, target_text, tgt_emitted = build_t4_messages(
    raw_row, cued_letter, use_emitted=args.use_emitted)
prefix_ids, _ = build_chat_text_and_mask(prefix_msgs, tokenizer, max_seq_len=None)

target_msg = [{"role": "assistant", "content": target_text}]
target_ids, target_mask = build_chat_text_and_mask(target_msg, tokenizer, max_seq_len=None)
# (trailer-masking happens here — see §3 below)

ids = prefix_ids + target_ids
mask = [0] * len(prefix_ids) + target_mask
# left-truncate to preserve the target span if combined exceeds max_seq_len ...

input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
labels   = input_ids.clone()
labels[~torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)] = -100

out  = model(input_ids=input_ids, labels=labels)
loss = out.loss               # masked-mean CE — same f as SFT
loss.backward()
```

Both sides now invoke HuggingFace's internal CE on positions where
`labels != -100`, averaged with `reduction="mean"`. They use the **same**
`build_chat_text_and_mask` helper for tokenization, so chat-template
boundaries and assistant-span detection are byte-identical.

The dot product `φ(trial)ᵀ (ΦᵀΦ)⁻¹ φ(sft)` now approximates the
leave-one-out influence of each SFT example on the model's tendency to
produce the cued answer at T4.

### 2. `build_t4_prompt` stub → real `build_t4_messages` function

The old file had a `NotImplementedError` stub `build_t4_prompt` while the
real prompt logic lived inline in `main()`. That is removed. New helper:

```python
def build_t4_messages(raw_row, cued_letter, use_emitted=True):
    """Returns (prefix_messages, target_text, target_is_emitted).
    prefix_messages = [system, T2-user, T2-assistant, T4-user]
    target_text     = cued-answer string we attribute toward.
    """
```

### 3. Trailer `<|im_end|>\n` is masked out on the trial side

On a ~5–15 token trial target span, the generic ChatML trailer would
account for ~30–50% of the loss tokens. Because the trailer is **identical
across all 300 trials**, it carries no cue-specific signal and would
dominate the trial gradient toward a constant (between-trial-shared)
direction — drowning out the discriminative signal we need.

The new code pre-computes the trailer length once and zeros it out of
the target mask before concatenation:

```python
# Computed once before the loop:
n_trailer = len(tokenizer.encode("<|im_end|>\n", add_special_tokens=False))

# Inside the loop, after build_chat_text_and_mask(target_msg, ...):
if n_trailer <= len(target_mask):
    target_mask[-n_trailer:] = [0] * n_trailer
```

The SFT side **keeps** the trailer in its loss (unchanged behavior); on
SFT the trailer rides on a real multi-token response and contributes
meaningful learning signal. This is a deliberate small asymmetry vs the
pure-f-identity ideal; the alternative (trailer dominating a tiny trial
span) is worse for discriminating between trials.

### 4. `--use_emitted` flag: target the model's real T4 response

By default, `target_text = f"({cued_letter}) {opts_map[cued_letter]}"`
(canonical rendering — same format as SFT renders assistant text).

With `--use_emitted` set, we instead try to extract the leading
`"({cued_letter}) ..."` clause from the model's actual T4 response
(`raw_row["turn4_raw_response"]`) and target that:

```python
def extract_answer_span(emitted, cued_letter):
    text = emitted.lstrip()
    if not text.startswith(f"({cued_letter})"):
        return None
    first_line = text.split("\n", 1)[0]
    # [.!?] followed by whitespace/EOS — avoids splitting "Type 1.5 diabetes"
    m = re.match(r"^(.+?[.!?])(?:\s+|$)", first_line)
    return m.group(1) if m else first_line
```

Why extract just the leading clause rather than the whole T4 response:
if the model emitted a long hedging paragraph, targeting the whole thing
re-introduces the trial-side length asymmetry the trailer fix was trying
to limit. Targeting just the answer-bearing clause keeps the gradient
focused on the cued assertion.

Extraction failure (response doesn't start with the cued letter) ⇒
silently fall back to canonical. The number of trials that actually used
the emitted path is recorded in `trial_grads.meta.json` as
`n_emitted_target` (and surfaces in the progress line as
`emitted=N`).

### 5. New `meta.json` fields

| Field | Meaning |
|---|---|
| `n_emitted_target` | How many of the trials used the emitted-response path vs canonical |
| `use_emitted`      | Boolean — was the flag set on this run |
| `n_empty`          | How many trials were skipped because the target span was empty after left-truncation (replaces the old per-trial-OOM counter, which never triggered on this script) |

### 6. `featurize_sft.py` is essentially unchanged

The only change there is `build_chat_text_and_mask(messages, tokenizer, max_seq_len=None)` — `max_seq_len` is now optional so the trial-side caller can disable per-message truncation, then left-truncate the combined prefix+target sequence itself (preserving the target span). The SFT-side `main()` still passes `max_seq_len=1024` and gets right-truncation as before.

---

## Accepted asymmetries (don't "fix" these without checking)

The new code is **not** pure f-identity. Two asymmetries are deliberate:

1. **Trailer-in-loss**: SFT loss includes the trailer; trial loss does not. See §3.
2. **Span length**: SFT loss span is dozens–hundreds of tokens (a full assistant response); trial loss span is ~5–15 tokens (one answer clause). `reduction="mean"` means each side's gradient is scaled by `1/L` with different `L`. There is no principled "answer span" inside a general SFT example, so we accept this as an **interpretation ceiling** rather than fabricating a comparable span on the SFT side.

The honest path for validating the resulting attributions is the **removal counterfactual** (retrain SFT without the top-K attributed examples, measure the change in sycophancy). The single-checkpoint, fixed-subsample TRAK score is a *ranking* under this loss formulation, not a magnitude estimate.

---

## File-by-file diff summary

### `analysis/trak/featurize_sft.py`

- `build_chat_text_and_mask(messages, tokenizer, max_seq_len=None)` — `max_seq_len` now optional; when `None`, no truncation is applied (caller handles it).
- No other changes.

### `analysis/trak/featurize_trials.py`

- Docstring rewritten — documents the new f-identity intent plus the two accepted asymmetries.
- New module-level helper `extract_answer_span(emitted, cued_letter)`.
- `build_t4_prompt` stub deleted; replaced by `build_t4_messages(raw_row, cued_letter, use_emitted=True)` returning a 3-tuple.
- Module-level imports of `build_chat_text_and_mask`, `build_projector`, `flat_grad_into` from `featurize_sft` (was lazy-imported inside `main()`).
- Deleted dead helpers: `get_choice_token_ids`, `flat_grad`.
- `main()` loop rewritten: build prefix + target through the shared SFT helper, mask out the trailer, concatenate, left-truncate to `max_seq_len`, run `model(..., labels=labels).loss.backward()`, project via the shared projector.
- New CLI flag `--use_emitted` (default off).
- Counters: `n_empty` (target empty after truncation), `n_emitted` (emitted path taken). Both surface in the progress line and in `trial_grads.meta.json`.

---

## Re-run instructions

Steps 0a, 0b, and 1 are unchanged — you do **not** need to re-featurize SFT (`step 1`) since `featurize_sft.py`'s loss path is the same.

Steps that must be re-run with the fix:

```bash
# Step 2 — smoke first (~15 min for 120 trials, writes to trial_grads_smoke/)
sbatch bash/run_trak_featurize_trials.sh --smoke 120
sbatch bash/run_trak_featurize_trials.sh --smoke 120 --use_emitted

# Step 2 — full run (~37 min for 300 trials, see "Reference timings" below)
sbatch bash/run_trak_featurize_trials.sh                # canonical target
sbatch bash/run_trak_featurize_trials.sh --use_emitted  # emitted target
```

Inspect each `trial_grads.meta.json` afterward to confirm `n_emitted_target` is non-zero on the `--use_emitted` run (if it's zero, the column name `turn4_raw_response` may have drifted or no T4 response starts with the cued letter — investigate before scoring).

Then re-score with the existing pipeline:

```bash
# Step 3 — both unnormalized and normalized
sbatch bash/run_trak_score.sh
NORMALIZE=1 OUT_NAME=attribution_scores_norm.npy \
    sbatch bash/run_trak_score.sh

# Step 4 — analysis
python analysis/trak/analyze_attributions.py \
    --scores_file attribution_scores_norm.npy --tag norm
```

If you keep both target variants (canonical vs emitted), tag the output directories so you don't overwrite — e.g. set `OUT_NAME=attribution_scores_norm_emitted.npy` for the emitted run.

---

## Reference timings (pre-fix runs, May 26)

Saved in `/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads/trial_grads.meta.json` and `.../trial_grads_smoke/trial_grads.meta.json`:

| Run | `n_trials` | `wallclock_sec` | Rate |
|---|---|---|---|
| Smoke (`--smoke 120`) | 120 | 892.4 s ≈ 14.9 min | ~7.4 s/trial |
| Full (`--smoke 0`) | 300 | 2238.1 s ≈ 37.3 min | ~7.5 s/trial |

The new loss path adds a couple of small CPU operations per trial (the `extract_answer_span` regex, the trailer-mask slice) and removes one forward pass (the old `best_tid` no-grad pass). Net wall-clock should be close to unchanged, perhaps marginally faster — budget the same ~37 min for the next 300-trial run.

---

## Outstanding follow-ups

1. **Update `v6_0_trak_attribution.md` §Step 2.** That doc still describes the old single-token NLL trial loss as if it were the canonical pipeline. Either inline-update it to point at this fix doc, or fold the changes back in.
2. **Validation experiment**: rerun analysis with both canonical and emitted targets and compare the source-dataset lift rankings. If they materially agree, canonical is the safer default (it doesn't condition on the model having actually emitted a sycophantic answer). If they disagree on which sources push toward flipping, that itself is an interesting finding about whether the gradient signal depends on the surface form of the agreement.
3. **Removal counterfactual** (long-standing) — the only way to ground-truth-validate the attribution rankings, regardless of the loss formulation.
