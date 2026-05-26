# V6.0 TRAK attribution — tracing sycophancy back to SFT training examples

## What this pipeline answers

We observed that the OLMo-3-7B **SFT** checkpoint is much more sycophantic
than the base pretrained model on the v6.0 benchmark: when a user pushes back
on the model's first-turn answer with a "cue" (authority, intensity,
emotional pressure, etc.), the SFT model flips its answer to the cued wrong
letter far more often than the base model does. SFT is the only training
stage that introduces this behavior gap — so *something the SFT corpus is
teaching the model* is responsible.

**The question:** *which* SFT training examples are responsible for
each sycophantic flip?

**The tool:** [TRAK](https://github.com/MadryLab/trak) (Park, Georgiev,
Ilyas, Leclerc, Madry; ICML 2023), a data attribution method that scores
every training example by an estimate of *how much that example's gradient
contributed to a model's behavior on a given test input*.

**The output:** a 300 × 50,000 matrix where row *i* is one held-out
sycophancy trial and column *j* is one SFT training example. Entry (i, j)
is a signed real number: large positive ⇒ "training on this example pushed
the model toward picking the cued wrong answer on this trial." Large
negative ⇒ "training on this example pushed *against* picking it."

Once we have that matrix we can ask:
* Which **training-data source datasets** (FLAN, Verifiable Reasoning,
  Persona MATH, Wildchat, …) are over-represented among the top-attributed
  examples for each level of cue effectiveness?
* What do the top-attributed examples *look like* — is there a coherent
  pattern (e.g. "always emit a confident single answer")?
* Does the attribution pattern *differ* across cue ranks (weak vs strong
  cues), suggesting different mechanisms?

---

## What TRAK is doing conceptually

### The naive question

If we wanted to know "would the model still flip on trial *i* if training
example *j* had been removed from the SFT corpus?" the textbook answer is
**leave-one-out retraining**: retrain the SFT checkpoint *without* example
*j*, then measure the change in the model's behavior on trial *i*. Doing
this for all 50,000 examples × 300 trials = 15 million retrains. At ~24
GPU-hours per OLMo-3 SFT run, that is roughly 4 × 10⁸ GPU-hours of compute.
Obviously infeasible.

### The key TRAK approximation

TRAK approximates leave-one-out retraining with a **first-order Taylor
expansion around the converged model**. Intuition: at convergence, the
loss gradient w.r.t. each training example tells you the direction in
parameter space that example would push the model. Two examples whose
gradients point in *similar* directions in parameter space have similar
influence; two with opposite gradients cancel each other. So we can
estimate "would removing example *j* change behavior on trial *i*" by
computing an *inner product* between two gradient vectors:

1. **SFT-side gradient g_j**: the gradient of the standard next-token
   training loss on example *j*, w.r.t. the SFT model's parameters.
2. **Test-side gradient g_i**: the gradient of *the behavior we care
   about* on trial *i* — for us, the gradient of
   `−log P(cued_wrong_letter | T4 prompt)`.

The TRAK influence score is, roughly, `<g_i, K⁻¹ g_j>` where `K` is the
Gram matrix of all training-example gradients (a regularization that
absorbs the redundancy across the corpus). This is the bone of the
method; the rest is engineering.

### Why this is hard in practice

Both gradients live in **parameter space** — for OLMo-3-7B that is
~7 × 10⁹ dimensions. Storing 50,000 such vectors at fp16 would take
50,000 × 7B × 2 bytes ≈ 700 TB. The Gram matrix would be 50,000² ≈
2.5 × 10⁹ entries, manageable, but inverting/Cholesky-solving across
7B-dim vectors is not.

**TRAK's engineering trick** is **random projection (JL projection)**:
pick a random matrix `R ∈ R^{D × d}` with `d ≪ D` (we use d=4096), and
work with the projected vectors `R^T g`. By the Johnson–Lindenstrauss
lemma, inner products between vectors are preserved (up to a small
relative error) when projected into `d` dimensions, as long as `d` is
chosen large enough relative to the number of vectors. After projection:
* SFT grads: a `(50000, 4096)` matrix → ~400 MB at fp16.
* Trial grads: a `(300, 4096)` matrix → tiny.
* Gram matrix: `(4096, 4096)` → ~67 MB, easy to Cholesky-solve.

The single-checkpoint version of TRAK then computes the attribution as
`P_trial @ G⁻¹ @ P_sft^T` where `G = P_sft^T P_sft / n + λI` is the
regularized Gram matrix in the projected space.

### What we sacrificed by using only one checkpoint

Full TRAK is designed to be **ensembled over multiple independent training
runs** (different random seeds, same data). The ensembling cancels out
direction noise from any single converged checkpoint. We have **one**
publicly-released OLMo-3-7B-Instruct-SFT, not an ensemble. So our scores
should be treated as **rankings**, not magnitudes — useful for "what's the
top-K?" questions, not for absolute counterfactual estimates.

---

## Pipeline overview

```
              (already done before TRAK starts)
              v6.0 sycophancy simulation → 9,500 trials per checkpoint
              Filter to trials where SFT got T2 right and T4 was parseable

  step 0a  sample_trials.py        → 300 paired trials covering 3 cue ranks
  step 0b  sample_sft_subset.py    → 50,000 SFT examples (source-stratified)

  step 1   featurize_sft.py        → 50,000 × 4,096 projected SFT gradients
  step 2   featurize_trials.py     → 300 × 4,096 projected trial gradients

  step 3   score_trak.py           → 300 × 50,000 attribution matrix

  step 4   analyze_attributions.py → source-dataset breakdowns + top examples
```

Each step is run on Stanford's NLP SLURM cluster. Steps 1 and 2 each need
a single 80 GB A100/H100 GPU; step 1 runs as a 15-way array
(~3 hours wall-clock if all shards start together); step 2 takes ~25
minutes; step 3 takes seconds.

---

## Step 0a — Sampling 300 trials (`sample_trials.py`)

### Conceptual rationale

The v6.0 simulation produced 9,500 (question × cue) trials per checkpoint.
Computing test-side gradients for all of them is wasteful — most pairs are
either trivially handled by every checkpoint, or so degenerate that they
add no information. We instead pick **300 trials** designed to span the
range of cue effectiveness, so the downstream analysis can answer "does
the attribution recipe differ between weak-cue and strong-cue flips?"

### What "cue effectiveness" means

For each of the 19 v6.0 cues we compute the **mean flip rate** averaged
across all four checkpoints (Base, DPO, RLVR, SFT). Cues are ranked from
"least effective at making any model flip" to "most effective." We then
pick three cues at rank 1 (low), rank 10 (median), and rank 19 (high)
in this ranking.

For the v6.0 pilot the chosen cues are:
* `low`  = `c11` — appeal to the user's professional credentials
* `med`  = `c18` — emotional / mild distress pressure
* `high` = `c10` — direct authoritative correction by an expert

### How questions are picked

Two constraints:

1. **Survival across checkpoints.** Pick only questions that exist in
   *all four checkpoints' result CSVs* for *all three chosen cues*. This
   guarantees apples-to-apples comparison (no missing-cell artifacts).
2. **Stratification by SFT-model T2 confidence.** Split the surviving
   questions into 4 quartiles of "how confident was the SFT model in its
   *initial* (pre-cue) answer?" Pick 25 questions per quartile, uniformly
   at random with `seed=0`. This guarantees we cover both questions where
   the SFT model was barely-correct (easy to flip) and confidently-correct
   (hard to flip).

100 questions × 3 cues = **300 trials**. Each row of `trial_manifest.csv`
records the trial ID, question ID, cue ID, cue rank label, the cued wrong
letter (we'll need this to define the test-side loss), and the observed
flip / shift statistics from the simulation.

### Implementation notes

* Pulls from `analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv`
  (per-cell aggregates from the v6.0 simulation pipeline).
* Pulls per-trial details (env_id, cued wrong letter, raw question text)
  from `sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot/`.
* Output: `analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest.csv`
  + a `trial_manifest_meta.json` recording the cue selections and
  stratification quartile edges.

---

## Step 0b — Sampling 50,000 SFT examples (`sample_sft_subset.py`)

### Conceptual rationale

The full OLMo-3 SFT corpus (Dolci-Instruct-SFT) has ~2 million examples
across 22 source datasets. Featurizing all of them would cost ~24× more
GPU-hours than we have budget for. Sampling **50,000** examples is the
largest size we can comfortably compute gradients for in a few hours of
SLURM array time, and is the size at which TRAK's JL projection (`d=4096`)
still preserves pairwise inner products with low error.

### Stratification

Naive uniform sampling over 2 M rows would severely under-represent small
source datasets (FLAN, Persona Algebra, OpenAssistant, etc.) — which we'd
then have no statistical power to detect. We instead allocate per-source
**quotas**:

* Quota proportional to corpus prevalence (`n_source / n_total`).
* **Cap any single source at 25% of the subsample** (i.e. ≤12,500 rows)
  so the biggest source (Wildchat, ~30% of corpus) doesn't crowd out
  smaller ones.
* Redistribute any deficit to under-quota sources that still have rows
  available.

Sampling within each source is uniform without replacement, `seed=0`.
After concatenating all source samples we **globally shuffle** so that
the 15-way SLURM array (next step) doesn't end up with shard 0 = "all
Wildchat" and shard 14 = "all FLAN."

### Output schema

`sft_subsample.parquet` columns:
* `sft_idx` (0..49,999) — the canonical column index used downstream
* `shard_path`, `shard_row_idx` — back-pointer into the raw corpus
* `source_dataset`, `domain`, `id` — provenance fields
* `messages` — the actual chat training example as a list of
  `{role, content}` dicts (this is what TRAK will compute the gradient of)

### Why parquet lives in scratch but the meta lives in repo

The parquet is ~74 MB, too big for a NFS-hosted repo, so it goes to
`/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/`.
The small `sft_subsample_meta.json` (per-source quotas, seed) goes under
`analysis_outputs/.../trak/` in the repo so the parameters of the sample
are version-controlled even if the parquet isn't.

---

## Step 1 — Featurize SFT examples (`featurize_sft.py` + SLURM array)

### What "featurize" means

For each of the 50,000 SFT examples we:
1. **Reconstruct the training example** as a token sequence with a
   per-token loss mask (so we only compute loss on assistant-generated
   tokens, not on user prompts or system messages).
2. **Forward pass** through the OLMo-3-7B SFT model in `bfloat16`.
3. **Compute the standard next-token cross-entropy loss** on the
   assistant tokens.
4. **Backward pass** — this populates `.grad` on every trainable parameter
   in the 7B-parameter model. The flat gradient vector is ~7 × 10⁹ entries.
5. **Project** this 7B-dim gradient down to a 4,096-dim vector with
   TRAK's fast Rademacher projector (CUDA kernel via `fast_jl`).
6. **Save** as one row of an `fp16` numpy array.

We do this for all 50,000 examples and stack into a `(50000, 4096) fp16`
matrix.

### Why a custom chat-template implementation

The OLMo-3 SFT tokenizer ships with `chat_template = None` — i.e., no
Jinja template is embedded. We hand-roll the ChatML format the model was
actually trained with:

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

We tokenize each piece (header, body, trailer) **separately** to know
exactly which token indices correspond to assistant content vs everything
else, then build a `loss_mask` that is 1 for assistant body+trailer tokens
and 0 elsewhere. This avoids relying on whitespace-based heuristics for
where assistant tokens start/end, which break for some tokenizer variants.

```python
labels = input_ids.clone()
labels[~assistant_mask] = -100   # HF ignores positions with -100 in CE loss
loss = model(input_ids=input_ids, labels=labels).loss
loss.backward()
```

### Why JL projection (`fast_jl`)

Without projection: a single 7B-dim gradient is ~14 GB at fp16. Storing
50,000 of them is 700 TB. We must project before storage.

JL = Johnson–Lindenstrauss: a random matrix `R ∈ {±1/√d}^{D × d}` (the
"Rademacher" variant — entries are random signs) has the property that
inner products between two vectors are approximately preserved after
projection: `<R^T g_1, R^T g_2> ≈ <g_1, g_2>` with high probability if
`d` is large enough. We use `d = 4096`, which is the empirical sweet spot
TRAK's authors recommend for 7B-scale models.

`fast_jl` is a custom CUDA kernel for this projection. We need it because
forming and materializing the `R` matrix explicitly would itself be
`7B × 4096 × 2 bytes` ≈ 57 TB. `fast_jl` instead generates `R`'s columns
on-the-fly via a deterministic pseudorandom kernel keyed on `(seed,
column_index)`, and streams the projection directly. This makes
projection a single-second per-example operation on an 80 GB GPU.

### Memory engineering

The naive recipe `g = torch.cat([p.grad.flatten() for _, p in params])`
allocates a *second* copy of the 14 GB gradient (`torch.cat` allocates
a new contiguous tensor). On a 80 GB H100 with the 7B model resident
in bf16 (~14 GB) plus optimizer state plus activations, that second copy
caused OOM. The fix:

```python
flat_buf = torch.zeros(n_params, device='cuda', dtype=torch.float16)
# ... in the loop:
offset = 0
for _, p in params:
    n = p.numel()
    if p.grad is None:
        flat_buf[offset:offset+n].zero_()
    else:
        flat_buf[offset:offset+n].copy_(p.grad.detach().reshape(-1))
        p.grad = None   # release per-param grad memory immediately
    offset += n
```

Pre-allocate the flat buffer once, copy each parameter's grad into its
slice, and **null out `p.grad` as we go** so the per-param grad memory is
released before we allocate the next one. Peak memory drops from ~28 GB
to ~14 GB of gradient overhead.

### SLURM sharding

15 array tasks, ~3,334 examples each. Each task loads the model, runs the
projector, processes its slice, and writes:

```
sft_grads_shard{NN}_of15.npy        # (n_shard, 4096) fp16
sft_grads_shard{NN}_of15.idx.npy    # (n_shard,) int64 — sft_idx for each row
sft_grads_shard{NN}_of15.meta.json  # wall-clock, OOM count, empty-mask count
```

Two failure modes the meta tracks:
* **OOM**: an unusually long sequence overflows GPU memory mid-backward.
  We `torch.cuda.empty_cache()` and skip the row (it stays as zeros in
  the projection matrix).
* **Empty mask**: an example's `messages` list contains no assistant
  content within the first `max_seq_len=1024` tokens. Skipped (also zero
  row).

For the v6.0 pilot: 0 OOMs, ~1009/50000 = 2% empty masks. Zero-row
contamination is downstream-managed by L2-normalization (see step 3).

### Worth knowing about `fast_jl`

Getting `fast_jl` installed on the cluster was the single hardest piece
of this pipeline. It required:
* CUDA 13.0 nvcc + headers via `conda install -c nvidia cuda-toolkit=13.0.2`
  (the cluster's system CUDA was 12.8; PyTorch was built against cu130).
* Patching `fast_jl-0.1.3/setup.py` to override `TORCH_CUDA_ARCH_LIST` to
  `"8.0;9.0"` (CUDA 13 dropped Volta `sm_70` support but `fast_jl` had it
  hard-coded).
* Building from a NFS-shared source dir (`/juice6/u/jshe/.../build/`)
  rather than `/tmp`, because login and worker nodes have separate `/tmp`.
* Loading `import torch` *before* `import fast_jl` so libtorch is on the
  process's lib search path when `fast_jl`'s `.so` is opened.
* Passing `max_batch_size=8` to `CudaProjector` — fast_jl only ships
  CUDA kernels for batch sizes in `{8, 16, 32}`.

These bits live in `analysis/trak/featurize_sft.py:build_projector()`.

---

## Step 2 — Featurize trials (`featurize_trials.py`)

### What is the "test-side loss" for a sycophancy trial?

For each trial we have a prompt sequence ending just before the model's
T4 assistant turn:

```
<|im_start|>system\n{SYSTEM}<|im_end|>
<|im_start|>user\n{T2_user_message: question + 4 options}<|im_end|>
<|im_start|>assistant\n{T2_response: e.g. "(B) IgE"}<|im_end|>
<|im_start|>user\n{T4_user_message: cue + same question + options}<|im_end|>
<|im_start|>assistant\n(   ← model's next token is the letter A/B/C/D
```

We feed this prompt through the SFT model and ask: **how does the
negative log-probability of the cued wrong letter at this position change
with the parameters?**

Concretely:

```python
out = model(input_ids=prompt_ids)
log_probs = log_softmax(out.logits[0, -1, :].float(), dim=-1)
loss = -log_probs[cued_wrong_letter_token_id]
loss.backward()
```

This loss is the gradient of *being sycophantic on this trial*. A
training example with a *positive* attribution to this trial is one whose
training gradient points in the same direction as this trial's
"go-along-with-the-cue" gradient — i.e., training on it pushes the model
toward sycophancy.

### Picking the right letter token

The model's tokenizer has multiple token IDs that could encode "the
letter B" depending on surrounding context: `B`, ` B`, `(B`, ` (B`,
`B)`. The simulation pipeline picked whichever variant had the highest
logit at the relevant position (this matched how `turn4_prob_X` was
extracted). We replicate that by running a no-grad forward pass first,
choosing the candidate letter token with the max logit at the final
position, *then* doing the backward pass on that token's NLL.

### Reconstructing the prompt exactly

The prompt text comes from the raw simulation CSV:

* `SYSTEM` = the v6.0 multi-sys-report system message
  (from `sample_olmo3_checkpoints_v5_0_sycophancy.SYSTEM_CONTEXT`).
* `T2_user_message` = `f"{question_text}\n\n(A) ... (D) {option_d}{ANSWER_PROMPT}"`
* `T2_response` = the *actual* response the SFT model produced at simulation
  time, pulled from `turn2_raw_response` in the raw CSV. This matters:
  the model's gradient on T4 depends on what the assistant said at T2, so
  using a synthetic T2 response would change the attribution.
* `T4_user_message` = `f"{cue}\n\n{question + options}{ANSWER_PROMPT}"`.

If reconstructed length exceeds `max_seq_len=2048`, we left-truncate (keep
the last 2048 tokens) — this preserves the T4 user message and the
assistant-open-turn marker, which is what determines the next-token
distribution.

### Same projector as SFT

Critical: the test-side gradients must be projected with the **same `R`
matrix** as the SFT-side gradients, otherwise their projected inner
products are meaningless. We import the `build_projector()` helper from
`featurize_sft.py` and pass the same `proj_seed=0` and `proj_dim=4096`.

### Smoke test

Before launching the full 300-trial run, a `--smoke 120` mode processes
only the first 120 trials (~10 min wall-clock) and writes to a separate
`trial_grads_smoke/` directory so the real run isn't polluted. We use
this to validate: (a) the prompt reconstruction works on real data,
(b) the candidate-token picker doesn't crash on edge-case letters,
(c) the projection produces non-zero rows.

---

## Step 3 — Score (`score_trak.py`)

### What this step computes

Given:
* `P_sft ∈ R^{50000 × 4096}` — projected SFT-side gradients
* `P_trial ∈ R^{300 × 4096}` — projected trial-side gradients

we compute the (normalized) influence:

```
G = P_sft^T P_sft / N_sft + λI               # (4096 × 4096) Gram
S = P_trial @ G⁻¹ @ P_sft^T                  # (300 × 50000) scores
```

`G` is the Tikhonov-regularized Gram matrix in projected space.
Conceptually it inverts out redundancy: if 1,000 SFT examples have nearly
identical gradients, the inversion divides their joint influence by ~1000
so they don't dominate the attribution.

The Tikhonov term `λI` (we use `λ = 10⁻²`) makes `G` invertible even when
the projected gradients are low-rank or near-singular. We didn't tune λ —
the default range `[10⁻³, 10⁻¹]` is what the TRAK authors recommend.

### Why Cholesky instead of explicit inverse

We never form `G⁻¹` explicitly. Instead:

```python
L = torch.linalg.cholesky(G)                 # G = L L^T, lower triangular
x = torch.cholesky_solve(P_trial.T, L)       # solves G x = P_trial.T
scores = (x.T @ P_sft.T).cpu().numpy()       # (300, 50000)
```

Cholesky-then-solve is numerically more stable than computing `G⁻¹` and
multiplying, and is roughly 3× faster.

The whole computation — Gram + Cholesky + scoring — runs in **<1 second**
on a single H100 because everything is in the 4096-dim projected space.
The bottleneck is loading the 50000×4096 fp16 grad matrix into GPU
memory (~400 MB).

### Output

* `scores/attribution_scores.npy` — `(300, 50000) fp32`, ~600 MB
* `scores/sft_idx_order.npy` — `(50000,)`, the `sft_idx` for each column
* `scores/trial_id_order.csv` — `(300,)`, the `trial_id` for each row
* `scores/attribution_meta.json` — pointers + parameters used

---

## Step 3.5 — The gradient-norm confound and the `--normalize` fix

### The red flag

First pass through `analyze_attributions.py` on the un-normalized scores
gave a bizarre result: the **same source datasets appeared with high
positive lift in both the top-100 and the bottom-100 of every cue rank**.
FLAN (4-7× lift), Aya (3-5×), and Logic Puzzles (2-4×) showed up as both
the strongest "pushes toward flipping" *and* the strongest "pushes against
flipping." That cannot be a real causal signal.

### What went wrong

Inner-product attribution scores conflate two things:

1. **Direction**: does this training example's gradient point in the same
   way as the test trial's gradient?
2. **Magnitude**: how big is this training example's gradient overall?

Some SFT examples produce gradients with much larger norms than others —
for example, examples where the model's per-token loss is high (the model
"didn't know" this content) have larger gradients than examples where the
model is already confident. When we do the raw inner product, high-norm
examples score high *in absolute value* regardless of direction. They
end up in **both** top and bottom of every ranking because their large
norm dominates the sign.

This is a known TRAK pitfall, especially severe in single-checkpoint
runs (ensembling cancels some of it).

### The fix

Add a `--normalize` flag to `score_trak.py` that **L2-normalizes both
the SFT-side and trial-side projected gradients before computing the
Gram and the scores**:

```python
P_sft  = P_sft  / P_sft.norm(dim=1, keepdim=True).clamp_min(1e-12)
P_trial = P_trial / P_trial.norm(dim=1, keepdim=True).clamp_min(1e-12)
```

This turns the attribution score from an inner product into a (Gram-
inverse-modulated) **cosine similarity in projection space** — purely a
direction comparison, with magnitudes equalized to 1.

The clamp at `1e-12` handles the ~2% zero-row contamination from
empty-mask SFT examples (their normalized rows stay as zero vectors and
thus contribute zero to the score, which is the right behavior).

### Result of normalization

The bottom-vs-top contradiction vanished. Top-attributed examples
became sharply distinguishable from bottom-attributed examples. The
source-dataset distribution shifted: math/reasoning/code SFT data
dominated top-attribution for low/medium-cue trials (driven by long
step-by-step problem-solving format), while chat-style "agree-with-
premise + commit-confidently" examples (Wildchat fanfic continuations,
Mickey Mouse trivia with "Great question!" suck-up openings,
WildJailbreak premise-engagement) dominated top-attribution for
high-cue trials. The bottom of all ranks was algorithmic computation
("compute the real answer, don't just agree"). The story finally
made sense.

We keep both score variants on disk:
* `scores/attribution_scores.npy` (unnormalized)
* `scores/attribution_scores_norm.npy` (normalized, **default for analysis**)

---

## Step 4 — Downstream analysis (`analyze_attributions.py`)

For each cue rank (`low`, `med`, `high`) we average the attribution
across the 100 trials of that rank, giving a single 50,000-vector of
mean attribution per SFT example. Then:

### Source-dataset lift

The **baseline** distribution is the per-source share of the 50,000-row
SFT subsample. For each cue rank we compute the per-source share among
the **top-K** and **bottom-K** (default K=100) attribution-ranked
examples. The **lift** for source *s* in the top-K is
`P(source=s | top-K) / P(source=s | baseline)`. Lift > 1 means the
source is over-represented; lift < 1 means under-represented.

We sort by lift and print the top 5 sources for both top-K and bottom-K
per cue rank. This is the headline summary table.

### χ² test across cue ranks

Build a 3 × n_sources contingency table where row *r* and column *s* is
"how many of the top-100 attribution examples for cue rank *r* come from
source *s*." Run a χ² independence test. If significant, the *recipe* of
which sources push toward flipping differs between weak-cue and strong-cue
trials.

For the v6.0 pilot the post-normalization p-value was **0.069** —
borderline non-significant. Interpretation: the *source-dataset mix* of
top-attributed examples is roughly the same across cue ranks; what
differs is the *attribution magnitude* (std grows monotonically:
0.63 → 0.85 → 1.14 from low to high cues) and the *content of the
examples within each source*.

### Top-N example dumps

For each cue rank we dump the top 10 and bottom 10 attributed examples
to a markdown file, including a 180-character snippet of the first user
message and first assistant response. This is what we eyeball to verify
the attribution is picking up coherent patterns (and not e.g. some
tokenization artifact).

Outputs:
* `analysis_outputs/.../trak/analysis/attribution_summary_norm.json`
* `analysis_outputs/.../trak/analysis/top_attributed_examples_norm.md`
* `analysis_outputs/.../trak/analysis/eyeball_notes_norm.txt` (manual
  notes on what the example content reveals)

---

## Output layout summary

Small artifacts (manifests, metas, score metadata, analysis outputs) live
in the version-controlled repo:

```
analysis_outputs/v6_0_olmo3_7b_pilot/trak/
├── trial_manifest.csv                  # 300 trials, paired
├── trial_manifest_meta.json
├── sft_subsample_meta.json             # parquet itself is in scratch
├── scores/
│   ├── attribution_meta.json
│   ├── attribution_scores_norm.meta.json
│   └── trial_id_order.csv
└── analysis/
    ├── attribution_summary.json        # unnormalized (kept for comparison)
    ├── attribution_summary_norm.json   # normalized (canonical)
    ├── top_attributed_examples.md      # unnormalized
    ├── top_attributed_examples_norm.md # normalized
    └── eyeball_notes_norm.txt          # manual interpretation
```

Sizable artifacts (parquet, projected gradients, attribution matrix) live
in scratch:

```
/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/
├── sft_subsample.parquet               # ~74 MB
├── sft_grads/
│   ├── sft_grads_shard{NN}_of15.npy    # (~n_shard × 4096) fp16
│   ├── sft_grads_shard{NN}_of15.idx.npy
│   └── sft_grads_shard{NN}_of15.meta.json
├── trial_grads/
│   ├── trial_grads.npy                 # (300, 4096) fp16
│   ├── trial_grads.idx.csv
│   └── trial_grads.meta.json
└── scores/
    ├── attribution_scores.npy          # (300, 50000) fp32 ~60 MB
    ├── attribution_scores_norm.npy     # (300, 50000) fp32 ~60 MB
    └── sft_idx_order.npy
```

---

## Caveats and known limitations

1. **Single-checkpoint TRAK.** The single converged SFT checkpoint we
   attribute against is noisy compared to the full TRAK ensemble. Use
   scores as **rankings**, not magnitudes; do permutation tests over
   top-K source distributions for any significance claim.
2. **Subsampled SFT corpus.** We attribute against 50,000 of ~2 M training
   examples. An influential example outside the subsample is invisible
   to this analysis. The stratified sample (capped per source) preserves
   tail-source coverage, but a target-source-specific oversampling pass
   would be needed if a particular source looked interesting.
3. **Sequence-length cap.** SFT examples are truncated at 1,024 tokens
   (right-truncated); trial prompts at 2,048 (left-truncated to preserve
   the answer-open marker). Long math derivations may have their tails
   clipped — this is logged as a per-shard count but can't easily be
   undone without re-featurizing at higher length.
4. **fp16 projection.** Quantization noise from the fp16 projector is
   small relative to the JL projection noise at `d=4096`. If results
   look noisier than expected, bumping `proj_dim` to 8192 would help
   (~2× memory and time).
5. **No counterfactual retrain validation.** We have not removed any of
   the top-attributed examples and re-trained the SFT model to confirm
   that the predicted attribution direction holds. That would be the
   gold-standard test; with one checkpoint and 24 GPU-hours per retrain,
   it's a separate study.
6. **Gradient-norm confound.** Use `attribution_scores_norm.npy`
   (cosine-normalized) for any source-distribution analysis; the raw
   `attribution_scores.npy` is kept only for completeness.

---

## Quick-start re-run recipe

If you need to re-run the whole pipeline from scratch:

```bash
# (Step 0) sampling — both are fast, single-CPU, no SLURM needed
python analysis/trak/sample_trials.py
python analysis/trak/sample_sft_subset.py

# (Step 1) SFT featurize — 15-way SLURM array, ~3 h end-to-end
sbatch bash/run_trak_featurize_sft.sh

# (Step 2) trial featurize — single GPU, ~25 min
sbatch bash/run_trak_featurize_trials.sh

# (Step 3) score — single GPU, <1 minute
sbatch bash/run_trak_score.sh                              # unnormalized
NORMALIZE=1 OUT_NAME=attribution_scores_norm.npy \
    sbatch bash/run_trak_score.sh                          # normalized

# (Step 4) downstream analysis — local CPU
python analysis/trak/analyze_attributions.py \
    --scores_file attribution_scores_norm.npy --tag norm
```

To re-run only the scoring + analysis with a different λ:

```bash
LAM=1e-1 NORMALIZE=1 OUT_NAME=attribution_scores_norm_lam1e-1.npy \
    sbatch bash/run_trak_score.sh
python analysis/trak/analyze_attributions.py \
    --scores_file attribution_scores_norm_lam1e-1.npy --tag norm_lam1e-1
```

To re-run with a different projector seed (sanity-check projection noise):

```bash
# Move scratch out of the way first, then:
PROJ_SEED=1 sbatch bash/run_trak_featurize_sft.sh
PROJ_SEED=1 sbatch bash/run_trak_featurize_trials.sh
sbatch bash/run_trak_score.sh
```

The seed must match between SFT and trial featurize jobs — otherwise the
projected gradients live in different subspaces and the attribution
scores are noise.
