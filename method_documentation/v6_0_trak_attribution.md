# V6.0 TRAK attribution — tracing sycophancy back to SFT training examples

## What this pipeline answers

We observed that the OLMo-3-7B **SFT** checkpoint is much more sycophantic
than the base pretrained model on the v6.0 benchmark: when a user pushes back
on the model's first-turn answer with a "cue" (authority, intensity,
emotional pressure, etc.), the SFT model flips its answer to the cued wrong
letter far more often than the base model does. SFT is the only training
stage that introduces this behavior gap — so *something the SFT corpus is
teaching the model* is responsible.

**The question:** *which* SFT training examples are responsible for each
sycophantic flip?

**The tool:** [TRAK](https://github.com/MadryLab/trak) (Park, Georgiev,
Ilyas, Leclerc, Madry; ICML 2023), a data attribution method that scores
every training example by an estimate of *how much that example's gradient
contributed to a model's behavior on a given test input*.

**The output:** a 4,432 × 50,000 matrix where row *i* is one held-out
sycophancy trial and column *j* is one SFT training example. Entry (i, j)
is a signed real number: large positive ⇒ "training on this example
pushed the model toward picking the cued wrong answer on this trial."
Large negative ⇒ "training on this example pushed *against* picking it."

Once we have that matrix we can ask:
* Which **training-data source datasets** (FLAN, Verifiable Reasoning,
  Persona MATH, Wildchat, …) are over- or under-represented among the
  top-attributed examples for each cue?
* What do the top-attributed examples *look like* — is there a coherent
  pattern (e.g. "always emit a confident single answer")?
* Does the attribution pattern *differ* across cues (the 20-cue
  ranking spans from c11 at 3% flip rate to c10 at 92%), suggesting
  different mechanisms?

---

## What TRAK is doing conceptually

### The naive question

If we wanted to know "would the model still flip on trial *i* if training
example *j* had been removed from the SFT corpus?" the textbook answer is
**leave-one-out retraining**: retrain the SFT checkpoint *without* example
*j*, then measure the change in the model's behavior on trial *i*. Doing
this for all 50,000 examples × 4,432 trials = 221 million retrains. At
~24 GPU-hours per OLMo-3 SFT run, that is roughly 5 × 10⁹ GPU-hours of
compute. Obviously infeasible.

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
   about* on trial *i* — for us, the gradient of the masked-mean
   cross-entropy that produces the cued wrong answer at T4 (see
   §"Step 2" for the exact loss definition; this is the loss-identity
   fix applied in 2026-06).

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
* Trial grads: a `(4432, 4096)` matrix → ~36 MB at fp16.
* Gram matrix: `(4096, 4096)` → ~67 MB, easy to Cholesky-solve.

The single-checkpoint version of TRAK then computes the attribution as
`P_trial @ G⁻¹ @ P_sft^T` where `G = P_sft^T P_sft / n + λI` is the
regularized Gram matrix in the projected space.

### What we sacrificed by using only one checkpoint

Full TRAK is designed to be **ensembled over multiple independent training
runs** (different random seeds, same data). The ensembling cancels out
direction noise from any single converged checkpoint. We have **one**
publicly-released OLMo-3-7B-Instruct-SFT, not an ensemble. So our scores
should be treated as **rankings**, not magnitudes — useful for "what's
the top-K?" questions, not for absolute counterfactual estimates.

---

## Pipeline overview

```
              (already done before TRAK starts)
              v6.0 sycophancy simulation → 9,500 trials per checkpoint
              Filter to trials where SFT got T2 right and T4 was parseable
              (yields 4,432 surviving (q, cue) pairs across 20 cues)

  step 0a  sample_trials.py --all_trials   → trial_manifest_all.csv (4,432)
  step 0b  sample_sft_subset.py            → 50,000 SFT examples (stratified)

  step 1   featurize_sft.py                → 50,000 × 4,096 projected SFT grads
           (15-way SLURM array)
  step 2   featurize_trials.py             → 4,432 × 4,096 projected trial grads
           (15-way SLURM array)

  step 3   score_trak.py                   → 4,432 × 50,000 attribution matrix

  step 4   plot_attribution_distribution   → score-vs-rank diagnostic
           plot_category_attribution        → per-cue category attribution shares
           plot_top_source                  → per-cue stacked bars + lift heatmap (source_dataset)
```

Each step is run on Stanford's NLP SLURM cluster. Steps 1 and 2 are both
15-way arrays; SFT featurize is the longer one (~3 h wall per shard).
Trial featurize is ~37 min per shard. Step 3 takes ~2 min on a single
80 GB GPU. Step 4 plots run on CPU in seconds-to-minutes.

---

## Step 0a — Sampling trials (`sample_trials.py --all_trials`)

### Conceptual rationale

The v6.0 simulation produced 9,500 (question × cue) trials per
checkpoint. Filtering to "SFT got T2 right AND T4 was parseable AND the
(q, cue) pair survives in all 4 checkpoint result CSVs" leaves **4,432
surviving (q, cue) pairs across 20 cues**. We featurize the trial-side
gradient for every one of them — no per-cue or per-question
sub-sampling, no quartile stratification.

This is a deliberate departure from the original 300-trial pilot, which
picked 3 cues at rank 0 / mid / last in cue effectiveness, then 100
questions per cue stratified into 4 confidence quartiles. The expanded
4,432-trial design gives ~10–15× more statistical power per cue and
exposes per-cue structure that bucketed analysis was averaging out.

### Cue ranking

For each of the 20 v6.0 cues we compute the **mean flip rate** averaged
across all four checkpoints (Base, DPO, RLVR, SFT). Cues are ranked from
"least effective at making any model flip" to "most effective." This
ranking is recorded in `cue_ranking_full` inside
`trial_manifest_all_meta.json` and is used downstream by the plot
scripts to order the x-axis (so plots match
`per_cue_effectiveness_ranking.png`):

```
rank 0  c11  flip=0.029  (least effective)
rank 1  c20  flip=0.068
...
rank 18 c06  flip=0.778
rank 19 c10  flip=0.923  (most effective)
```

The manifest also includes a `cue_rank` column (low / med / high)
bucketing the 20 cues into thirds — 6 cues for `low` (ranks 0-5), 7 for
`med` (6-12), 7 for `high` (13-19). This is informational; per-cue
analysis is the default downstream.

### Output schema

`trial_manifest_all.csv` columns:
* `trial_id` (t00000 … t04431) — stable, sorted by `(question_id, cue_id)`
* `question_id`, `cue_id`, `cue_rank` (low/med/high bucket)
* `env_id`, `question_domain`, `cue_strength`, `cue_strength_label`
* `correct_letter`, `cued_wrong_letter` — recovered from raw SFT CSV
* `t2_conf_sft`, `t2_conf_quartile` — per-question SFT T2 confidence
* `sft_flip`, `sft_shift`, `sft_shift_norm`
* `base_flip`, `base_shift`, `base_shift_norm`
* `gain_flip = sft_flip - base_flip`, `gain_shift_norm`

`trial_manifest_all_meta.json` records the full per-cue ranking, the
bucketing thirds, quartile edges, source cell_summary path, and
`n_missing_raw`.

### Implementation notes

* Pulls from `analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv`
  (per-cell aggregates from the v6.0 simulation pipeline; already
  enforces the "T2-correct + T4-parseable + surviving all 4 ckpts" filter).
* Pulls per-trial fields (`env_id`, `correct_position`, `wrong_position`)
  from `sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot/
  results_OLMo-3-7B-Instruct-SFT_seed0.csv`.
* The legacy 300-trial mode (`sample_trials.py` with no flag) still
  exists and writes `trial_manifest.csv`; it is kept for back-comparison
  but is not part of the canonical pipeline.

---

## Step 0b — Sampling 50,000 SFT examples (`sample_sft_subset.py`)

### Conceptual rationale

The full OLMo-3 SFT corpus (Dolci-Instruct-SFT) has ~2 million examples
across 22 source datasets. Featurizing all of them would cost ~24× more
GPU-hours than we have budget for. Sampling **50,000** examples is the
largest size we can comfortably compute gradients for in a few hours of
SLURM array time, and is the size at which TRAK's JL projection
(`d=4096`) still preserves pairwise inner products with low error.

### Stratification

Naive uniform sampling over 2 M rows would severely under-represent
small source datasets (FLAN, Persona Algebra, OpenAssistant, etc.) —
which we'd then have no statistical power to detect. We instead allocate
per-source **quotas**:

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
The small `sft_subsample_meta.json` (per-source quotas, seed) goes
under `analysis_outputs/.../trak/` in the repo so the parameters of the
sample are version-controlled even if the parquet isn't.

---

## Step 1 — Featurize SFT examples (`featurize_sft.py` + SLURM array)

### What "featurize" means

For each of the 50,000 SFT examples we:
1. **Reconstruct the training example** as a token sequence with a
   per-token loss mask (so we only compute loss on assistant-content
   tokens, not on user prompts or system messages).
2. **Forward pass** through the OLMo-3-7B SFT model in `bfloat16`.
3. **Compute the standard masked-mean next-token cross-entropy loss**
   on the assistant content + trailer tokens.
4. **Backward pass** — this populates `.grad` on every trainable parameter
   in the 7B-parameter model. The flat gradient vector is ~7 × 10⁹ entries.
5. **Project** this 7B-dim gradient down to a 4,096-dim vector with
   TRAK's fast Rademacher projector (CUDA kernel via `fast_jl`).
6. **Save** as one row of an `fp16` numpy array.

We do this for all 50,000 examples and stack into a `(50000, 4096) fp16`
matrix.

### Why a custom chat-template implementation

The OLMo-3 SFT tokenizer ships with `chat_template = None` — i.e., no
Jinja template is embedded. We hand-roll the ChatML format the model
was actually trained with:

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

We tokenize each piece (header, body, trailer) **separately** to know
exactly which token indices correspond to assistant content vs everything
else, then build a `loss_mask` that is 1 for assistant body+trailer
tokens and 0 elsewhere. This avoids relying on whitespace-based
heuristics for where assistant tokens start/end, which break for some
tokenizer variants.

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
`d` is large enough. We use `d = 4096`, which is the empirical sweet
spot TRAK's authors recommend for 7B-scale models.

`fast_jl` is a custom CUDA kernel for this projection. We need it
because forming and materializing the `R` matrix explicitly would
itself be `7B × 4096 × 2 bytes` ≈ 57 TB. `fast_jl` instead generates
`R`'s columns on-the-fly via a deterministic pseudorandom kernel keyed
on `(seed, column_index)`, and streams the projection directly. This
makes projection a single-second per-example operation on an 80 GB GPU.

### Memory engineering

The naive recipe `g = torch.cat([p.grad.flatten() for _, p in params])`
allocates a *second* copy of the 14 GB gradient (`torch.cat` allocates
a new contiguous tensor). On a 80 GB H100 with the 7B model resident
in bf16 (~14 GB) plus optimizer state plus activations, that second
copy caused OOM. The fix:

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
slice, and **null out `p.grad` as we go** so the per-param grad memory
is released before we allocate the next one. Peak memory drops from
~28 GB to ~14 GB of gradient overhead.

### SLURM sharding

15 array tasks, ~3,334 examples each. Each task loads the model, runs
the projector, processes its slice, and writes:

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
  content within the first `max_seq_len=1024` tokens. Skipped (also
  zero row).

For the v6.0 run: 0 OOMs, ~1009/50000 = 2% empty masks. Zero-row
contamination is downstream-managed by L2-normalization (see step 3.5).

### Worth knowing about `fast_jl`

Getting `fast_jl` installed on the cluster was the single hardest piece
of this pipeline. It required:
* CUDA 13.0 nvcc + headers via `conda install -c nvidia cuda-toolkit=13.0.2`
  (the cluster's system CUDA was 12.8; PyTorch was built against cu130).
* Patching `fast_jl-0.1.3/setup.py` to override `TORCH_CUDA_ARCH_LIST`
  to `"8.0;9.0"` (CUDA 13 dropped Volta `sm_70` support but `fast_jl`
  had it hard-coded).
* Building from a NFS-shared source dir (`/juice6/u/jshe/.../build/`)
  rather than `/tmp`, because login and worker nodes have separate `/tmp`.
* Loading `import torch` *before* `import fast_jl` so libtorch is on the
  process's lib search path when `fast_jl`'s `.so` is opened.
* Passing `max_batch_size=8` to `CudaProjector` — fast_jl only ships
  CUDA kernels for batch sizes in `{8, 16, 32}`.

These bits live in `analysis/trak/featurize_sft.py:build_projector()`.

---

## Step 2 — Featurize trials (`featurize_trials.py` + SLURM array)

### What is the "test-side loss" for a sycophancy trial?

The loss formulation went through a significant revision in 2026-06 (the
**loss-identity fix**, documented in
`method_documentation/v6_0_trak_loss_identity_fix.md`). The current
canonical definition:

For each trial we construct the full T4 conversation as a 5-message
chat:

```
<|im_start|>system\n{SYSTEM}<|im_end|>
<|im_start|>user\n{T2_user_message: question + 4 options}<|im_end|>
<|im_start|>assistant\n{T2_response: e.g. "(B) IgE"}<|im_end|>
<|im_start|>user\n{T4_user_message: cue + same question + options}<|im_end|>
<|im_start|>assistant\n{TARGET: the cued answer}<|im_end|>
```

where `TARGET` is either:
* (default) canonical `"({cued_letter}) {option_text}"`, rendered the
  same way SFT examples render assistant text, or
* (with `--use_emitted`) the leading `"({cued_letter}) ..."` clause
  extracted from the model's *actual* T4 response
  (`raw_row["turn4_raw_response"]`) — useful when you want the
  attribution to track what the model in fact emitted, not the
  counterfactual.

We tokenize via the same `build_chat_text_and_mask()` helper used on the
SFT side, mask everything except the cued-answer **content tokens** to
-100 (the trailer `<|im_end|>\n` is excluded from the loss span on the
trial side; see "Deliberate asymmetries" below), and compute:

```python
out = model(input_ids=input_ids, labels=labels)
loss = out.loss  # masked-mean CE over cued-answer content tokens
loss.backward()
```

This makes `f = mean(CE over assistant-text span)` — the **same scalar
function** the SFT side optimizes — evaluated on the trial input. That
identity is what makes TRAK's whitening `(ΦᵀΦ)⁻¹` produce interpretable
influence scores; without it, the dot product `φ(trial)ᵀ (ΦᵀΦ)⁻¹
φ(sft)` approximates nothing with a clean interpretation.

A training example with a *positive* attribution to this trial is one
whose SFT gradient points in the same direction as this trial's
"emit-the-cued-answer" gradient — i.e., training on it pushes the model
toward sycophancy.

### Deliberate asymmetries vs SFT

The current trial-side loss is **not** strict f-identity with the SFT
side. Two asymmetries are accepted on purpose:

1. **Trailer-in-loss**: SFT keeps the `<|im_end|>\n` trailer in its loss
   span; trial masks it out. Reason: on a ~5–15 token trial target the
   trailer would dominate the gradient toward a constant direction (the
   trailer is identical across all 4,432 trials and carries no
   cue-specific signal). On SFT the trailer rides on a real multi-token
   response and contributes meaningful learning signal.
2. **Length asymmetry**: SFT loss span is a full assistant response
   (dozens-hundreds of tokens); trial span is ~5-15 tokens.
   `reduction="mean"` divides by L differently on each side, so the
   gradient magnitudes are scaled differently. There is no principled
   "answer span" inside a general SFT example, so we accept this as an
   interpretation ceiling — validate via the removal counterfactual,
   not by trying to fabricate a comparable span on the SFT side.

### Reconstructing the prompt exactly

The prompt text comes from the raw simulation CSV:

* `SYSTEM` = the v6.0 multi-sys-report system message
  (from `sample_olmo3_checkpoints_v5_0_sycophancy.SYSTEM_CONTEXT`).
* `T2_user_message` =
  `f"{question_text}\n\n(A) ... (D) {option_d}{ANSWER_PROMPT}"`
* `T2_response` = the *actual* response the SFT model produced at
  simulation time, pulled from `turn2_raw_response` in the raw CSV.
  This matters: the model's gradient on T4 depends on what the
  assistant said at T2, so using a synthetic T2 response would change
  the attribution.
* `T4_user_message` =
  `f"{cue}\n\n{question + options}{ANSWER_PROMPT}"`.

If the reconstructed length exceeds `max_seq_len=2048`, we
**left-truncate** to preserve the cued-answer target span at the
sequence's tail (which is what determines the loss).

### Same projector as SFT

Critical: the test-side gradients must be projected with the **same
`R` matrix** as the SFT-side gradients, otherwise their projected
inner products are meaningless. We import the `build_projector()`
helper from `featurize_sft.py` and pass the same `proj_seed=0` and
`proj_dim=4096`.

### SLURM sharding

Mirrors the SFT-side pattern: 15-way array, `np.linspace(0, 4432, 16,
dtype=int)` slicing, output filenames
`trial_grads_shard{NN}_of15.{npy,idx.csv,meta.json}`. The submit
script is `bash/run_trak_featurize_trials_array.sh`. Per-shard wall
time ≈ 37 min (4432 / 15 ≈ 295 trials × ~7.5 s); full array wall ≈
37 min if all shards start together.

### Smoke test

Before launching the full 15-way array, run a single shard
interactively:

```bash
python analysis/trak/featurize_trials.py \
    --shard_idx 0 --num_shards 15 --smoke 5
```

Check that it writes `trial_grads_shard00_of15.{npy,idx.csv,meta.json}`,
the npy has shape `(5, 4096)`, and the meta has plausible `lo`/`hi`
values. Repeat with `--shard_idx 14` to verify the last shard slices
cleanly.

---

## Step 3 — Score (`score_trak.py`)

### What this step computes

Given:
* `P_sft ∈ R^{50000 × 4096}` — projected SFT-side gradients (15 shards
  concatenated, **un-normalized rows**)
* `P_trial ∈ R^{4432 × 4096}` — projected trial-side gradients (15
  shards concatenated)

we compute the TRAK attribution scores:

```
G = ΦᵀΦ + λI                # (4096 × 4096) Gram
S = P_trial @ G⁻¹ @ Φᵀ      # (4432 × 50000) scores
```

`G` is the Tikhonov-regularized Gram matrix in projected space. This
matches the TRAK paper's eq (13) and the upstream
`trak/score_computers.py` — **no `/n`**. Conceptually `(ΦᵀΦ)⁻¹`
inverts out redundancy: if 1,000 SFT examples have nearly identical
projected gradients (= a high-eigenvalue direction in `ΦᵀΦ`), the
inversion shrinks that direction so those examples don't dominate
the attribution. Important: this whitening is the mechanism that
absorbs per-example *gradient-magnitude* structure too — a source
with many high-norm examples gets down-weighted by `(ΦᵀΦ)⁻¹` because
those examples contribute disproportionately to the Gram. So **the
canonical run does NOT L2-normalize rows** before scoring; pre-stripping
magnitude defeats the mechanism the Gram inverse is supposed to perform.
(`--normalize` flag is kept for back-compat with the pre-2026-06
`_legacy_norm_overn` artifacts.)

### λ choice

With un-normalized 50k SFT-side projected grads, the eigenspectrum of
`ΦᵀΦ` is:

| Quantity | Value |
|---|---|
| min eigenvalue | 8.6 × 10⁵ |
| median eigenvalue | 4.6 × 10⁶ |
| max eigenvalue | **1.245 × 10¹⁰** |

The canonical default is **λ = 1 × 10⁷** (~8 × 10⁻⁴ of max_eig,
Mahalanobis whitening Frobenius ratio 0.69). Robustness-sweep
variants at λ = 10⁶ (lighter; ratio 0.29) and λ = 10⁸ (heavier;
ratio 0.91) are also produced. **λ is provisional** until the
group-removal counterfactual (see "NEXT TODO" in caveats) validates
the attribution causally; the λ sweep tests stability of the
source-distribution headline, not correctness. See
`method_documentation/v6_0_trak_lambda_fix.md` for the diagnosis
that motivated this revision.

### Why Cholesky instead of explicit inverse

We never form `G⁻¹` explicitly. Instead:

```python
L = torch.linalg.cholesky(G)                 # G = L L^T, lower triangular
x = torch.cholesky_solve(P_trial.T, L)       # solves G x = P_trial.T
scores = (x.T @ P_sft.T).cpu().numpy()       # (4432, 50000)
```

Cholesky-then-solve is numerically more stable than computing `G⁻¹`
and multiplying, and is roughly 3× faster.

The whole computation — Gram + Cholesky + scoring — runs in **<6
seconds** on a single H100 because everything is in the 4096-dim
projected space. The bottleneck is loading the 50000×4096 fp16 grad
matrix into GPU memory (~400 MB on disk → ~800 MB after fp32 promotion).

### Output

* `scores/attribution_scores_all_lam1e7.npy` — canonical `(4432, 50000)
  fp32`, ~887 MB
* `scores/attribution_scores_all_lam{1e6,1e8}.npy` — robustness sweep
* `scores/sft_idx_order.npy` — `(50000,)`, the `sft_idx` for each column
* `analysis_outputs/.../scores/trial_id_order.csv` — `(4432,)`, the
  `trial_id` for each row
* `analysis_outputs/.../scores/attribution_scores_all_lam{1e6,1e7,1e8}.meta.json` —
  pointers + parameters used

---

## Step 3.5 — Pre-2026-06 normalize+`/n` pipeline (legacy)

**Historical: kept here for context on the legacy `_norm_overn` artifacts.**

The first pass of this pipeline used `G = ΦᵀΦ/n + λI` with `λ=1e-2`
and L2-normalized rows. That setup was an attempt to address a
"gradient-norm confound" observed in the 300-trial pilot, where
high-norm SFT examples dominated top-K regardless of direction.

Two problems compounded:
1. **`/n_sft` in the Gram is non-canonical.** Paper eq (13) and
   upstream `trak/score_computers.py` use `K = ΦᵀΦ` (no `/n`). With
   the `/n`, our nominal `λ = 1e-2` corresponded to `λ_paper ≈
   n·λ = 500` — much heavier regularization than the paper's
   recommended range. Measurement: `λ=1e-2` was larger than 99.88%
   of the eigenvalues of `ΦᵀΦ/n`; only 5 of 4096 directions had
   eigenvalue > λ; the Frobenius ratio `‖G⁻¹‖_F / ‖(1/λ)I‖_F = 0.98`.
   In effect, `G⁻¹ ≈ (1/λ)I` and the whitening was **not happening**.
2. **L2-normalizing rows pre-strips the magnitude structure that
   `(ΦᵀΦ)⁻¹` is supposed to absorb.** With correct (un-divided) Gram
   and proper λ, the Mahalanobis-like whitening down-weights sources
   whose examples cluster in high-magnitude or high-frequency
   directions. Stripping magnitude first defeats that mechanism.

Combined, the legacy pipeline was approximately computing
`scores ≈ (1/λ) · P_trial · P_sftᵀ` — i.e., scaled cosine
similarity in projection space, which is **TracIn-style attribution,
not TRAK**.

### The fix

The 2026-06 lambda fix (`v6_0_trak_lambda_fix.md`):

1. Drop the `/n_sft` from the Gram — now `G = ΦᵀΦ + λI`.
2. Stop L2-normalizing rows before scoring (default `--normalize`
   off; flag retained for legacy comparison only).
3. Retune λ for the un-normalized eigenspectrum: canonical `λ = 1e7`,
   plus 1e6 and 1e8 robustness variants.

The pre-fix scores file and dependent PNGs are preserved in
`scores/legacy_norm_overn/` (scratch) and
`analysis_outputs/.../analysis/legacy_norm_overn/` (repo) with a
README explaining why they're not canonical. They are useful for
"what changed under the fix" comparisons — the legacy c10 top-3 was
`Persona MATH / Persona GSM / Python Algorithms`; the post-fix
canonical c10 top-3 is `FLAN / Dolci Tool Use / Wildchat`. The
qualitative composition differs.

---

## Step 4 — Plotting (3 scripts, all CPU)

The `analyze_attributions.py` script from the pilot is gone; its
numerical output (per-source lift tables, top-K example dumps) has been
folded into focused plot scripts below, and any remaining tabular
analysis is done in notebooks.

All plot scripts default to:
* `--scores_file attribution_scores_all_lam1e7.npy` (the canonical
  corrected-Gram un-normalized scores at the working-default λ)
* `--tag all_lam1e7` (output filename suffix)
* `--group_by cue_id` (per-cue analysis; one PNG per cue or
  cue-aware comparison)

Per-cue ordering on shared axes uses `cue_ranking_full` from
`trial_manifest_all_meta.json`, so cue columns appear left-to-right by
ascending checkpoint-pooled flip rate (matches
`per_cue_effectiveness_ranking.png`).

**Terminology note**: until the group-removal counterfactual (NEXT
TODO in caveats) validates causality, charts and outputs are labeled
"attribution share" / "attribution score," not "influence." The
former `plot_category_influence.py` was renamed to
`plot_category_attribution.py` and its output PNGs from
`category_influence_*.png` to `category_attribution_*.png`.

### Step 4a — Distribution diagnostic (`plot_attribution_distribution.py`)

For each cue, sort the (50,000,) mean-attribution vector descending and
plot rank vs score. Reference vertical lines at K=100/500/5000. Shows
where the long tail ends and the bulk begins — i.e., **which K is
worth interpreting**.

The pre-fix (legacy) run showed:
* median attribution ≈ 0 across cues; robust σ ≈ 0.7–1.1
* K=100 sits ~3σ above the median (rank@3σ ≈ 93 averaged across cues)
  → strongly tail-selected, **interpret seriously**
* K=500 sits ~2σ above → borderline tail
* K=5000 sits ~1σ above → mostly bulk → essentially a prevalence
  baseline, **don't draw substantive conclusions**

The K-trust guidance is expected to transfer qualitatively to the
post-fix canonical run, but verify by reading the
`attribution_distribution_by_cue_all_lam1e7.png` for the actual
post-fix tail shape.

### Step 4b — Per-cue category attribution share (`plot_category_attribution.py`)

For each trial: `category_share[c] = Σ(attr in c) / Σ(attr all)`.
Average per-category share across trials within each (cue_id,
sft_flip) group; plot one figure per cue with two panels (flipped /
not-flipped), one bar per category with SEM error bars.

Source_dataset only — domain plotting was removed 2026-06 to keep the
focus on the canonical attribution category (20 PNGs total, one per cue).

**Caveat**: the share metric divides by the per-trial sum of
attributions, which is sign-variable (often negative). On the legacy
run ~57% of trials had non-positive denominator, which complicates
the signed-mean interpretation. The lift-heatmap version below
sidesteps this and is the more reliable headline.

### Step 4c — Top-K composition (`plot_top_source.py`)

For each cue, average attribution across the cue's trials → (50,000,)
mean-attribution vector. Take the top-K examples. Two metrics:

* `--metric count` (default) — stacked bar of top-K examples by
  `source_dataset` or `domain`. One x-bar per cue (20 bars).
* `--metric lift` — heatmap of `log₂(lift)` where
  `lift_S = (top-K share of S) / (subsample share of S)`. Sources on
  y-axis sorted by mean lift across cues; cues on x-axis by
  effectiveness ranking. Diverging colormap centered at lift=1.

`--top_k` defaults to 100 (the high-SNR regime per the distribution
diagnostic). The canonical headline run produces both metrics at
K=100/500/5000 against `attribution_scores_all_lam1e7.npy`; the λ
robustness sweep produces lift heatmaps at K=100/500/5000 against
both `attribution_scores_all_lam1e6.npy` and
`attribution_scores_all_lam1e8.npy` to check whether the
source/domain composition is stable across regularization strength.

**Reading the λ-robustness comparison**: a top-K that's stable
across λ ∈ {1e6, 1e7, 1e8} for a given source is a stronger signal
than a top-K that appears only at one λ. **But** stability across λ
is not validation of causality — even a robust ranking could be a
robustly-wrong ranking. Only the removal counterfactual (caveat
section below) adjudicates correctness.

---

## Output layout summary

Small artifacts (manifests, metas, score metadata, analysis outputs)
live in the version-controlled repo:

```
analysis_outputs/v6_0_olmo3_7b_pilot/trak/
├── trial_manifest_all.csv               # 4,432 trials, all cues
├── trial_manifest_all_meta.json         # cue ranking, bucketing, edges
├── sft_subsample_meta.json
├── scores/
│   ├── attribution_scores_all_lam{1e6,1e7,1e8}.meta.json
│   ├── trial_id_order.csv
│   └── legacy_norm_overn/               # pre-2026-06 normalize+/n meta
└── analysis/                            # source_dataset only; domain removed 2026-06
    ├── README.md                                                 # subfolder layout
    ├── score_distribution/
    │   └── attribution_distribution_by_cue_all_lam1e7.png        (1)
    ├── category_attribution_per_cue/
    │   └── category_attribution_source_dataset_c{01..20}_all_lam1e7.png  (20)
    ├── top_k_composition_canonical/
    │   ├── top{100,500,5000}_source_stacked_by_cue_id_all_lam1e7.png     (3)
    │   └── top{100,500,5000}_source_lift_heatmap_by_cue_id_all_lam1e7.png (3)
    ├── top_k_composition_lambda_robustness/
    │   └── top{100,500,5000}_source_lift_heatmap_by_cue_id_all_lam{1e6,1e8}.png  (6)
    ├── top_k_composition_by_flip_outcome/
    │   └── top{100,500,5000}_source_lift_heatmap_by_flip_all_lam{1e6,1e7,1e8}.png (9 paired)
    └── legacy_norm_overn/               # pre-2026-06 _norm PNGs (53 files; includes domain)
```

Sizable artifacts (parquet, projected gradients, attribution matrix)
live in scratch:

```
/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/
├── sft_subsample.parquet                # ~74 MB
├── sft_grads/
│   ├── sft_grads_shard{NN}_of15.npy     # (~n_shard × 4096) fp16
│   ├── sft_grads_shard{NN}_of15.idx.npy
│   └── sft_grads_shard{NN}_of15.meta.json
├── trial_grads_all/
│   ├── trial_grads_shard{NN}_of15.npy   # (~n_shard × 4096) fp16
│   ├── trial_grads_shard{NN}_of15.idx.csv
│   └── trial_grads_shard{NN}_of15.meta.json
└── scores/
    ├── attribution_scores_all_lam1e7.npy  # (4432, 50000) fp32 ~887 MB (canonical)
    ├── attribution_scores_all_lam1e6.npy  # robustness, lighter reg
    ├── attribution_scores_all_lam1e8.npy  # robustness, heavier reg
    ├── sft_idx_order.npy
    └── legacy_norm_overn/                  # pre-2026-06 normalize+/n scores
        ├── attribution_scores_all_norm.npy
        └── README.txt
```

---

## Caveats and known limitations

1. **Single-checkpoint TRAK.** The single converged SFT checkpoint we
   attribute against is noisy compared to the full TRAK ensemble. Use
   scores as **rankings**, not magnitudes; do permutation tests over
   top-K source distributions for any significance claim.
2. **Subsampled SFT corpus.** We attribute against 50,000 of ~2 M
   training examples. An influential example outside the subsample is
   invisible to this analysis. The stratified sample (capped per
   source) preserves tail-source coverage, but a
   target-source-specific oversampling pass would be needed if a
   particular source looked interesting.
3. **Trial-side loss asymmetries vs SFT.** See §"Step 2 — Deliberate
   asymmetries". Trailer is masked out on the trial side (but not SFT
   side), and the loss-span length differs by an order of magnitude.
   Both are accepted tradeoffs, not bugs.
4. **Which K to trust.** The distribution diagnostic says K=100 is
   ~3σ-selected (real signal), K=500 ~2σ (borderline), K=5000 ~1σ
   (essentially baseline). Headline conclusions should come from
   K=100; K=5000 should be treated as a prevalence-baseline reference.
5. **Sequence-length cap.** SFT examples truncated at 1,024 tokens
   (right-truncated); trial prompts at 2,048 (left-truncated to
   preserve the cued-answer target span). Long math derivations may
   have their tails clipped — this is logged as a per-shard count but
   can't easily be undone without re-featurizing at higher length.
6. **fp16 projection.** Quantization noise from the fp16 projector is
   small relative to the JL projection noise at `d=4096`. If results
   look noisier than expected, bumping `proj_dim` to 8192 would help
   (~2× memory and time).
7. **No counterfactual retrain validation.** We have not removed any
   of the top-attributed examples and re-trained the SFT model to
   confirm that the predicted attribution direction holds. That would
   be the gold-standard test; with one checkpoint and 24 GPU-hours
   per retrain, it's a separate study.
8. **Gradient-norm confound is absorbed by the (now-corrected) Gram
   inverse — DO NOT use `--normalize`.** The pre-2026-06 attempt to
   fix the confound by L2-normalizing rows is now known to defeat
   the mechanism `(ΦᵀΦ)⁻¹` uses to absorb per-example magnitude;
   that legacy path is preserved in `legacy_norm_overn/` for
   comparison only. The canonical run uses un-normalized rows with
   the corrected Gram. See `v6_0_trak_lambda_fix.md`.
9. **Cue-rank bucketing is informational, not analytical.** The
   `cue_rank` column (low/med/high thirds of the 20-cue ranking) is in
   the manifest for back-compat with the pilot, but the canonical
   analysis is per-cue. The `high` bucket alone spans a 55% → 92%
   flip-rate range; treating it as a single category averages out
   real structure.
10. **λ is provisional pending causal validation.** Canonical λ=1e7
    was chosen because it puts the Mahalanobis-like whitening into
    its substantive regime (Frobenius ratio 0.69, neither pure
    dot-product nor inflating low-eigenvalue noise). The robustness
    sweep at λ ∈ {1e6, 1e7, 1e8} tests whether the headline
    source/domain composition is stable across regularization
    strength — and at the time of writing, the c10 (high-flip cue)
    top-3 of `FLAN / Dolci Tool Use` is stable across all three.
    But **stability across λ does not imply causal correctness** —
    a robust ranking could be a robustly-wrong ranking. The removal
    counterfactual below is what licenses the causal claim.

---

## NEXT TODO — group-removal counterfactual (blocked on compute)

**The lambda fix is a formulation correction; it is NOT a validation
of causality.** Until we retrain SFT without the TRAK-top-K examples
and confirm flipping drops more than under control removals, the
rankings remain *descriptive attribution shares*, not *causal
influence*.

The plan (see `v6_0_trak_lambda_fix.md` §"NEXT TODO" for the full
spec): 4 re-SFTs from OLMo-3-7B Base — baseline (no removal), TRAK
top-K removed, length-matched-random K removed, BM25-matched K
removed — evaluated on the same high-flip cues. TRAK should reduce
flipping more than both controls.

Cost estimate: ~96 GPU-hours (4 × 24 h SFT runs). Not feasible on
current compute budget; deferred until compute opens up.

**Until then, all charts and tables labeled "attribution share" or
"attribution score" are descriptive only; do not present them as
causal influence claims.**

---

## Quick-start re-run recipe

If you need to re-run the whole pipeline from scratch:

```bash
# (Step 0) sampling — both are fast, single-CPU, no SLURM needed
python analysis/trak/sample_trials.py --all_trials
python analysis/trak/sample_sft_subset.py

# (Step 1) SFT featurize — 15-way SLURM array, ~3 h end-to-end
sbatch bash/run_trak_featurize_sft.sh

# (Step 2) trial featurize — 15-way SLURM array, ~37 min end-to-end
sbatch bash/run_trak_featurize_trials_array.sh
# or with the model's actual emitted answer as the target:
USE_EMITTED=1 sbatch bash/run_trak_featurize_trials_array.sh

# (Step 3) score — single GPU, ~2 min per λ. Canonical = un-normalized
# rows with corrected Gram. Default LAM=1e7 (the working canonical).
sbatch bash/run_trak_score.sh                  # → attribution_scores_all_lam1e7.npy
# robustness sweep:
LAM=1e6 OUT_NAME=attribution_scores_all_lam1e6.npy sbatch bash/run_trak_score.sh
LAM=1e8 OUT_NAME=attribution_scores_all_lam1e8.npy sbatch bash/run_trak_score.sh

# (Step 4) plots — CPU, seconds each. All plots default to source_dataset
# (domain plotting was removed 2026-06).
python analysis/trak/plot_attribution_distribution.py
python analysis/trak/plot_category_attribution.py
for K in 100 500 5000; do
  python analysis/trak/plot_top_source.py --top_k $K                   # count, canonical λ
  python analysis/trak/plot_top_source.py --top_k $K --metric lift     # lift,  canonical λ
done
# Robustness sweep — lift heatmaps at lighter/heavier λ:
for LAM_FILE in attribution_scores_all_lam1e6.npy attribution_scores_all_lam1e8.npy; do
  TAG=${LAM_FILE%.npy}; TAG=${TAG#attribution_scores_}
  for K in 100 500 5000; do
    python analysis/trak/plot_top_source.py \
        --scores_file $LAM_FILE --tag $TAG --top_k $K --metric lift
  done
done
# Flip-split — paired (flipped|not-flipped) lift heatmaps with bucketed cues
for LAM_FILE in attribution_scores_all_lam1e6.npy attribution_scores_all_lam1e7.npy attribution_scores_all_lam1e8.npy; do
  TAG=${LAM_FILE%.npy}; TAG=${TAG#attribution_scores_}
  for K in 100 500 5000; do
    python analysis/trak/plot_top_source.py \
        --scores_file $LAM_FILE --tag $TAG --top_k $K \
        --metric lift --group_by cue_rank --flip_split
  done
done
```

To re-run with a different projector seed (sanity-check projection noise):

```bash
# Move scratch out of the way first if you want to keep both seeds, then:
PROJ_SEED=1 sbatch bash/run_trak_featurize_sft.sh
PROJ_SEED=1 sbatch bash/run_trak_featurize_trials_array.sh
OUT_NAME=attribution_scores_all_lam1e7_seed1.npy sbatch bash/run_trak_score.sh
```

The seed must match between SFT and trial featurize jobs — otherwise
the projected gradients live in different subspaces and the
attribution scores are noise.

## Doc cross-references

| Topic | Doc |
|---|---|
| Trial-side loss formulation (masked-mean CE over cued-answer span) | `v6_0_trak_loss_identity_fix.md` |
| Expansion from 300-trial pilot to 4,432-trial canonical | `v6_0_trak_expansion_4432_trials.md` |
| Gram-formulation correction (`/n` drop + un-normalized canonical) | `v6_0_trak_lambda_fix.md` |
| **This doc** — overall pipeline reference | `v6_0_trak_attribution.md` |
