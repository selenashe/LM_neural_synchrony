# TRAK attribution for v6.0 sycophancy flipping

Trace Base→SFT flipping behavior on the v6.0 multi-sys-report sycophancy benchmark
back to specific OLMo-3 SFT training examples using TRAK
(https://github.com/MadryLab/trak).

## Output layout

Small artifacts (manifests, metas, score metadata) live under the repo:
```
analysis_outputs/v6_0_olmo3_7b_pilot/trak/
├── trial_manifest.csv                  # 300 trials, paired
├── trial_manifest_meta.json
├── sft_subsample_meta.json             # parquet itself is in scratch
└── scores/
    ├── attribution_meta.json
    └── trial_id_order.csv
```

Sizable artifacts (parquet, projected gradients, attribution matrix) live in
scratch:
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
├── trial_grads_smoke/                  # only when --smoke is used
└── scores/
    ├── attribution_scores.npy          # (300, 50000) fp32 (~60 MB)
    └── sft_idx_order.npy
```

## Pipeline

```
                                +----------------------+
  trial_manifest.csv  ──────►   |  featurize_trials.py |
  (300 paired trials)           |  trial_grads.npy     |
                                +----------+-----------+
                                           |
                                           ▼
  sft_subsample.parquet  ──►   +----------------------+      +------------+
  (50,000 SFT examples)        |  featurize_sft.py    |  ──► | score_trak |
                               |  sft_grads_shard*    |      | scores.npy |
                               +----------------------+      +------------+
                                           ▲
                                  (15-way SLURM array)
```

## Step 0: sampling (already done; outputs in place)

* `sample_trials.py` — picks 3 cues at flip-effectiveness rank 1/10/19
  (c11 / c18 / c10) and 100 questions per cue (300 trials total), stratified
  into SFT-T2 confidence quartiles. Writes `trial_manifest.csv` to
  `analysis_outputs/.../trak/`.
* `sample_sft_subset.py` — samples 50k dolci-instruct-sft rows
  (stratified by source dataset). Writes `sft_subsample.parquet` to **scratch**
  and the meta json to `analysis_outputs/.../trak/`.

If you need to re-sample:

```bash
python analysis/trak/sample_trials.py
python analysis/trak/sample_sft_subset.py
```

## Step 1: smoke-test (recommended first)

The trial featurize is the most fragile bit (custom prompt reconstruction,
candidate-letter token picking, two forward passes per example). Smoke-test it
first — it's a single GPU, no array dep, ~10-15 min, and outputs go to
`trial_grads_smoke/` so the real run is unaffected.

```bash
SMOKE=120 sbatch bash/run_trak_featurize_trials.sh    # ~10 min
# or
SMOKE=180 sbatch bash/run_trak_featurize_trials.sh    # ~15 min
```

Check:
```bash
python -c "
import numpy as np
p = np.load('/juice6/scr6/nlp/interp-models/OLMo-3-7B/'
            'trak_v6_0_olmo3_7b_pilot/trial_grads_smoke/trial_grads.npy')
print(p.shape, p.dtype, '(nonzero rows:', int((p != 0).any(axis=1).sum()), ')')
"
```

Expected: `(120, 4096) float16` (or 180), with ~all rows non-zero.

Optionally also smoke-test one SFT shard to verify the chat-template + loss
mask logic:

```bash
SMOKE=16 sbatch --array=0 bash/run_trak_featurize_sft.sh   # ~1-2 min
```

## Step 2: full SFT featurize (15-way array)

```bash
sbatch bash/run_trak_featurize_sft.sh
```

15 jobs × ~3,334 examples × ~3 s/ex ≈ 2.5-3 h wall per shard
(12 h cap for safety). All 15 in parallel = ~3 h end-to-end if all start
together. Per-shard outputs land in `scratch/.../sft_grads/`.

## Step 3: trial featurize (single GPU, full run)

Must use the **same** `--proj_seed` and `--proj_dim` as the SFT job
(defaults match: 0 / 4096).

```bash
sbatch bash/run_trak_featurize_trials.sh
```

300 trials × ~5 s ≈ 25 min.

## Step 4: score

Stacks SFT shards, builds the 4096×4096 Gram matrix with Tikhonov λ=1e-2,
Cholesky-solves, multiplies. ~5-15 min on a single 80G GPU.

```bash
sbatch bash/run_trak_score.sh
```

Outputs:
* `scratch/.../scores/attribution_scores.npy` — `(300, 50000) fp32` (~60 MB)
* `scratch/.../scores/sft_idx_order.npy` — column ordering (sft_idx values)
* `analysis_outputs/.../trak/scores/trial_id_order.csv` — row ordering
* `analysis_outputs/.../trak/scores/attribution_meta.json` — also records the
  scratch paths for the big arrays

## Test-loss definition

For each trial we measure the gradient of
`-log P(cued_wrong_letter_token | prompt_up_to_open_assistant_turn)`
where the candidate letter token is the variant with max next-token logit
(matches how `turn4_prob_X` was extracted at simulation time). This is the
gradient of *being sycophantic*, so positive attribution = "training this
example pushes the model toward picking the cued wrong answer."

The prompt is reconstructed from the raw SFT CSV (system + T2 user +
recorded T2 assistant response + T4 user + open assistant turn), matching
the hand-rolled ChatML format used in `sample_olmo3_checkpoints_v5_0_*`.

## SFT loss definition

Standard next-token CE on assistant-content tokens only
(`labels[~assistant_mask] = -100`). The chat template is hand-rolled in
`build_chat_text_and_mask()` because the OLMo-3 SFT tokenizer ships with
`chat_template=None`.

## Caveats

* **Single-checkpoint TRAK** — we have one SFT checkpoint, not the ensemble
  TRAK was designed for. Scores are noisy point estimates; treat them as
  **rankings**, not magnitudes. Permutation tests on top-K category
  distributions are the right inference target.
* **Sequence-length cap** — SFT examples truncated at 1024 tokens, trials at
  2048. A handful of long SFT rows will have their tail clipped; this is
  logged via the OOM/empty-mask counters.
* **Bf16 grads, fp16 projection** — chosen for memory; quantization noise is
  small relative to the projection noise of JL at D=4096.
* **λ tuning** — if scores look degenerate try `LAM=1e-1` or `1e-3`.

## Re-running with a different projector seed

```bash
PROJ_SEED=1 sbatch bash/run_trak_featurize_sft.sh
PROJ_SEED=1 sbatch bash/run_trak_featurize_trials.sh
sbatch bash/run_trak_score.sh
```

Outputs overwrite in place. Move/rename `trak_v6_0_olmo3_7b_pilot/` in scratch
first if you want both seeds preserved.
