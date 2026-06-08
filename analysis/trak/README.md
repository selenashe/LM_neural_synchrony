# TRAK attribution for v6.0 sycophancy flipping

Trace Base→SFT flipping behavior on the v6.0 multi-sys-report sycophancy
benchmark back to specific OLMo-3 SFT training examples using TRAK
(https://github.com/MadryLab/trak).

## Output layout

Small artifacts (manifests, metas, score metadata) live under the repo:

```
analysis_outputs/v6_0_olmo3_7b_pilot/trak/
├── trial_manifest_all.csv               # 4,432 trials (all 20 cues × all
├── trial_manifest_all_meta.json         #   filtered questions per cue)
├── sft_subsample_meta.json              # parquet itself is in scratch
├── scores/
│   ├── attribution_scores_all_lam{1e6,1e7,1e8}.meta.json
│   ├── trial_id_order.csv
│   └── legacy_norm_overn/               # pre-2026-06 normalize+/n meta
└── analysis/                            # PNGs from the plot_* scripts
    ├── README.md                        # explains the subfolder layout
    ├── score_distribution/              # 1 PNG (diagnostic)
    ├── category_attribution_per_cue/    # 23 PNGs (20 per-cue + 3 low/med/high bucket aggregates)
    ├── top_k_composition_canonical/     # 6 PNGs (λ=1e7, source × {stacked,lift} × 3 Ks)
    ├── top_k_composition_lambda_robustness/  # 6 PNGs (λ=1e6, 1e8 source lift)
    ├── top_k_composition_by_flip_outcome/    # 9 paired PNGs (source, flipped|not-flipped side-by-side, bucketed)
    └── legacy_norm_overn/               # 53 pre-2026-06 PNGs
```

Sizable artifacts (parquet, projected gradients, attribution matrix) live
in scratch:

```
/juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/
├── sft_subsample.parquet                # ~74 MB
├── sft_grads/
│   ├── sft_grads_shard{NN}_of15.npy     # (~n_shard × 4096) fp16
│   ├── sft_grads_shard{NN}_of15.idx.npy
│   └── sft_grads_shard{NN}_of15.meta.json
├── trial_grads_all/                     # 15 shards from the array job
│   ├── trial_grads_shard{NN}_of15.npy   # (~n_shard × 4096) fp16
│   ├── trial_grads_shard{NN}_of15.idx.csv
│   └── trial_grads_shard{NN}_of15.meta.json
└── scores/
    ├── attribution_scores_all_lam1e7.npy  # (4432, 50000) fp32 (~887 MB, canonical)
    ├── attribution_scores_all_lam1e6.npy  # robustness, lighter reg
    ├── attribution_scores_all_lam1e8.npy  # robustness, heavier reg
    ├── sft_idx_order.npy
    └── legacy_norm_overn/                 # pre-2026-06 normalize+/n scores
```

## Pipeline

```
                                +-----------------------------+
  trial_manifest_all.csv ─────► |  featurize_trials.py        |
  (4,432 trials, 20 cues)       |  trial_grads_shard*_of15.*  |
                                +--------------+--------------+
                                               |
                                               ▼   (15-way array)
  sft_subsample.parquet ──►   +----------------------+      +-------------+
  (50,000 SFT examples)       |  featurize_sft.py    |  ──► | score_trak  |
                              |  sft_grads_shard*    |      |  scores.npy |
                              +----------------------+      +------+------+
                                          ▲                        |
                                  (15-way array)                   ▼
                                                          plot_* scripts
                                                          (PNGs in analysis/)
```

## Step 0: sampling (already done; outputs in place)

* `sample_trials.py --all_trials` — emits one row per filtered (q, cue) pair
  in cell_summary. 4,432 rows across 20 cues. Bucketed into low/med/high
  thirds by checkpoint-pooled flip rate (purely informational; per-cue
  analysis is the default downstream). Writes
  `trial_manifest_all.csv` + `trial_manifest_all_meta.json` (which carries
  the full per-cue effectiveness ranking, used downstream to order plots).
* `sample_sft_subset.py` — samples 50k Dolci-Instruct-SFT rows
  (stratified by source dataset, capped at 25% per source). Writes
  `sft_subsample.parquet` to **scratch** and the meta json to the repo.

If you need to re-sample:

```bash
python analysis/trak/sample_trials.py --all_trials
python analysis/trak/sample_sft_subset.py
```

The script still supports the legacy 300-trial pilot mode (`sample_trials.py`
with no flag), which emits `trial_manifest.csv` instead — but the canonical
analysis runs against `trial_manifest_all.csv`.

## Step 1: smoke-test (recommended first)

The trial featurize is the most fragile bit (custom prompt reconstruction,
chat-template tokenization, target-span masking). Smoke-test a single shard
interactively before launching the array:

```bash
python analysis/trak/featurize_trials.py \
    --shard_idx 0 --num_shards 15 --smoke 5
```

Outputs land in `trial_grads_all/` as
`trial_grads_shard00_of15.{npy,idx.csv,meta.json}`. Verify shape `(5,
4096) fp16` and a non-zero `n_emitted_target` only if you passed
`--use_emitted`. Repeat with `--shard_idx 14` to verify the last shard
slices cleanly.

Optionally also smoke-test one SFT shard to verify the chat-template +
loss-mask logic:

```bash
SMOKE=16 sbatch --array=0 bash/run_trak_featurize_sft.sh   # ~1-2 min
```

## Step 2: full SFT featurize (15-way array)

```bash
sbatch bash/run_trak_featurize_sft.sh
```

15 jobs × ~3,334 examples × ~3 s/ex ≈ 2.5-3 h wall per shard
(12 h cap for safety). All 15 in parallel ≈ ~3 h end-to-end if all start
together. Per-shard outputs land in `scratch/.../sft_grads/`.

This step is unchanged from the pilot — SFT-side gradients depend only on
the SFT subsample + the model, not on which trials you score against.

## Step 3: trial featurize (15-way array)

Must use the **same** `--proj_seed` and `--proj_dim` as the SFT job (the
default env vars match: `PROJ_SEED=0`, `PROJ_DIM=4096`).

```bash
sbatch bash/run_trak_featurize_trials_array.sh
# or with the model's actual emitted answer as the target:
USE_EMITTED=1 sbatch bash/run_trak_featurize_trials_array.sh
```

4,432 trials / 15 shards ≈ 295 per shard × ~7.5 s ≈ ~37 min per shard,
~37 min wall-clock parallel.

## Step 4: score

Stacks SFT shards + trial shards, builds the 4096×4096 Gram matrix
`G = ΦᵀΦ + λI` (matches TRAK paper eq 13 and upstream
`trak/score_computers.py`), Cholesky-solves, multiplies. ~2 min on a
single 80G GPU (bottleneck is loading ~800 MB of sft_grads into VRAM).

The defaults are set for the canonical run (un-normalized rows,
λ=1e7). Just submit:

```bash
sbatch bash/run_trak_score.sh        # → attribution_scores_all_lam1e7.npy
```

Robustness sweep (run if you want to test λ stability of source
composition):

```bash
LAM=1e6 OUT_NAME=attribution_scores_all_lam1e6.npy sbatch bash/run_trak_score.sh
LAM=1e8 OUT_NAME=attribution_scores_all_lam1e8.npy sbatch bash/run_trak_score.sh
```

Outputs (per λ):
* `scratch/.../scores/attribution_scores_all_lam{N}.npy` — `(4432,
  50000) fp32` ≈ 887 MB
* `scratch/.../scores/sft_idx_order.npy` — column ordering (sft_idx values)
* `analysis_outputs/.../trak/scores/trial_id_order.csv` — row ordering
* `analysis_outputs/.../trak/scores/attribution_scores_all_lam{N}.meta.json`

**Do NOT use `--normalize` (or `NORMALIZE=1`) for canonical
analysis.** Pre-2026-06 the `--normalize` flag was the canonical
path, combined with `G = ΦᵀΦ/n + λI`; that turned out to be
effectively TracIn-style cosine attribution rather than TRAK (the
whitening was inactive because λ swamped the data term, and
normalizing pre-stripped the magnitude structure the Gram inverse is
supposed to absorb). The flag is kept for back-comparison only. See
`method_documentation/v6_0_trak_lambda_fix.md`.

## Step 5: plotting

All plot scripts default to reading `attribution_scores_all_lam1e7.npy`
(the canonical un-normalized corrected-Gram scores) and grouping
trials by `cue_id` (per-cue analysis; one PNG per cue or per cue-aware
comparison). Per-cue ordering on shared axes is by ascending
checkpoint-pooled flip rate (c11 = least effective → c10 = most),
pulled from `cue_ranking_full` in `trial_manifest_all_meta.json` to
match `per_cue_effectiveness_ranking.png`.

```bash
# Distribution diagnostic — which K to interpret?
python analysis/trak/plot_attribution_distribution.py

# Per-cue category attribution shares (20 PNGs)
python analysis/trak/plot_category_attribution.py
# Low/med/high bucket aggregates (3 PNGs)
python analysis/trak/plot_category_attribution.py --group_by cue_rank

# Top-K composition: stacked bars (count) + lift heatmaps
for K in 100 500 5000; do
  python analysis/trak/plot_top_source.py --top_k $K                # count
  python analysis/trak/plot_top_source.py --top_k $K --metric lift  # lift
done

# λ-robustness sweep — lift heatmaps at lighter/heavier reg
for LAM_FILE in attribution_scores_all_lam1e6.npy attribution_scores_all_lam1e8.npy; do
  TAG=${LAM_FILE%.npy}; TAG=${TAG#attribution_scores_}
  for K in 100 500 5000; do
    python analysis/trak/plot_top_source.py \
        --scores_file $LAM_FILE --tag $TAG --top_k $K --metric lift
  done
done

# Behavior-split lift heatmaps — flipped vs not-flipped, bucketed
# into low/med/high cue thirds so per-cell n_trials stays > 190
for LAM_FILE in attribution_scores_all_lam1e6.npy attribution_scores_all_lam1e7.npy attribution_scores_all_lam1e8.npy; do
  TAG=${LAM_FILE%.npy}; TAG=${TAG#attribution_scores_}
  for K in 100 500 5000; do
    python analysis/trak/plot_top_source.py \
        --scores_file $LAM_FILE --tag $TAG --top_k $K \
        --metric lift --group_by cue_rank --flip_split
  done
done
```

PNGs are auto-routed to subfolders under
`analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis/` based on what's
being plotted; see [`analysis/README.md`](../../analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis/README.md)
for the layout. Briefly:

- `score_distribution/` — diagnostic
- `category_attribution_per_cue/` — per-cue category bars (40 PNGs)
- `top_k_composition_canonical/` — λ=1e7 stacked + lift, all-trials, per-cue
- `top_k_composition_lambda_robustness/` — λ=1e6 + λ=1e8 lift heatmaps
- `top_k_composition_by_flip_outcome/` — flipped vs not-flipped × low/med/high buckets

To revert to the old low/med/high bucket grouping, pass
`--group_by cue_rank`.

**Terminology note**: until the group-removal counterfactual (see
§Caveats > "NEXT TODO") validates causality, charts and outputs are
labeled "attribution share" / "attribution score," not "influence."
The legacy `plot_category_influence.py` was renamed to
`plot_category_attribution.py`.

## Test-loss definition (loss-identity fix, applied 2026-06)

For each trial we append the **cued-answer assistant continuation** as the
target span (default canonical "({letter}) {option_text}", or with
`--use_emitted` the leading "({letter}) ..." clause of the model's actual
T4 response) and compute `model(input_ids, labels=labels).loss` — i.e.,
masked-mean cross-entropy over the cued-answer **content tokens only**
(trailer `<|im_end|>\n` is excluded from the loss span to avoid signal
dilution on a tiny ~5-token target).

This makes the trial-side and SFT-side gradients both `∇f` of the same
scalar `f = mean(CE over assistant-text span)`, which is what makes
TRAK's whitening `(ΦᵀΦ)⁻¹` produce interpretable influence scores. See
`method_documentation/v6_0_trak_loss_identity_fix.md` for the full
rationale and the two deliberate asymmetries vs SFT (trailer masking,
length-mean asymmetry).

## SFT loss definition

Standard next-token CE on assistant-content tokens only
(`labels[~assistant_mask] = -100`). The chat template is hand-rolled in
`build_chat_text_and_mask()` because the OLMo-3 SFT tokenizer ships with
`chat_template=None`. Trailer `<|im_end|>\n` is **included** in the
SFT-side loss span (it's part of a real multi-token training response).

## Caveats

* **Single-checkpoint TRAK** — we have one SFT checkpoint, not the ensemble
  TRAK was designed for. Scores are noisy point estimates; treat them as
  **rankings**, not magnitudes. Permutation tests on top-K category
  distributions are the right inference target.
* **Length asymmetry between trial and SFT loss spans** — accepted as an
  interpretation ceiling. Trial spans are ~5-15 tokens (the cued answer);
  SFT spans are dozens-hundreds (full assistant responses). The
  `reduction="mean"` divides by L differently on each side. Use the
  removal-counterfactual validation, not absolute score magnitudes.
* **Which top-K to interpret** — `plot_attribution_distribution.py` shows
  the score-vs-rank curve per cue. As of the 2026-06 run, K=100 sits ~3σ
  above the noise floor; K=500 ~2σ; K=5000 ~1σ (mostly bulk). Lean on
  K=100 for headline conclusions; K=5000 is essentially a prevalence
  baseline.
* **Sequence-length cap** — SFT examples truncated at 1024 tokens
  (right-truncated); trial prompts at 2048 (left-truncated, preserving
  the cued-answer target span). A handful of long rows have their tail
  clipped; this is logged via per-shard counters.
* **Bf16 grads, fp16 projection** — chosen for memory; quantization noise
  is small relative to the projection noise of JL at D=4096.
* **λ is provisional** — canonical λ=1e7 was picked because it puts
  the Mahalanobis-like whitening into its substantive regime
  (Frobenius ratio 0.69). The robustness sweep at λ ∈ {1e6, 1e8}
  checks whether the source/domain composition is stable. Stability
  across λ does NOT validate causality — it just rules out
  "different λ → different headline." See
  `method_documentation/v6_0_trak_lambda_fix.md`.
* **Gradient-norm confound — absorbed by the corrected Gram, DO NOT
  use `--normalize`.** Pre-2026-06 the canonical path L2-normalized
  rows; that turned out to defeat the mechanism `(ΦᵀΦ)⁻¹` uses to
  absorb per-example magnitude. Legacy `_norm_overn` artifacts are
  preserved in `legacy_norm_overn/` subdirs for back-comparison only.

### NEXT TODO — group-removal counterfactual (blocked on compute)

**Until this runs, the rankings are descriptive attribution, not
causal influence.** All charts and outputs label themselves
"attribution share / attribution score" accordingly.

The plan: 4 re-SFTs from OLMo-3-7B Base — baseline (no removal),
TRAK top-K removed (from λ=1e7 high-flip-cue ranking),
length-matched-random K removed, BM25-matched K removed. Eval all
4 on the same high-flip trials and check that TRAK-removal reduces
flipping more than both controls. ~96 GPU-h total; not feasible on
current budget. See `method_documentation/v6_0_trak_lambda_fix.md`
§"NEXT TODO" for the full spec.

## Re-running with a different projector seed

```bash
PROJ_SEED=1 sbatch bash/run_trak_featurize_sft.sh
PROJ_SEED=1 sbatch bash/run_trak_featurize_trials_array.sh
OUT_NAME=attribution_scores_all_lam1e7_seed1.npy sbatch bash/run_trak_score.sh
```

Outputs overwrite in place. Move/rename `trak_v6_0_olmo3_7b_pilot/` in
scratch first if you want both seeds preserved.
