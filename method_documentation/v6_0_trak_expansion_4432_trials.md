# V6.0 TRAK — expansion to all 4,432 filtered trials with 15-way sharding

Sister change to `v6_0_trak_loss_identity_fix.md`. The pilot featurized
300 trials (100 questions × 3 cues, stratified). This change generalizes
the trial pipeline to **every** filtered (question, cue) pair in
`cell_summary.csv` — 4,432 trials across **20 cues** — and adds SLURM-
array sharding so the run wall-clock stays comparable to the pilot.

---

## TL;DR

| | Pilot (existing) | Expanded (this change) |
|---|---|---|
| Manifest | `trial_manifest.csv` (300 rows) | `trial_manifest_all.csv` (4,432 rows) |
| Sampling | 3 cues × 100 questions, quartile-stratified | none — every (q, cue) pair in SFT's cell_summary slice |
| `cue_rank` | 3 specific cues labeled `low`/`med`/`high` | 20 cues bucketed into thirds (`low`=6, `med`=7, `high`=7) |
| `t2_conf_quartile` | quartile of the 100 chosen questions | quartile of the full 368-question candidate pool |
| Trial-side featurizer | single GPU, ~37 min | 15-way SLURM array, ~37 min wall-clock |
| Trial-grad output | `trial_grads.npy` (single file) | `trial_grads_shard{NN}_of{MM}.{npy,idx.csv,meta.json}` |
| Scoring | unchanged code path | needs `--num_trial_shards 15` to concatenate |

The pilot artifacts in `trial_manifest.csv`, `trial_grads/`, and
`scores/attribution_scores*.npy` are **not** touched — kept for back-
comparison against the expanded run.

---

## Trial count derivation

```
raw SFT CSV (sotopia_results_olmo3_instruct_v6_0_multi_sys_report_pilot/
             results_OLMo-3-7B-Instruct-SFT_seed0.csv)        : 10,000 rows
↓ filter to T2 correct, T4 parseable, surviving all 4 ckpts
cell_summary.csv (SFT slice)                                  :  4,432 rows
↓ this is what trial_manifest_all.csv enumerates
trial_manifest_all.csv                                        :  4,432 rows
```

Per-cue counts range from 186 (`c16`) to 262 (`c20`), averaging ~222 per
cue. The 20 cues are bucketed into thirds by checkpoint-pooled mean flip
rate: `low` = ranks 0–5 (6 cues), `med` = 6–12 (7 cues), `high` = 13–19
(7 cues). Per-bucket trial totals from the actual run: 1,447 low / 1,500
med / 1,485 high.

**Cue-count drift**: `v6_0_trak_attribution.md` says "19 cues" but the
current `cell_summary.csv` (regenerated 2026-05-28) has **20**. Likely a
cue was added between the pilot and now; the canonical doc should be
updated.

**Manifest stability**: re-running `sample_trials.py` (default mode)
today produces a different 300-question selection than the May 26 pilot
manifest, because `cell_summary.csv` was regenerated on May 28 and the
candidate pool changed. The existing `trial_manifest.csv` is therefore
the authoritative pilot manifest — don't regenerate it. The expansion
runs against `trial_manifest_all.csv`, which is generated fresh from the
current `cell_summary.csv`.

---

## New CLI flags

### `analysis/trak/sample_trials.py`

- `--all_trials` — skip cue selection and quartile stratification; emit
  one manifest row per filtered (q, cue) pair. Writes
  `trial_manifest_all.csv` + `trial_manifest_all_meta.json`. Does not
  touch the 300-row default-mode files.

### `analysis/trak/featurize_trials.py`

- `--shard_idx N` (default 0)
- `--num_shards M` (default 1)

Backwards-compatible: `--num_shards 1` reproduces the single-process
behavior of the pilot. Output filenames are always sharded — the legacy
`trial_grads.npy` filename is gone — so existing scoring runs against
the pilot data either keep the old file in place or pass
`--num_trial_shards 1` (the default) when calling `score_trak.py`.

The manifest is sliced via `np.linspace(0, n_total, num_shards+1, int)`,
matching `featurize_sft.py`.

### `analysis/trak/score_trak.py`

- `--num_trial_shards N` (default 1)

Default = legacy single-file layout. Set to 15 to concatenate sharded
trial grads from the expansion run.

---

## Files modified / added

| Path | Change |
|---|---|
| `analysis/trak/sample_trials.py` | + `--all_trials` flag and helper |
| `analysis/trak/featurize_trials.py` | + sharding args, sharded output filenames |
| `analysis/trak/score_trak.py` | + `--num_trial_shards`, sharded trial-grad loader |
| `bash/run_trak_featurize_trials_array.sh` (NEW) | 15-way SLURM array script |
| `method_documentation/v6_0_trak_expansion_4432_trials.md` (this doc) | NEW |

The existing `bash/run_trak_featurize_trials.sh` (single-job) is
**preserved** for smoke runs and legacy 300-trial invocations.

---

## Re-run recipe

### 1. Generate the expanded manifest

```bash
cd /juice6/u/jshe/nlp/LM_neural_synchrony
python analysis/trak/sample_trials.py --all_trials
# → analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv (4,432 rows)
```

Verify: `wc -l analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv`
should print **4433** (header + 4,432 rows). The script also prints
per-bucket counts and a sanity flip-rate breakdown.

### 2. Shard wiring smoke (recommended before full array)

Run shard 0 + shard 14 with `--smoke 5` interactively on a GPU node to
verify both the first and last shard slice cleanly:

```bash
python analysis/trak/featurize_trials.py \
    --trial_manifest analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv \
    --out_dir /juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads_all_smoke \
    --shard_idx 0  --num_shards 15 --smoke 5
python analysis/trak/featurize_trials.py \
    --trial_manifest analysis_outputs/v6_0_olmo3_7b_pilot/trak/trial_manifest_all.csv \
    --out_dir /juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads_all_smoke \
    --shard_idx 14 --num_shards 15 --smoke 5
```

Each should write
`trial_grads_shard{00,14}_of15.{npy,idx.csv,meta.json}`. Confirm
`meta.json` has plausible `lo`/`hi` (shard 0 → [0, 295), shard 14 →
[4138, 4432)).

### 3. Full SLURM array

```bash
sbatch bash/run_trak_featurize_trials_array.sh
# or with the emitted-answer target:
USE_EMITTED=1 sbatch bash/run_trak_featurize_trials_array.sh
```

Defaults: writes to `/juice6/scr6/.../trial_grads_all/`, uses
`trial_manifest_all.csv`. Override via `TRIAL_MANIFEST`, `OUT_DIR`,
`CHECKPOINT`, `PROJ_DIM`, `PROJ_SEED`, `MAX_SEQ_LEN`, `USE_EMITTED` env
vars.

Per-shard wall-clock should land around **37 min**; total parallel
wall-clock is the same if all 15 shards start together.

### 4. Score

```bash
NUM_TRIAL_SHARDS=15 sbatch bash/run_trak_score.sh   # or equivalent
# Direct invocation:
python analysis/trak/score_trak.py \
    --trial_grads_dir /juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/trial_grads_all \
    --out_dir /juice6/scr6/nlp/interp-models/OLMo-3-7B/trak_v6_0_olmo3_7b_pilot/scores \
    --num_trial_shards 15 \
    --num_shards 15 \
    --normalize \
    --out_name attribution_scores_all_norm.npy
```

Output: `attribution_scores_all_norm.npy` shape (4432, 50000) fp32
≈ **887 MB**. Scoring is still GPU-fast (<1 min total — the bottleneck
is loading 800+ MB of grads into VRAM, not the matmul).

### 5. Downstream analysis

`analyze_attributions.py` already groups by `cue_rank`; with the
expanded manifest it'll average across ~1,500 trials per bucket instead
of 100. Use `--scores_file attribution_scores_all_norm.npy --tag all_norm`
or similar to tag outputs distinct from the pilot.

---

## Reference timings & sizes

Pilot timing (`trial_grads/trial_grads.meta.json`, 2026-05-26):
- 300 trials → 2,238 s ≈ **37.3 min** on a single H100 → **~7.46 s/trial**.

Projected expansion timing (linear scaling):
- 4,432 trials, single GPU: ~9.2 h.
- 4,432 trials, 15-way array: ~37 min wall-clock (~295 trials × 7.5 s
  per shard).

Disk:
- Trial grads: 4,432 × 4,096 × 2 bytes ≈ **36 MB** total (~2.4 MB per
  shard).
- Attribution scores: 4,432 × 50,000 × 4 bytes ≈ **887 MB**.

---

## Outstanding follow-ups

1. Refresh `v6_0_trak_attribution.md` — cue count (19 → 20), Step 2
   describes the pre-fix trial loss, Step 4 baselines from 300 trials
   not 4,432. After the expanded run lands, fold both this doc and
   `v6_0_trak_loss_identity_fix.md` back into the canonical doc.
2. Regenerate the pilot's source-dataset analysis on the expanded
   manifest and check the cue-rank chi-square (p was 0.069 on 300
   trials; with 15× more trials per bucket, the test has more power
   either to confirm "recipes are the same" or to detect a real
   between-bucket difference).
3. The pilot's `attribution_scores_norm.npy` was computed against the
   May-26 cell_summary; the expanded run will use the May-28 one. If
   you want a side-by-side comparison, either regenerate the pilot at
   the current cell_summary or compare both at the source-dataset lift
   level (more robust to per-trial drift).
