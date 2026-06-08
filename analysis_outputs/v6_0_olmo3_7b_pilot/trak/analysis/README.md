# TRAK attribution plots — layout

5 subfolders organized by analysis type. Each is populated by a
specific plot script in `analysis/trak/`. Filenames within each
subfolder encode the run parameters (`K`, λ, field, behavior class).

| Subfolder | Files | Produced by | What it shows |
|---|---|---|---|
| [`score_distribution/`](score_distribution/) | 1 PNG | `plot_attribution_distribution.py` | Per-cue attribution score distribution: sorted curves + pooled histogram. The diagnostic that tells you **which K to interpret seriously** (K=100 is ~3σ-selected, K=5000 is essentially the bulk). |
| [`category_attribution_per_cue/`](category_attribution_per_cue/) | 23 PNGs (20 per-cue + 3 bucket-aggregate) | `plot_category_attribution.py` (default `--group_by cue_id` for per-cue, `--group_by cue_rank` for low/med/high bucket aggregates) | Per-trial `Σ(attr in source) / Σ(attr all)` for each `source_dataset`, averaged across trials in that cue (or bucket), split flipped vs not-flipped. Per-trial signed attribution share, NOT top-K filtered. The 3 bucket-aggregate files (`_low_`, `_med_`, `_high_`) average over the same low/med/high cue thirds from `cue_ranking_full` that the lift heatmaps use; bucket cue membership is shown in the suptitle. |
| [`top_k_composition_canonical/`](top_k_composition_canonical/) | 6 PNGs (3 Ks × 1 field × 2 metrics) | `plot_top_source.py` at canonical λ=1e7 | Per-cue (20 cols) stacked bars (counts of top-K examples by `source_dataset`) + lift heatmaps (counts normalized by subsample baseline share). The canonical headline. |
| [`top_k_composition_lambda_robustness/`](top_k_composition_lambda_robustness/) | 6 PNGs (λ ∈ {1e6, 1e8} × 3 Ks × 1 field) | `plot_top_source.py --tag all_lam{1e6,1e8}` | Source-dataset lift heatmaps at lighter/heavier regularization. Stability check on the canonical λ=1e7 source composition. |
| [`top_k_composition_by_flip_outcome/`](top_k_composition_by_flip_outcome/) | 9 paired PNGs (3 λs × 3 Ks × 1 field) | `plot_top_source.py --flip_split --group_by cue_rank --metric lift` | **Bucketed** (low/med/high thirds of the 20-cue effectiveness ranking) source-dataset lift heatmaps. Each PNG is a **paired figure**: flipped trials on the left panel, not-flipped on the right, with shared color scale and shared y-axis (sources). Bucketing keeps each (bucket × behavior) cell at ≥190 trials so per-cell estimates are stable. Bucket cue membership + per-cell n_trials printed in chart titles / x-axis labels. |
| [`legacy_norm_overn/`](legacy_norm_overn/) | 54 files (53 PNGs + README) | Pre-2026-06 `plot_*.py` runs | Outputs of the legacy `--normalize` + `G = ΦᵀΦ/n + λI` pipeline. Per `method_documentation/v6_0_trak_lambda_fix.md`, this is now known to be ≈ TracIn-style cosine similarity rather than TRAK; kept for back-comparison only. |

## Filename conventions

`top{K}_source_{metric}{behavior_suffix}_by_{group_by}{tag}.png` where:
- `K` ∈ {100, 500, 5000} — top-K cutoff
- `metric` ∈ {stacked (count), lift_heatmap}
- `behavior_suffix` ∈ {`""`, `_by_flip`} — `_by_flip` for the paired flipped-vs-not figures from `--flip_split --metric lift`. (When `--flip_split` is paired with `--metric count`, the legacy per-behavior split — `_flipped` / `_notflipped` — is still produced as separate stacked-bar PNGs.)
- `by_{group_by}` only appears when `group_by` ≠ `cue_rank` (e.g., `_by_cue_id`); for cue_rank-bucketed plots it's omitted
- `tag` ∈ {`_all_lam1e7`, `_all_lam1e6`, `_all_lam1e8`} — λ identifier

Examples:
- `top100_source_stacked_by_cue_id_all_lam1e7.png` — canonical λ, per-cue (20 bars), top-100 source-dataset stacked bar
- `top100_source_lift_heatmap_by_flip_all_lam1e7.png` — canonical λ, paired (flipped | not-flipped) heatmap with low/med/high buckets on each panel's x-axis

## Caveats

- All numbers labeled "attribution share" or "attribution score" are
  **descriptive**, not causal, until the group-removal counterfactual
  in `v6_0_trak_lambda_fix.md` §"NEXT TODO" validates the rankings.
- λ-robustness sweep tests **stability** of the source/domain
  composition under regularization choice. Stability ≠ causal
  correctness — see `v6_0_trak_lambda_fix.md` §"What changed in the
  results" for the rationale.
- For the flip-split plots: cell `n_trials` varies by bucket × behavior
  (range ~190 to ~1300 per cell on the current run; printed in the
  per-cue script output but not on the chart itself). Cells with the
  smallest n (typically `high` bucket × not-flipped, n≈192) are the
  noisiest.
