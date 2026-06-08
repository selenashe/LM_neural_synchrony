# V6.0 TRAK — formulation correction (Gram /n + L2-normalization)

Revision applied 2026-06 on top of the pipeline described in
`v6_0_trak_attribution.md`, `v6_0_trak_loss_identity_fix.md`, and
`v6_0_trak_expansion_4432_trials.md`. The previous `_all_norm` scoring
run was effectively computing **TracIn-style cosine-similarity
attribution**, not TRAK. This doc records the diagnosis, the fix, and
the (provisional) λ choice. The fix has not yet been causally
validated — see the "NEXT TODO" section at the bottom.

---

## TL;DR

| | Pre-fix (`_legacy_norm_overn`) | Post-fix (canonical) |
|---|---|---|
| Gram formula | `G = ΦᵀΦ / n_sft + λI` (with /n) | `G = ΦᵀΦ + λI` (matches paper eq 13 and upstream `trak/score_computers.py`) |
| Φ rows | L2-normalized to unit length | un-normalized (preserves per-example gradient magnitude that `(ΦᵀΦ)⁻¹` is supposed to absorb) |
| λ | 1e-2 | **1e7 canonical**, plus 1e6 / 1e8 robustness sweep |
| Effective behavior | `G⁻¹ ≈ (1/λ)I` (Frobenius ratio 0.98) → ≈ scaled cosine similarity → **TracIn** | `G⁻¹` shrinks high-eig directions (Frobenius ratio 0.69 at λ=1e7) → **actual Mahalanobis-like whitening** |
| Output file | `attribution_scores_all_norm.npy` (now in `legacy_norm_overn/`) | `attribution_scores_all_lam{1e6,1e7,1e8}.npy` |
| Causally validated? | No | **No** — still requires the group-removal counterfactual (see NEXT TODO) |

---

## What was wrong

### The bug, in math

Paper eq (13) defines the TRAK attribution score as
`τ(z, S) = φ(z)ᵀ · (ΦᵀΦ)⁻¹ · ΦᵀQ`. The matrix `K = ΦᵀΦ` is the
training-side Gram of stacked projected gradients (rows = examples).
Regularization is added as `K + λI` for invertibility. **No `/n`.**

Our `score_trak.py` was computing `G = (ΦᵀΦ) / n_sft + λI` with
`n_sft = 50000` and `λ = 1e-2`. The extra `/n` makes `λ` a much
larger fraction of the (now-shrunk) data-term eigenvalues than the
paper intends:

```
G_ours = (ΦᵀΦ)/n + λI = (1/n) · (ΦᵀΦ + nλI)
       → equivalent to paper's K + λ'I with λ' = nλ = 500
```

So our nominal `λ=1e-2` was effectively `λ_paper = 500`, much larger
than the regularization the paper or upstream `traker` package would
apply (typically `1e-4 × max_eig` to `1e-2 × max_eig`).

### The bug, measured

Diagnostic on the actual `Φ` (50000 × 4096, L2-normalized rows, the
configuration the canonical pre-fix run used):

| Quantity | Value |
|---|---|
| Min eigenvalue of `ΦᵀΦ/n` | 7.6 × 10⁻⁵ |
| Median eigenvalue | 1.58 × 10⁻⁴ |
| Max eigenvalue | 2.79 × 10⁻² |
| `λ = 1e-2` rank in eigenspectrum | 99.88th percentile |
| Eigenvalues `> λ` | **5 / 4096** |
| Eigenvalues `> 10λ` | **0 / 4096** |
| `‖G⁻¹‖_F / ‖(1/λ)I‖_F` | **0.98** (≈ 1.00 means `G⁻¹` is identical to `(1/λ)I`) |

`G⁻¹` was 98% identical (in Frobenius norm) to `(1/λ)I`. The
whitening step that distinguishes TRAK from a naive dot-product
attribution was **inactive for 4,091 of 4,096 projection directions**.
The estimator was approximately

```
scores ≈ (1/λ) · P_trial · P_sftᵀ   (= scaled TracIn / cosine similarity)
```

### Why `--normalize` made it worse, not better

The original `--normalize` flag was added to fix a separate pilot-era
problem: high-norm SFT examples dominating the un-normalized inner
product regardless of direction. With L2-normalization on both sides,
all rows became unit vectors and the comparison was direction-only.

Under the corrected Gram, that normalization is the **wrong fix**.
The `(ΦᵀΦ)⁻¹` whitening's job *is* to absorb per-example magnitude
structure — a source with many high-norm examples gets down-weighted
by the Gram-inverse because those examples contribute disproportionately
to `ΦᵀΦ`. Pre-stripping magnitude via L2-normalization defeats the
mechanism the whitening is supposed to perform.

So "drop the /n but keep `--normalize`" would still not be TRAK — it'd
be cosine-similarity-on-cosine-normalized-grads with weak whitening.
The canonical path is **drop the /n AND drop `--normalize`**: feed the
raw projected gradients to the corrected Gram, let `(ΦᵀΦ)⁻¹` do the
work.

---

## The fix

### Code changes

`analysis/trak/score_trak.py`:

```diff
- G = (P_sft_t.T @ P_sft_t) / n_sft
+ G = P_sft_t.T @ P_sft_t                # paper: K = ΦᵀΦ + λI
  G = G + args.lam * torch.eye(D, ...)
```

- `--lam` default: `1e-2` → **`1e7`**
- `--normalize` default unchanged (off); flag retained for legacy
  comparison, documented as such
- `--out_name` default: `attribution_scores.npy` →
  `attribution_scores_all_lam1e7.npy`

`bash/run_trak_score.sh` env-var defaults updated to match.

### λ choice

With un-normalized rows, the eigenspectrum of `ΦᵀΦ` is much larger
than the normalized case (gradient norms span ~10⁰ to ~3×10⁴):

| Quantity | Value |
|---|---|
| trace(ΦᵀΦ) = sum of squared row norms | 1.38 × 10¹¹ |
| Min eigenvalue | 8.6 × 10⁵ |
| Median eigenvalue | 4.6 × 10⁶ |
| Max eigenvalue | **1.245 × 10¹⁰** |

Picked three λ values spanning a 100× sweep, rooted in the actual
eigenspectrum:

| Label | λ | as fraction of max_eig | Frobenius `‖G⁻¹‖_F / ‖(1/λ)I‖_F` | Interpretation |
|---|---|---|---|---|
| λ_low | 1 × 10⁶ | 8 × 10⁻⁵ | 0.29 | Strong whitening (top dirs kept, bulk shrunk) |
| **λ_mid (canonical default)** | **1 × 10⁷** | 8 × 10⁻⁴ | 0.69 | Moderate whitening — meaningful but partial |
| λ_high | 1 × 10⁸ | 8 × 10⁻³ | 0.91 | Light whitening — TracIn floor for sanity |

**λ=1e7 is provisional.** We do not yet have an LDS-style validation
(let alone the gold-standard group-removal counterfactual; see NEXT
TODO). λ_mid was chosen as the first regime where whitening is
genuinely substantial (Frobenius ratio ≤ 0.7) without being so
aggressive that low-eigenvalue noise directions get inflated.

**The λ sweep tests robustness only; it does not adjudicate
correctness.** A more diverse top-K under heavier whitening is not
evidence of a better λ. Whitening can correctly down-weight a
genuinely-causal source if its examples cluster tightly in gradient
space — redundancy and causal importance are not mutually exclusive.
Only the removal counterfactual licenses any causal claim.

---

## What changed in the results

Headline check: top-100 SFT sources for **c10** (highest-flip cue,
92.3% flip rate across checkpoints):

| Pipeline | Top-3 sources |
|---|---|
| **Legacy** (`_norm` + /n, TracIn-style) | Dolci Python Algorithms 23, Persona MATH 16, Persona GSM 11 |
| **λ = 1e6** (light reg, strong whitening) | FLAN 33, Aya 20, Dolci Tool Use 11 |
| **λ = 1e7** (canonical) | **FLAN 40, Dolci Tool Use 36, Wildchat 8** |
| **λ = 1e8** (heavy reg, light whitening) | Dolci Tool Use 48, FLAN 43, Precise IF 6 |

The legacy result and the corrected results are **qualitatively
different**. The legacy pipeline highlighted code/math sources
(Persona MATH, Persona GSM, Python Algorithms); the corrected
pipeline highlights instruction-following / tool-use / Q&A sources
(FLAN, Dolci Tool Use). Across the three corrected variants, **FLAN
and Dolci Tool Use are robustly in the top 3** — the headline is
λ-robust at the source level.

The full per-cue lift heatmaps for K=100/500/5000 are in
`analysis_outputs/v6_0_olmo3_7b_pilot/trak/analysis/`, with the
canonical λ=1e7 set as the default the plot scripts produce.

Other observations:
- The pre-fix "K=5000 distribution converges to corpus baseline"
  finding still holds in the post-fix scores. K=100 remains the
  sharpest-signal regime.
- The trial-side score distribution shifted: per-cue median ≈ 0 (as
  before), but the noise floor and tail magnitudes are on a
  different scale (since the Gram is now un-normalized).

---

## File renaming

- `plot_category_influence.py` → `plot_category_attribution.py`
  (the script's job is the same; the rename reflects that until
  causality is validated we should not call these "influence")
- Per-cue PNG outputs: `category_influence_*.png` → `category_attribution_*.png`
- Output scores file: `attribution_scores_all_norm.npy` →
  `attribution_scores_all_lam{1e6,1e7,1e8}.npy`
- Plot tag default: `--tag all_norm` → `--tag all_lam1e7`

The pre-fix `_legacy_norm_overn` artifacts (1 scores file, 53 PNGs,
1 meta JSON) are preserved under
`legacy_norm_overn/` subdirectories in both scratch and repo, with a
short `README.txt` explaining what they are and why they shouldn't
be interpreted as canonical TRAK output.

---

## NEXT TODO — group-removal counterfactual (blocked on compute)

**The lambda fix is a formulation correction; it is NOT a validation
of causality.** The corrected scores rank SFT examples by predicted
influence on the cued-answer behavior, but until we actually retrain
without those examples and check that flipping drops, the rankings
remain *descriptive attribution*, not *causal influence*.

The canonical TRAK validation is LDS (Leave-out Sufficient Set,
Definition 2.4 in the TRAK paper). A cheaper variant we can run when
compute frees up:

**Group-removal counterfactual** (4 re-SFTs from OLMo-3-7B Base):

| Condition | What | How |
|---|---|---|
| Baseline | Full SFT, no removal | Standard SFT recipe on the 50k subsample |
| TRAK removal | Top-K from λ=1e7 high-flip-cue ranking removed | Take c06/c10/c16/c19 + other high-flip cues; aggregate top-K SFT-example ranking; re-SFT on the 50k subsample minus that top-K |
| Length-matched random | K random examples removed (matched on token length) | Control for "removing any K examples reduces flipping" |
| BM25-matched | K BM25-matched-to-trial-prompts examples removed | Control for "removing topically-similar examples reduces flipping" |

Eval all 4 checkpoints on the same high-flip trials and check that
**TRAK-removal reduces flipping more than both controls**. If yes →
attribution is causal; if no → we're computing a sophisticated
ranking that doesn't recover causally-influential examples, and
either λ needs retuning, the projection seed needs more replicates,
or the single-checkpoint estimator is fundamentally insufficient
(and we'd need to ensemble across multiple SFT seeds).

**Until this runs, the lift heatmaps and category attribution shares
are descriptive, not causal.** All charts and docs use "attribution
score" not "influence." Any presentation of these results should
caveat that the causal claim is pending validation.

**Cost estimate**: ~24 GPU-hours per OLMo-3 SFT run × 4 conditions
≈ 96 GPU-hours. Plus eval time. Not feasible on current budget —
scheduled as a separate task when compute opens up.

---

## Followup doc-housekeeping (done in this revision)

- [analysis/trak/README.md](../analysis/trak/README.md) — updated
- [v6_0_trak_attribution.md](v6_0_trak_attribution.md) — updated Step 3, replaced
  Step 3.5 (the old `--normalize` section is now in
  `legacy_norm_overn/README.txt`), added NEXT-TODO pointer in caveats
- [v6_0_trak_loss_identity_fix.md](v6_0_trak_loss_identity_fix.md) and
  [v6_0_trak_expansion_4432_trials.md](v6_0_trak_expansion_4432_trials.md)
  still describe their respective changes accurately — both pre-dated
  the lambda fix and remain historical changelog entries.
