LEGACY artifacts from the 2026-06 pre-correction scoring run.

These scores were computed with:
  G = (Φᵀ Φ) / n_sft + λ I        ← non-canonical: paper uses no /n
  Φ rows L2-normalized to unit length   ← strips per-example gradient magnitude
  λ = 1e-2

The 2026-06 lambda-fix diagnostic confirmed that under this setup
G⁻¹ ≈ (1/λ) I (Frobenius ratio = 0.98), so the whitening was essentially
inactive. The output is closer to TracIn-style cosine-similarity attribution
than to TRAK.

The corrected scores live in ../attribution_scores_all_lam{1e6,1e7,1e8}.npy
(un-normalized rows, K = ΦᵀΦ + λI). Working default: λ=1e7.

These legacy files are kept for back-comparison only — do not interpret as
canonical TRAK attribution. See method_documentation/v6_0_trak_lambda_fix.md
for the full rationale.
