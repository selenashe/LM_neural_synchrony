v6.0 filtered-trial tier-1 analysis
===================================

Input:  analysis_outputs/v6_0_olmo3_7b_pilot/cell_summary.csv
        (16680 rows; 360 unique questions; 19 unique cues;
         balanced across 4 checkpoints)

Outcomes:
  flip       = 1 - t4_accuracy  (binary)
  shift      = prob_correct_shift_mean  (continuous, t4 - t2)
  shift_norm = bound-normalized shift, in [-1, 1]:
                 shift / (1 - t2_prob)   if shift >= 0
                 shift / t2_prob         if shift < 0
               Removes the mechanical ceiling/floor artifact: a
               high-confidence trial can't move up much; a
               low-confidence trial can't move down much. Pair
               with `shift` rather than replacing it.
  confidence = t2_prob_correct_mean (raw correct-letter prob;
               NOT normalized over A/B/C/D — see goal-6 caveats)

Files in this directory:
  coverage_summary.txt              — overall coverage stats
  coverage_per_question.csv         — cue count per (q, ckpt)
  coverage_by_domain.csv            — balanced-subset bias check
  balanced_subset_question_ids.csv  — ≥18-cue questions
  coverage_heatmap.png

  variance_decomposition.csv/.png   — η² per factor per ckpt per outcome

  per_question_profile.csv          — (ckpt × q) summary
  per_question_resistance_ranking.png

  per_cue_profile.csv               — (ckpt × c) summary
  per_cue_effectiveness_ranking.png

  rank_stability.csv/.png           — Spearman ρ across ckpts

  confidence_vs_outcome.csv/.png    — trial + per-question correlations

  cue_clusters.csv / cue_clusters_shift_norm.csv
  cue_clusters_dendrogram.png / cue_clusters_dendrogram_shift_norm.png
  cue_by_checkpoint_heatmap.png / cue_by_checkpoint_heatmap_shift_norm.png
                                    — empirical cue groups
                                      (raw + bound-normalized)

  cue_strength_validation.csv/.png  — does hand-coded strength match
                                      observed effectiveness?

  q_by_c_interaction_residuals_shift.png
  q_by_c_interaction_residuals_shift_norm.png
                                    — residuals from additive model
                                      (raw + bound-normalized)
  q_by_c_top_residuals.csv          — biggest deviations (replicate?)

Caveats:
  * Each (q, c, ckpt) cell has count=1. Interaction terms are
    unidentified from residual variance — Q×C is descriptive only.
  * `confidence` is the raw probability assigned to the correct
    letter token, not normalized over answer options. Calibration
    may differ across checkpoints; trust Spearman ρ over Pearson r
    when comparing across stages.
  * Variance-decomposition factors are estimated independently and
    overlap (questions and domains are nested; cue and cue_strength
    also overlap). Treat η² as "how much variation aligns with this
    factor," not as orthogonal partitions.
