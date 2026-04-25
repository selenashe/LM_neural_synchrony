# Neural Synchrony Between Socially Interacting Language Models

> Code for **"Neural Synchrony Between Socially Interacting Language Models"**.
> Neural synchrony, measured by predictive performance between neural representations of interacting LMs, correlates strongly with collective social performance.

<p align="center">
  <img src="visuals/teaser.png" alt="Neural Synchrony Correlation" width="1000"/>
</p>

---

## Repository Structure

```
LM_neural_synchrony/
├── experiment_config.py          # Global config (seeds, episodes, result paths)
├── model_paths.json              # Model short name -> local checkpoint path
│
├── core/                         # Shared infrastructure
│   ├── model_loader.py           #   Unified loading for 9 model architectures
│   ├── LM_hf.py                  #   nnsight-based inference + hidden state extraction
│   ├── prompt_formatting.py      #   Chat template formatting per model
│   └── utils.py                  #   Shared utilities (seeds, device, state loading)
│
├── simulation/                   # Stage A: dialog generation
│   ├── sample_normal_agent.py    #   Entry point: run episodes, save CSVs + .npy states
│   ├── sot_env.py                #   Sotopia environment: turn-taking, goal tracking
│   ├── sample_gemini_agent.py    #   Gemini/Vertex variant (no hidden states)
│   └── gemini_sot_env.py         #   Sotopia env backed by Vertex AI API
│
├── analysis/                     # Stage B+C: synchrony measurement + logit lens
│   ├── affine_transformation.py  #   Train ridge regression A->B per layer pair, R^2
│   ├── analyze_alignment.py      #   Correlate per-episode R^2 with alignment labels
│   ├── analyze_controls_sbert.py #   SBERT text-only baseline control
│   ├── analyze_svd.py            #   SVD analysis of hidden representations
│   ├── logit_lens_false_belief.py#   Layerwise logit lens for false-belief scenarios
│   ├── logit_lens_mistral.py     #   Original logit lens (Mistral, Sotopia)
│   ├── aggregate_logit_lens.py   #   Aggregate logit lens results across episodes
│   └── summarize_false_belief_results.py
│
├── labeling/                     # Evaluation + label generation
│   ├── goal_alignment_labels.py  #   Classify goal alignment via Gemini -> JSON
│   ├── build_labels_v2.py        #   Add task_structure + outcome to labels
│   ├── evaluate_with_oss.py      #   Social performance eval with gpt-oss-120b
│   ├── annotate_goal_tracking_task_success.py
│   ├── analyze_prompt_goal_tracking.py
│   ├── analyze_prompt_goal_tracking_gemini.py
│   ├── analyze_transcript_domain_metrics.py
│   ├── fix_parse_errors.py
│   ├── check_dond_pareto.py
│   └── get_control_conditions.py
│
├── visualization/                # Plotting scripts
│   └── visualize_analyses_mia_ava_90ep.py
│
├── playground/                   # Interactive analysis + notebooks
│   ├── data_analysis.py          #   Master analysis/plotting script
│   ├── visualize_analyses.ipynb
│   ├── visualize_analyses_mia_ava_90ep.ipynb
│   ├── layer_r2_heatmaps_pilot.ipynb
│   └── sbert_affine_r2_bars.ipynb
│
├── bash/                         # SLURM scripts (active)
│   ├── run_simulate_and_save_states.sh
│   ├── run_logit_lens_false_belief.sh
│   ├── run_affine_and_controls_sbert_false_belief.sh
│   ├── run_affine_and_controls_sbert_mia_ava.sh
│   ├── run_alignment_all_layers.sh
│   ├── run_controls_sbert.sh
│   ├── run_svd_all_layers.sh
│   ├── train_affine_remaining_pairs_slurm.sh
│   └── without_genuine_interaction.sh
│
├── scripts/                      # Utilities
│   └── list_false_belief_pairs.py
│
├── sotopia_utils/                # Vendored Sotopia data (envs, agent combos)
├── archive/                      # Deprecated scripts + old data, kept for reference
├── logs/                         # SLURM logs
└── analysis_files/               # Precomputed R^2, control JSONs, labels
```

**Output directories** (generated, not checked in):
- `sotopia_results/` -- 90-episode Mia/Ava simulation outputs
- `sotopia_results_false_belief/` -- 18-episode false-belief outputs (original 3 scenarios)
- `sotopia_results_false_belief_100/` -- 600-episode false-belief outputs (100 scenarios)
- `sotopia_results_gemini/` -- Gemini API runs
- `affine_transformation/` -- Trained affine models + R^2 JSON
- `logit_lens_results_false_belief/` -- Logit lens heatmaps + JSONs (original 3 scenarios)
- `logit_lens_results_false_belief_100/` -- Logit lens heatmaps + JSONs (100 scenarios)
- `summary_plots_false_belief/` -- Aggregated plots (original 3 scenarios)
- `summary_plots_false_belief_100/` -- Aggregated plots (100 scenarios)

---

## Pipelines

### 1. Dialog Simulation (Stage A)

Generate turn-by-turn dialogues between LM pairs and save hidden states at every layer.

```bash
sbatch bash/run_simulate_and_save_states.sh
```

Runs 36 model pairs (9 architectures, choose-2 + self-play). Outputs go to `sotopia_results_false_belief_100/`.

### 2. Neural Synchrony / Affine R^2 (Stage B)

Train affine maps between agent hidden states, measure R^2 as synchrony metric.

```bash
sbatch bash/run_affine_and_controls_sbert_false_belief.sh
```

This runs affine transformation training across layer pairs + SBERT text-only control baselines.

### 3. Logit Lens (Stage C)

Layer-wise probing: apply unembedding at each layer to track belief/intent formation.

```bash
sbatch bash/run_logit_lens_false_belief.sh
```

38400 array tasks (64 pairs x 600 episodes), submitted in batches due to SLURM array limits. Results aggregated with `analysis/aggregate_logit_lens.py`.

### 4. Social Performance Evaluation

> **Prerequisite:** Launch vLLM serving gpt-oss-120b locally ([setup guide](https://cookbook.openai.com/articles/gpt-oss/run-vllm)).

```bash
python labeling/evaluate_with_oss.py
```

### 5. Analysis and Visualization

```bash
python playground/data_analysis.py
```

Or use the Jupyter notebooks in `playground/` for interactive exploration.

---

## Requirements

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure model paths in `model_paths.json` (maps short names to local checkpoint directories).

### Supported Models

OLMo-3-7B-Instruct, Gemma-7B-IT, Llama-3-8B-Instruct, Mistral-7B (v0.2, v0.3), Ministral-3-14B (Instruct, Reasoning), Qwen2.5-7B-Instruct, Qwen3-14B.
