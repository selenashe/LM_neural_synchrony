# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for "Neural Synchrony Between Socially Interacting Language Models." Measures neural synchrony (via predictive performance between neural representations) of interacting LLMs and correlates it with social performance.

**Core workflow:**
1. Run social simulations between LLM pairs (Sotopia framework)
2. Extract neural representations during interactions
3. Train affine transformations between representations
4. Analyze synchrony vs social performance correlation

## Configuration

### Critical Configuration Files

- **`model_paths.json`**: Maps model names to local HuggingFace model paths. Must be configured before running simulations.
- **`experiment_config.py`**: Centralized settings for episode counts, seeds, data paths. Respects environment variables for run-specific overrides.

### Environment Variables

Set these to customize data paths and episode selection:

- `SOTOPIA_ENV_AGENT_COMBOS_BASENAME`: Episode list JSON (default: `env_agent_combos_fixed_two_agents.json`)
- `SOTOPIA_ENVS_BASENAME`: Environment definitions JSON (default: `envs.json`)
- `SOTOPIA_RESULTS_DIR`: Output directory name (default: `sotopia_results`)
- `SOTOPIA_RUN_LABEL`: Optional suffix to prevent output overwrites
- `LM_DEVICE`: Force device selection (`cuda`, `mps`, or `cpu`)

**Example for false-belief experiments:**
```bash
export SOTOPIA_ENV_AGENT_COMBOS_BASENAME=env_agent_combos_false_belief_fixed_two_agents.json
export SOTOPIA_ENVS_BASENAME=envs_false_belief.json
export SOTOPIA_RESULTS_DIR=sotopia_results_false_belief
```

## Core Workflows

### 1. Run Social Simulations

**Local execution (one model pair):**
```bash
python sample_normal_agent.py --model_1 "Mistral-7B-Instruct-v0.3" --model_2 "Mistral-7B-Instruct-v0.2"
```

**Batch execution (multiple pairs):**
```bash
bash bash/simulate_and_save_states.sh
```

**SLURM cluster (array jobs):**
```bash
sbatch bash/run_simulate_and_save_states.sh
```

Results saved to `{SOTOPIA_RESULTS_DIR}/{model1}_{model2}/`. Each episode generates:
- Conversation transcripts
- Neural representations (saved states)
- Social performance metrics

### 2. Train Affine Transformations

Train linear projections between model representations to measure synchrony:

```bash
# Single model pair
python affine_transformation.py --model "Mistral-7B-Instruct-v0.3_None_0_Mistral-7B-Instruct-v0.2_None_0" --setting A_forward

# Batch
bash bash/train_affine_transformation.sh
```

**Settings:**
- `A_forward`: Agent 1 → Agent 2 prediction
- `B_forward`: Agent 2 → Agent 1 prediction

Results in `./affine_transformation/` include R² scores (synchrony measure).

### 3. Analyze Results

**Jupyter notebooks in `playground/`:**
- `visualize_analyses.ipynb`: Main analysis for 450-episode runs
- `visualize_analyses_mia_ava_90ep.ipynb`: Analysis for 90-episode fixed-agent runs
- `data_analysis.py`: Programmatic analysis and figure generation

**Control analyses:**
```bash
bash bash/without_genuine_interaction.sh  # Non-interactive controls
python analyze_controls.py                # Analyze control conditions
python analyze_controls_sbert.py          # Sentence-BERT based controls
```

## Architecture

### Key Modules

**`LM_hf.py`**: Language model wrapper using nnsight
- `BaseLM`: Base class with model initialization, tokenization, prompt formatting
- `LM_nnsight`: Extended class for neural state extraction during generation
- Supports Mistral and Llama instruction-tuned models

**`sot_env.py`** / **`gemini_sot_env.py`**: Sotopia environment implementations
- `SotopiaEnv`: Orchestrates two-agent dialogues
- Manages turns, hidden states extraction, goal tracking
- Supports multiple test modes (baseline, ablations, control conditions)

**`affine_transformation.py`**: Neural synchrony measurement
- `LinearProjection`: Single-layer affine transformation (W·x + b)
- Train/test split per episode or stratified by goal condition
- Outputs R² scores as synchrony metric

**`sotopia_utils/`**: Sotopia framework adaptation
- `utils.py`: Message schemas, agent backgrounds, evaluation dimensions
- `sotopia_data/`: Episode configurations and environment definitions

**Analysis scripts:**
- `analyze_alignment.py`: Layer-wise alignment analysis
- `analyze_svd.py`: SVD-based dimensionality analysis
- `logit_lens_*.py`: Token prediction analysis at intermediate layers
- `visualize_analyses_*.py`: Generate correlation plots and figures

### Data Flow

1. **Episode data** (`sotopia_data/`): JSON configs defining scenarios, agents, goals
2. **Simulation** (`sample_normal_agent.py`): Run episodes, save to `sotopia_results/{model_pair}/`
3. **State files**: Pickled tensors of neural representations per turn
4. **Affine training**: Load states, train projections, save to `affine_transformation/`
5. **Analysis**: Aggregate R² scores, correlate with social metrics, generate plots in `playground/viz_outputs*/`

### Episode Configuration

- **Standard runs**: 90 episodes (Mia + Ava fixed agents) or 450 episodes (multiple agent pairs)
- **Episode files**: `env_agent_combos_*.json` in `sotopia_utils/sotopia_data/`
- **False-belief runs**: Use `envs_false_belief.json` with `goal_condition` metadata for stratification

## Directory Structure

- **`bash/`**: Shell scripts for batch processing and SLURM jobs
- **`sotopia_utils/`**: Sotopia framework code and episode data
- **`sotopia_results*/`**: Simulation outputs (transcripts, states, scores)
- **`playground/`**: Jupyter notebooks and analysis scripts
- **`affine_transformation/`**: Trained projection matrices and R² scores
- **`analysis_outputs/`**: Cached analysis results
- **`logs/`**: SLURM job logs

## Common Patterns

### Adding a New Model

1. Download model to cluster: `{BASE_PATH}/{MODEL_NAME}`
2. Add entry to `model_paths.json`:
   ```json
   "MODEL_NAME": "/path/to/model"
   ```
3. Update prompt formatting in `LM_hf.py` → `get_inst()` if needed
4. Add to model lists in bash scripts (e.g., `bash/simulate_and_save_states.sh`)

### Running a New Episode Set

1. Create episode config JSON in `sotopia_utils/sotopia_data/`
2. Set `SOTOPIA_ENV_AGENT_COMBOS_BASENAME` to new file
3. Update `SOTOPIA_RESULTS_DIR` to avoid overwriting prior runs
4. Run simulations → train affine → analyze

### Analyzing a Subset of Episodes

For false-belief experiments, episodes are stratified by `goal_condition` in `envs_false_belief.json`. The affine training automatically detects this metadata and creates per-condition train/test splits (see `_build_goal_episode_map()` in `affine_transformation.py`).

## Device Selection

Models run on CUDA by default. Override via `LM_DEVICE`:
```bash
export LM_DEVICE=mps  # For Apple Silicon
export LM_DEVICE=cpu  # CPU-only
```

Falls back to: cuda > mps > cpu if not set.

## Notes

- **No formal test suite**: Validate changes by running end-to-end on a small episode subset
- **SLURM dependencies**: Cluster scripts assume SLURM array jobs; adapt `#SBATCH` directives for your cluster
- **GPU requirements**: Most scripts need 1 GPU with 40-80GB VRAM for 7B-8B models
- **Gemini experiments**: Use `gemini_sot_env.py` and `sample_gemini_agent.py` for API-based models (requires API credentials)
