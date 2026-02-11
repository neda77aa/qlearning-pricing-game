# Algorithmic Collusion Replication (Reference Dependence)

This repository contains simulation code for reference-dependent demand and Q-learning pricing dynamics.

## Project Structure

- `main.py`: clean experiment launcher with CLI options
- `input/init.py`: core model/environment and economic primitives
- `input/qlearning.py`: firm learning loop, convergence checks, and cycle detection
- `input/qlearning_reference.py`: consumer reference-price Q-learning agent
- `input/ConvResults*.py`: experiment runners and result saving logic
- `input/visualization.py`: plotting utilities for saved experiment results
- `simulation.py`: standalone comparative plotting script

## Running Experiments

Use `main.py` with CLI flags (no manual editing required).

### Basic usage

```bash
python main.py --experiment gamma_only
```

### Experiment choices

- `trial_test`
- `alpha_beta`
- `gamma_lambda`
- `loss_aversion`
- `gamma_only`
- `mu_only`
- `misspecification`

For misspecification mode, choose sub-test:

```bash
python main.py --experiment misspecification --misspecification-test gamma_lambda
python main.py --experiment misspecification --misspecification-test alpha_beta
```

### Common reference toggle (default: true)

```bash
python main.py --experiment gamma_only --common-reference true
python main.py --experiment gamma_only --common-reference false
```

### Choose output root at runtime

```bash
python main.py --experiment gamma_only --output-root "/Users/neda/Documents/MyCleanResults"
```

You can combine options:

```bash
python main.py --experiment gamma_lambda --common-reference false --output-root "/Users/neda/Documents/MyCleanResults"
```

## Outputs

Experiments write outputs under:

- `<output-root>/<experiment_subfolder>/...`

Typical outputs include:

- `config.csv`
- `session_summaries.csv`
- `cycle_statistics.csv`
- `session_details.npz`
- heatmap figures under `Figures/`

## Current defaults in `main.py`

- `num_sessions = 200`
- all `np.linspace(..., 30)` grids use 30 points
- `common_reference = true` unless you pass `--common-reference false`

## Requirements

Install dependencies from:

```bash
pip install -r requirements.txt
```
