# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a full pipeline for constructing, actively learning, and pruning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3). The workflow progresses from raw XYZ data → seed/pool split → MTP training → active learning refinement → pruning → LAMMPS MD testing.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mtp4ge
```

The `mlp` binary path is hardcoded in `config/training.yaml` (`mlp_binary`). On this cluster it points to the MLIP-3-prune fork at `/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp`. The environment also uses a separate venv at `/scratch/project_2012355/Paper_3/mtp-env/bin/`.

## Pipeline Commands

All scripts must be run from the **repository root**.

### Step 0 — Convert data and build seed + candidate pool
```bash
bash scripts/00_convert.sh [input.xyz]
# Default input: /scratch/project_2012355/Paper_3/train_dataset/train_liquid.xyz
```
Produces: `data/seed.cfg`, `data/candidate_pool.cfg`, `data/train.cfg`, `logs/seed_selection.log`

### Step 1 — Train initial potential
```bash
bash scripts/01_train.sh [config/training.yaml] [random_seed]
# Or submit to SLURM:
sbatch scripts/submit_train.sh
```
Output: `results/potentials/pot.almtp`

### Step 2 — Evaluate errors
```bash
bash scripts/02_test_errors.sh [pot.almtp] [errors|efs]
```
Runs `mlp check_errors` (RMSE) or `mlp calculate_efs` (per-config predictions) on all available train/val/test splits. Reports in `results/errors/`.

### Step 3 — Active learning loop
```bash
bash scripts/03_active_learning.sh [pot.almtp] [--no-auto] [--max-iter N]
```
Auto mode (default): iterates `select_add → merge → retrain` using the pre-labeled candidate pool until convergence or `max_iterations`. At the end the remaining pool is 50/50 split into `data/val.cfg` and `data/test.cfg`.

Interactive mode (`--no-auto`): pauses each iteration for DFT labeling before retraining.

### Step 4 — Prune potential
```bash
bash scripts/04_prune.sh [pot.almtp] [pareto_row]
# pareto_row: 0 = most accurate, higher = sparser
# To re-apply a different row without re-running NSGA-II:
python src/prune.py --step mask --row N
```
Output: `results/potentials/pruned.almtp`, Pareto front CSV in `results/pruning/pareto*/`.

### MD testing
```bash
python src/md_test.py --pot results/potentials/pot.almtp --T 1200 --steps 50000 --outdir results/md/run01
```

## Architecture

### Data flow
```
data/train_diamond.xyz (or train_liquid.xyz)
  → src/convert.py --mode pool
      data/seed.cfg        (1 distorted_bulk config)
      data/candidate_pool.cfg  (all others, excl. dimers)
      data/train.cfg       (copy of seed; AL appends here)
  → src/train.py          →  results/potentials/pot.almtp
  → src/active_learning.py →  iteratively expands train.cfg, retrains pot.almtp
                              → data/val.cfg, data/test.cfg (at end)
  → src/prune.py          →  results/potentials/pruned.almtp
```

### Configuration files
- `config/training.yaml` — `mlp_binary` path, MTP template, loss weights (`energy_weight`, `force_weight`, `stress_weight`), `iteration_limit`, `al_mode`
- `config/active_learning.yaml` — `grade_threshold`, `max_iterations`, `preselected_cfg` path, `auto_labeled`, seed set parameters (`seed_per_type`, `seed_types`, `always_include_types`, `exclude_types`)
- `config/pruning.json` — NSGA-II hyperparameters (`pop_size`, `n_gen`, `time`); `<auto>` fields are filled by `src/prune.py` at runtime
- `config/lammps/md_nvt.in` — LAMMPS input template; `${T_RUN}` and `${RUN_STEPS}` are substituted by `src/md_test.py`

### Source modules (`src/`)
- `convert.py` — reads extended XYZ via ASE, writes MLIP-3 CFG format. `SPECIES_MAP` maps element symbol → integer type index (extend for multi-species). Virial stress is read from `atoms.info` keys or computed from `atoms.get_stress()`. Three modes: `split` (random train/val/test), `no-split` (single file), `pool` (stratified seed + candidate pool for AL).
- `train.py` — thin wrapper calling `mlp train`; all hyperparameters come from `config/training.yaml`.
- `active_learning.py` — drives the AL loop: calls `mlp select_add`, merges selected configs into `train.cfg`, removes them from the pool, retrains. Logs per-iteration selections to `logs/al_selection.log`. After convergence splits remaining pool → `val.cfg`/`test.cfg`.
- `prune.py` — three-step pruning pipeline: `extract_problem` (compute XᵀWX/XᵀWy matrices), `prune` (NSGA-II via `mlp prune`), `mask_inherited` (apply chosen Pareto row). Can run individual steps with `--step`.
- `test_errors.py` — wraps `mlp check_errors` and `mlp calculate_efs`.
- `md_test.py` — writes a run-specific `mlip.ini`, patches the LAMMPS template, launches `lmp_mpi`.

### Key conventions
- CFG files use `Feature orig_index` (original XYZ index) and `Feature type` (config_type) per block — these are parsed by `active_learning.py` for pool management and logging.
- Energy label used is `free_energy` from calculator results (not `energy`); configs without it are silently skipped during conversion.
- All scripts use `set -euo pipefail` and change to the repo root before running Python, so all paths in YAML/JSON configs are relative to the repo root.
- `data/*.cfg` and `results/` are gitignored; only `data/train_diamond.xyz` and `data/.gitkeep` are tracked.
