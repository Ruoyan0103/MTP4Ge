# MTP4Ge

A complete pipeline for constructing, testing, actively learning, and pruning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3).

## Pipeline Overview

The training dataset (451 configs, 134 config_types from a GAP study) is used as a **labeled candidate pool**. Active learning selects only the configurations that are actually informative for MTP — typically a fraction of the full dataset.

```
train_liquid.xyz
      │
      ▼  --mode pool
00_convert.sh ──► seed.cfg (~1 per config_type, all dimers)
              └──► candidate_pool.cfg (remaining ~300+ labeled configs)
      │
      ▼  train on seed.cfg
01_train.sh          Train MTP-16 on seed set (initial potential)
      │
      ▼  automated AL loop (no DFT pause — pool is already labeled)
03_active_learning.sh
      select_add ──► merge into train.cfg ──► retrain
      repeat until pool is exhausted or no extrapolative configs remain
      │
      ├──► 02_test_errors.sh    Static error evaluation (RMSE on test.cfg)
      │
      ├──► MD simulation        LAMMPS NVT (see config/lammps/)
      │
      └──► 04_prune.sh          NSGA-II pruning → Pareto front → pruned potential
```

> **Note:** Run `00_convert.sh split` separately on the full dataset *before* starting the AL loop to carve out a held-out `test.cfg` for unbiased evaluation.

## Environment

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| ASE | ≥ 3.22 |
| mlp binary | MLIP-3-prune fork |
| LAMMPS | compiled with MLIP plugin |

```bash
conda env create -f environment.yml
conda activate mtp4ge
```

## Quick Start

### 0. Carve out a held-out test set (do this first, once)

```bash
bash scripts/00_convert.sh split /path/to/train_liquid.xyz
```

Produces `data/train.cfg`, `data/val.cfg`, `data/test.cfg` (80/10/10 random split).
Keep `data/test.cfg` untouched for final evaluation.

### 1. Build seed + candidate pool

```bash
bash scripts/00_convert.sh pool /path/to/train_liquid.xyz
```

Produces:
- `data/seed.cfg` — ~1 config per config_type, all dimers → minimal diverse starting set
- `data/candidate_pool.cfg` — remaining labeled configs for AL selection
- `data/train.cfg` — copy of seed.cfg; the AL loop appends to this

### 2. Train on seed set

```bash
bash scripts/01_train.sh
```

Output: `results/potentials/pot.almtp`

### 3. Run automated active learning loop

```bash
bash scripts/03_active_learning.sh results/potentials/pot.almtp
```

The loop runs fully automatically: each iteration calls `mlp select_add` on `candidate_pool.cfg`, merges the selected (already-labeled) configs into `train.cfg`, retrains, and repeats. Stops when the pool is exhausted or no extrapolative configs remain.

To force interactive mode (for new, unlabeled candidates requiring DFT):

```bash
bash scripts/03_active_learning.sh results/potentials/pot.almtp --no-auto
```

### 4. Evaluate errors

```bash
bash scripts/02_test_errors.sh results/potentials/pot.almtp
```

Reports energy (meV/atom), force (eV/Å), and stress (GPa) RMSE on train/val/test.

### 5. MD testing

```bash
python src/md_test.py --pot results/potentials/pot.almtp --T 1200 --steps 50000
```

### 6. Prune

```bash
bash scripts/04_prune.sh results/potentials/pot.almtp 0
```

`0` selects row 0 of the Pareto front (highest accuracy). Inspect `results/pruning/pareto/` to choose a different point. Output: `results/potentials/pruned.almtp`.

## Configuration

| File | Purpose |
|---|---|
| `config/training.yaml` | MTP level, weights, iteration limit |
| `config/active_learning.yaml` | Grade threshold, max iterations |
| `config/pruning.json` | NSGA-II parameters (auto-filled by prune.py) |
| `config/lammps/md_nvt.in` | LAMMPS NVT input template |
| `config/lammps/mlip.ini` | MLIP-LAMMPS interface config |

## MTP Templates

Pre-built MTP architectures (levels 6–28) are in `mtp_templates/`. The default pipeline uses **MTP-16**. Edit `config/training.yaml` to switch levels.

## References

- Shapeev, A. V. *Moment Tensor Potentials: A class of systematically improvable interatomic potentials.* Multiscale Model. Simul. 14, 1153–1173 (2016).
- Podryabinkin, E. V. & Shapeev, A. V. *Active learning of linearly parametrized interatomic potentials.* Comput. Mater. Sci. 140, 171–180 (2017).
- Bochkarev, A. et al. *Efficient parametrization of the NSGA-II multi-objective optimization for pruning interatomic potentials.* (2024).
