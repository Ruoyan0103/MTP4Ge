# MTP4Ge

A complete pipeline for constructing, testing, actively learning, and pruning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3).

## Pipeline Overview

```
data/train_diamond.xyz
      │
      ▼  00_convert.sh
data/train.cfg (converted MLIP-3 CFG)
      │
      ▼  01_train.sh
results/potentials/pot.almtp (trained MTP-16 potential)
```

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

### Step 1 — Convert data and build seed + candidate pool

The input file `data/train_diamond.xyz` is already in the repository. Convert it to MLIP-3 CFG format and partition it into a seed set and a candidate pool:

```bash
bash scripts/save/00_convert.sh data/train_diamond.xyz
```

Produces:
- `data/seed.cfg` — all `distorted_bulk` configs (one kept as seed); dimers excluded
- `data/candidate_pool.cfg` — remaining labeled configs for AL selection
- `data/train.cfg` — copy of `seed.cfg`; the AL loop appends to this
- `logs/seed_selection.log` — index and config_type of the chosen seed config

### Step 2 — Train the initial potential

Trains MTP-16 (template: `mtp_templates/16.almtp`) on the seed set:

```bash
bash scripts/save/01_train.sh
```

Output: `results/potentials/pot.almtp`

Training hyperparameters (iteration limit, loss weights, MTP level) are configured in `config/training.yaml`.


## Configuration

| File | Purpose |
|---|---|
| `config/training.yaml` | MTP level, loss weights, iteration limit |

## MTP Templates

Pre-built MTP architectures are in `mtp_templates/`. The default pipeline uses **MTP-16** (`mtp_templates/16.almtp`). Edit `config/training.yaml` to switch levels.

## References

- Shapeev, A. V. *Moment Tensor Potentials: A class of systematically improvable interatomic potentials.* Multiscale Model. Simul. 14, 1153–1173 (2016).
- Podryabinkin, E. V. & Shapeev, A. V. *Active learning of linearly parametrized interatomic potentials.* Comput. Mater. Sci. 140, 171–180 (2017).
- Bochkarev, A. et al. *Efficient parametrization of the NSGA-II multi-objective optimization for pruning interatomic potentials.* (2024).
