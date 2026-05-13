# MTP4Ge

A complete pipeline for constructing, testing, actively learning, and pruning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3).

## Pipeline Overview

```
train_liquid.xyz
      │
      ▼
00_convert.sh        XYZ → MLIP-3 CFG (train / val / test split)
      │
      ▼
01_train.sh          Train MTP-16 potential
      │
      ├──► 02_test_errors.sh    Static error evaluation (RMSE)
      │
      ├──► MD simulation        LAMMPS NVT (see config/lammps/)
      │
      ├──► 03_active_learning.sh  Grade → select → relabel → retrain loop
      │
      └──► 04_prune.sh          NSGA-II pruning → Pareto front → pruned potential
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

### 1. Convert training data

```bash
bash scripts/00_convert.sh /path/to/train_liquid.xyz data/ "0.8 0.1 0.1"
```

Produces `data/train.cfg`, `data/val.cfg`, `data/test.cfg`.

### 2. Train potential

```bash
bash scripts/01_train.sh
```

Output: `results/potentials/pot.almtp`

### 3. Evaluate errors

```bash
bash scripts/02_test_errors.sh results/potentials/pot.almtp errors
```

Reports energy (meV/atom), force (eV/Å), and stress (GPa) RMSE for each split.

### 4. Active learning

Generate candidate structures (e.g. from a short MD run), write them to `data/preselected.cfg`, then:

```bash
bash scripts/03_active_learning.sh results/potentials/pot.almtp
```

Each iteration pauses to let you add DFT labels before retraining.

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
