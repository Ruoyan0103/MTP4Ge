# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Full pipeline for constructing, actively learning, and pruning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3). Workflow: raw XYZ → CFG conversion → MTP training → LAMMPS+VASP active learning → pruning → LAMMPS MD testing.

## Current Status

- **Initial potential**: `results/potentials/pot-init.mtp` — trained on `data/train-init.cfg`, which contains solid (diamond + distorted bulk), defect (vacancy, SIA), and liquid structures.
- **Baseline tests**: completed for the initial potential; results collected in `results/tests/`.
- **Next steps**: run active learning (Step 3) to build `pot-<yyyy-mm-dd>.almtp`, then re-run the full test suite to compare against the baseline.

The iterative cycle is: train → test → active learning → retrain → test → (repeat until convergence).

## Environment

The `mlp` binary is at `/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp` (MLIP-3-prune fork). All scripts must be run from the **repository root**; all config paths are relative to the root.

```bash
conda env create -f environment.yml
conda activate mtp4ge
# or activate the existing venv directly:
source /scratch/project_2012355/Paper_3/mtp-env/bin/activate
```

## Pipeline Commands

### Step 0 — Convert XYZ to CFG
```bash
bash scripts/00_convert.sh [input.xyz]
```
Output: `data/<input_stem>.cfg`.

### Step 1 — Train initial potential
```bash
bash scripts/01_train.sh [config/training.yaml] [random_seed]
sbatch scripts/submit_train.sh          # SLURM (1 node, 1 task, 30 min)
# Output: results/potentials/pot-<yyyy-mm-dd>.mtp  (date set at training time)
```

### Step 2 — Evaluate errors
```bash
bash scripts/02_test_errors.sh [pot.mtp] [errors|efs]
# errors → mlp check_errors (RMSE);  efs → mlp calculate_efs (per-config)
# Reports in results/errors/
```

### Step 3 — Active learning loop
```bash
# Defect-targeted AL (vacancy + 4 SIA types + ±2% vol, 2×2×2 supercells)
sbatch scripts/submit_active_learning_sia.sh [pot.mtp] [--iter-per-structure N]

# Thermal AL (bulk diamond at 1500/2000/2500 K)
sbatch scripts/submit_active_learning_thermal.sh [pot.mtp] [--iter-per-temp N]
```
Each iteration: **A** LAMMPS MD (MLIP selection) → **B** `select_add` → **C/D** VASP single-point DFT → **E** retrain 80/10/10 split. AL potential saved as `pot-<yyyy-mm-dd>.almtp` (original `.mtp` never overwritten). Iteration outputs in `results/active_learning/iter_NNN/`.

The SIA and thermal AL scripts share the same output potential via `--al-pot`, so training accumulates across all structure types.

### Step 4 — Prune potential
```bash
bash scripts/04_prune.sh [pot.almtp] [pareto_row]   # pareto_row 0 = most accurate
sbatch scripts/submit_prune.sh [pot.almtp] [pareto_row]
python src/prune.py --step mask --row N              # re-apply row without re-running NSGA-II
# Output: results/potentials/pruned.almtp, Pareto CSV in results/pruning/pareto*/
```

### Quality tests
```bash
# Smoke test (pytest, requires pot.almtp)
pytest tests/test_mtp_calculator.py            # or: MTP_POT=<path> pytest tests/

# Crystal properties
python tests/energy_volume.py     --pot results/potentials/pot.almtp
python tests/elastic_constants.py --pot results/potentials/pot.almtp
# Ge reference: C11≈126 GPa, C12≈48 GPa, C44≈67 GPa, B≈74 GPa

python tests/defect_formation.py  --pot results/potentials/pot.almtp
# Ge reference: V E_f≈2.24 eV, SIA ground state is <110> dumbbell

python tests/phonon_dispersion.py  --pot results/potentials/pot.almtp
python tests/thermal_properties.py --pot results/potentials/pot.almtp

# Multi-phase E-V (8 phases: diamond, hd, hcp, fcc, bcc, beta-Sn, bc8, st12)
python tests/multiphase_ev.py --pot results/potentials/pot.almtp

# Defect dynamics
python tests/vacancy_migration.py  --pot results/potentials/pot_al.almtp
# 1NN barrier ref: 0.22 eV (DFT), 0.21 eV (GAP); 2NN: 1.77 eV / 1.71 eV

python tests/quasi_static_drag.py  --pot results/potentials/pot.almtp
# Stepwise atom displacement along [100], [110], [111], [031]

# SLURM for heavy tests (thermal)
sbatch scripts/submit_amorphous.sh
sbatch scripts/submit_liquid.sh

# All outputs → results/tests/<test_name>/
```

## Data Augmentation Workflow
To add new DFT data and retrain (used for strained diamond C11 fix):

```bash
# 1. Generate VASP input directories
python scripts/generate_strained_diamond.py     # 42 dirs: 6 Voigt modes × ±1/2/5%
python scripts/generate_sia_configs.py          # 4 SIA types, 2×2×2 supercells, perturbed

# 2. Run DFT
sbatch scripts/submit_vasp_array.sh             # array job for single-points

# 3. Collect results → XYZ
python scripts/collect_dft_results.py --indir <vasp_root> --out data/new.xyz

# 4. Augment training set and retrain
bash scripts/augment_training_set.sh data/new.xyz   # splits 80/10/10 into train/val/test CFG
sbatch scripts/submit_train.sh
```

## Architecture

### Data flow
```
<input>.xyz
  → src/convert.py              →  data/seed.cfg, candidate_pool.cfg, train.cfg
  → src/train.py                →  results/potentials/pot-<date>.mtp
  → src/active_learning.py      →  results/potentials/pot-<date>.almtp
      per-iter: results/active_learning/iter_NNN/{md.in, preselected.cfg, vasp/, labelled.cfg}
  → src/prune.py                →  results/potentials/pruned.almtp
                                    results/pruning/pareto*/pareto_final_*.csv
```

### Configuration
- `config/training.yaml` — `mlp_binary`, `mtp_template` (currently `mtp_templates/16.almtp`), `train_cfg`, `output_potential`, loss weights (`energy_weight`, `force_weight`, `stress_weight`), `iteration_limit`, `al_mode`, `weight_scaling`/`weight_scaling_forces` (atom-count scaling: use 2/1 for periodic DFT supercells, 1/0 for MD trajectories), `init_random`
- `config/active_learning.yaml` — `grade_threshold`, `max_iterations`, `temperature`, `md_steps`, `lammps_binary`, `lammps_structure` (pre-equilibrated LAMMPS data file), VASP settings (`vasp_command`, `vasp_encut`, `vasp_ediff`, `vasp_pp_path`, `vasp_setups`)
- `config/pruning.json` — NSGA-II hyperparameters; `<auto>` fields filled by `src/prune.py` at runtime
- `config/lammps/md_nvt.in` — NVT template with placeholders: `${T_RUN}`, `${RUN_STEPS}` (used by `md_test.py`); `${POT_PATH}`, `${GRADE_THRESHOLD}`, `${GRADE_BREAK}`, `${READ_DATA_LINE}` (additionally used by `active_learning.py`)

### Source modules
- `src/convert.py` — reads extended XYZ via ASE, writes MLIP-3 CFG. `SPECIES_MAP` maps element symbol → integer type index. Configs without `free_energy` are silently skipped. Energy written to CFG is `free_energy` (not `energy`).
- `src/train.py` — thin wrapper around `mlp train`; all hyperparameters from `config/training.yaml`.
- `src/active_learning.py` — full MLIP-2 tutorial-2 loop (steps A–E). CLI flags: `--pot`, `--max-iter`, `--structure` (overrides `lammps_structure` in YAML), `--al-pot` (explicit output path, prevents suffix-chaining), `--temperature` (overrides `temperature` in YAML). `_IDX_TO_SYMBOL` is the reverse of `SPECIES_MAP` and must stay in sync with `convert.py`.
- `src/prune.py` — three-step pipeline: `extract_problem` (XᵀWX/XᵀWy matrices), `prune` (NSGA-II via `mlp prune`), `mask_inherited` (apply Pareto row). Run individual steps with `--step {extract,prune,mask}`.
- `src/dump_to_cfg.py` — converts LAMMPS dump files to MLIP-3 CFG format.
- `src/md_test.py` — writes `mlip.ini`, patches the LAMMPS template, launches `lmp_mpi`.
- `src/test_errors.py` — wraps `mlp check_errors` / `mlp calculate_efs`.

### Tests (`tests/`)
- `tests/utils.py` — shared helpers: `load_structure` (CFG/XYZ/LAMMPS data → cell+positions+types; defaults to 2-atom Ge diamond), `write_bare_cfg`, `calc_efs`, `compute_rdf`, `read_lammps_dump`, `coordination_number`. Sets `LD_LIBRARY_PATH` for OpenBLAS via `_mlp_env()`.
- `tests/mtp_calculator.py` — `MTPCalculator`: ASE `Calculator` subclass wrapping `mlp calculate_efs` via subprocess. One subprocess per evaluation; intended for NEB images and small-cell relaxations.
- `tests/test_mtp_calculator.py` — pytest smoke test: `MTPCalculator` returns a float energy and near-zero forces on perfect diamond. `MTP_POT` env var overrides the default potential path.
- `tests/energy_volume.py` — E-V curve via uniform lattice scaling; Birch-Murnaghan EOS fit; PNG plot.
- `tests/elastic_constants.py` — full 6×6 stiffness tensor via central-difference finite strain; derives C11, C12, C44, B, G, E, ν.
- `tests/defect_formation.py` — formation energies for vacancy and SIA types (T, H, X, B).
- `tests/phonon_dispersion.py` — phonon dispersion along high-symmetry paths.
- `tests/thermal_properties.py` — thermal expansion coefficient αL and heat capacity via phonon DOS.
- `tests/multiphase_ev.py` — E-V curves for 8 Ge phases (diamond, hd, hcp, fcc, bcc, beta-Sn, bc8, st12); overlays DFT reference from `tests/reference/`. Structure parameters match the ilearn DFT reference exactly.
- `tests/vacancy_migration.py` — 1NN and 2NN vacancy hop barriers via climbing-image NEB; uses `MTPCalculator` for per-image EFS. `--a0` should match the MTP equilibrium lattice constant.
- `tests/quasi_static_drag.py` — stepwise atom displacement along [100], [110], [111], [031]; computes ΔE and parallel force per step; compares against DFT reference in `tests/reference/`.

### Structure generation scripts (`scripts/`)
- `generate_strained_diamond.py` — 42 VASP input dirs: 6 Voigt strain modes (xx, yy, zz, yz, xz, xy, vol) × ±1%/±2%/±5%. Required to constrain C11 and C44; isotropic volume alone is insufficient.
- `generate_sia_lammps.py` — LAMMPS data files for AL seeds: 4 SIA types + ±2% volume strains, 2×2×2 supercell (65 atoms). Auto-called by `03_active_learning_sia.sh` if files are missing.
- `generate_sia_configs.py` — VASP input dirs for direct DFT: 4 SIA types in 2×2×2 supercell, 1 reference + 20 perturbed configs each.
- `cfg_to_xyz.py` — converts CFG files back to extended XYZ.
- `split_test.py` — splits a CFG file into train/val/test subsets.

### Key conventions
- CFG blocks carry `Feature orig_index` (XYZ frame index) and `Feature type` (config_type string).
- All SIA and vacancy supercells use 2×2×2 (64-atom base). The MTP cutoff (~5 Å) is well within the 11.5 Å box, isolating defect cores from periodic images.
- `data/*.cfg` and `results/` are gitignored; only source, scripts, and config are tracked.
- MTP templates live in `mtp_templates/`; current default is `16.almtp`. `20.almtp` and `22.almtp` are available for higher accuracy at greater cost.
