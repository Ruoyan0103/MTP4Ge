# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Full pipeline for constructing and actively learning a **Moment Tensor Potential (MTP)** for Germanium using [MLIP-3](https://gitlab.com/ashapeev/mlip-3). Workflow: raw XYZ → CFG conversion → MTP training → LAMMPS+VASP active learning → LAMMPS MD testing.

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
bash scripts/save/00_convert.sh [input.xyz]
```
Output: `data/<input_stem>.cfg`.

### Step 1 — Train initial potential
```bash
bash scripts/save/01_train.sh [config/training.yaml] [random_seed]
sbatch scripts/submit_train.sh          # SLURM (1 node, 1 task, 30 min)
# Output: results/potentials/pot-<yyyy-mm-dd>.mtp  (date set at training time)
```

### Step 2 — Evaluate errors
```bash
bash scripts/save/02_test_errors.sh [pot.mtp] [errors|efs]
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

### Quality tests
```bash
# Smoke test (pytest, requires pot.almtp)
pytest src/utils/test_mtp_calculator.py            # or: MTP_POT=<path> pytest src/utils/

# Crystal properties
python src/physical_validation/energy_volume.py     --pot results/potentials/pot.almtp
python src/physical_validation/elastic_constants.py --pot results/potentials/pot.almtp
# Ge reference: C11≈126 GPa, C12≈48 GPa, C44≈67 GPa, B≈74 GPa

python src/physical_validation/defect_formation.py  --pot results/potentials/pot.almtp
# Ge reference: V E_f≈2.24 eV, SIA ground state is <110> dumbbell

python src/physical_validation/phonon_dispersion.py  --pot results/potentials/pot.almtp
python src/physical_validation/thermal_properties.py --pot results/potentials/pot.almtp

# Multi-phase E-V (8 phases: diamond, hd, hcp, fcc, bcc, beta-Sn, bc8, st12)
python src/physical_validation/multiphase_ev.py --pot results/potentials/pot.almtp

# Defect dynamics
python src/physical_validation/vacancy_migration.py  --pot results/potentials/pot_al.almtp
# 1NN barrier ref: 0.22 eV (DFT), 0.21 eV (GAP); 2NN: 1.77 eV / 1.71 eV

python src/physical_validation/quasi_static_drag.py  --pot results/potentials/pot.almtp
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
python scripts/save/generate_strained_diamond.py     # 42 dirs: 6 Voigt modes × ±1/2/5%
python scripts/save/generate_sia_configs.py          # 4 SIA types, 2×2×2 supercells, perturbed

# 2. Run DFT
sbatch scripts/submit_vasp_array.sh             # array job for single-points

# 3. Collect results → XYZ
python scripts/save/collect_dft_results.py --indir <vasp_root> --out data/new.xyz

# 4. Augment training set and retrain
bash scripts/save/augment_training_set.sh data/new.xyz   # splits 80/10/10 into train/val/test CFG
sbatch scripts/submit_train.sh
```

## Architecture

### Data flow
```
<input>.xyz
  → src/utils/convert.py        →  data/seed.cfg, candidate_pool.cfg, train.cfg
  → src/train.py                →  results/potentials/pot-<date>.mtp
  → src/active_learning.py      →  results/potentials/pot-<date>.almtp
      per-iter: results/active_learning/iter_NNN/{md.in, preselected.cfg, vasp/, labelled.cfg}
```

### Configuration
- `config/training.yaml` — `mlp_binary`, `mtp_template` (currently `mtp_templates/16.almtp`), `train_cfg`, `output_potential`, loss weights (`energy_weight`, `force_weight`, `stress_weight`), `iteration_limit`, `al_mode`, `weight_scaling`/`weight_scaling_forces` (atom-count scaling: use 2/1 for periodic DFT supercells, 1/0 for MD trajectories), `init_random`
- `config/active_learning.yaml` — `grade_threshold`, `max_iterations`, `temperature`, `md_steps`, `lammps_binary`, `lammps_structure` (pre-equilibrated LAMMPS data file), VASP settings (`vasp_command`, `vasp_encut`, `vasp_ediff`, `vasp_pp_path`, `vasp_setups`)
- `config/lammps/md_nvt.in` — NVT template with placeholders: `${T_RUN}`, `${RUN_STEPS}` (used by `md_test.py`); `${POT_PATH}`, `${GRADE_THRESHOLD}`, `${GRADE_BREAK}`, `${READ_DATA_LINE}` (additionally used by `active_learning.py`)
- `config/lammps/energy_volume.in`, `config/lammps/relax.in`, `config/lammps/phonon_dispersion_mtp.in` — single-point / relaxation templates for `pair_style hybrid/overlay mtp nlh`, used by the `--backend lammps` path of the tests below (for potentials whose radial basis type `mlp calculate_efs`/`mlp relax` can't load)

### Source modules
- `src/train.py` — thin wrapper around `mlp train`; all hyperparameters from `config/training.yaml`.
- `src/active_learning.py` — full MLIP-2 tutorial-2 loop (steps A–E). CLI flags: `--pot`, `--max-iter`, `--structure` (overrides `lammps_structure` in YAML), `--al-pot` (explicit output path, prevents suffix-chaining), `--temperature` (overrides `temperature` in YAML). `_IDX_TO_SYMBOL` is the reverse of `SPECIES_MAP` (in `src/utils/convert.py`) and must stay in sync with it.
- `src/test_errors.py` — wraps `mlp check_errors` / `mlp calculate_efs`.

### Utils (`src/utils/`)
Holds shared code imported by nearly every module under `src/physical_validation/` (each of those adds `src/utils/` to `sys.path` alongside its own directory), plus `convert.py`, imported the same way by `src/active_learning.py` and `scripts/save/collect_thermal_iter.py`.

- `src/utils/convert.py` — `--format {xyz,dump}`. `xyz` (default): reads extended XYZ via ASE, writes MLIP-3 CFG with energy/forces (`write_cfg`); `SPECIES_MAP` maps element symbol → integer type index; configs without `free_energy` are silently skipped; energy written to CFG is `free_energy` (not `energy`). `dump`: reads a LAMMPS trajectory dump, writes a bare CFG with zero forces/no energy (`write_dump_cfg`, via `parse_dump`) for `mlp calculate_grade`.
- `src/utils/utils.py` — shared helpers: `load_structure` (CFG/XYZ/LAMMPS data → cell+positions+types; defaults to 2-atom Ge diamond), `write_bare_cfg`, `calc_efs`, `compute_rdf`, `read_lammps_dump`, `coordination_number`. Sets `LD_LIBRARY_PATH` for OpenBLAS via `_mlp_env()`.
- `src/utils/mtp_calculator.py` — `MTPCalculator` (wraps `mlp calculate_efs`) and `MTPLammpsCalculator` (wraps LAMMPS `pair_style hybrid/overlay mtp nlh`, via `config/lammps/phonon_dispersion_mtp.in`): ASE `Calculator` subclasses, one subprocess per evaluation; intended for NEB images and small-cell relaxations.
- `src/utils/test_mtp_calculator.py` — pytest smoke test: `MTPCalculator` returns a float energy and near-zero forces on perfect diamond. `MTP_POT` env var overrides the default potential path.

### Physical validation (`src/physical_validation/`)
Property-calculation/QA scripts that evaluate a trained potential against DFT/experimental references — not software unit tests. Most accept `--backend {mlp,lammps}` (or, for `phonon_dispersion.py`/`liquid_rdf.py`, a finer-grained `--method`/`--backend` with an `-mlip`/`-nlh` split — see each file's docstring): `mlp` (default) evaluates via `mlp calculate_efs`/`mlp relax`; the LAMMPS backend runs `pair_style hybrid/overlay mtp nlh` instead, for potentials trained with a radial basis type the `mlp` backend can't load ("Wrong radial basis type"). The two backends share all structure-building/reference-loading code within each file so they can't drift apart.

- `energy_volume.py` — E-V curve via uniform lattice scaling; Birch-Murnaghan EOS fit; PNG plot. Also exports `DEFAULT_LAMMPS`/`_run_lammps_energy`, reused by several other modules' LAMMPS backend.
- `elastic_constants.py` — full 6×6 stiffness tensor via central-difference finite strain; derives C11, C12, C44, B, G, E, ν.
- `elastic_constant/` — LAMMPS-native elastic constants (`in.elastic`, `potential.mod` with the `pair_style mtp+nlh` workaround), driven by `scripts/submit_elastic_constants_lammps.sh`.
- `defect_formation.py` — formation energies for vacancy and SIA types (T, H, X, B).
- `phonon_dispersion.py` — phonon dispersion along high-symmetry paths; also exports `_run_lammps_forces_nlh`, reused by `thermal_properties.py`'s LAMMPS backend.
- `thermal_properties.py` — thermal expansion coefficient αL and heat capacity via phonon DOS (QHA).
- `multiphase_ev.py` — E-V curves for 8 Ge phases (diamond, hd, hcp, fcc, bcc, beta-Sn, bc8, st12); overlays DFT reference from `reference/`. Structure parameters match the ilearn DFT reference exactly.
- `vacancy_migration.py` — 1NN and 2NN vacancy hop barriers via climbing-image NEB. `--a0` should match the MTP equilibrium lattice constant.
- `quasi_static_drag.py` — stepwise atom displacement along [100], [110], [111], [031]; computes ΔE and parallel force per step; compares against DFT reference in `reference/`.
- `dimer_curve.py` — Ge-Ge dimer energy curve via LAMMPS; no `mlp`-backend counterpart (short-range/NLH table reference).
- `liquid_rdf.py` — liquid Ge RDF via a 3-stage LAMMPS NVT→NPT→NVT MD run; `--backend {lammps-mlip,lammps-nlh}` picks the pair_style.
- `amorphous_rdf.py`, `melting_point.py` — amorphous-Ge RDF and melting-point bracketing (see each file's docstring).

### Scripts (`scripts/`)
`scripts/` holds only the SLURM/HPC submission wrappers (`submit_*.sh`, each with `#SBATCH` headers, launched via `sbatch`). Everything else — the plain, directly-run pipeline steps (`00_convert.sh`, `01_train.sh`, `02_test_errors.sh`, `03_active_learning_*.sh`, `augment_training_set.sh`, `run_light_tests.sh`) and the one-off Python structure-generation/data-wrangling scripts — lives in `scripts/save/` and is invoked as `python scripts/save/<name>.py` / `bash scripts/save/<name>.sh`. Some `submit_*.sh` scripts call into `scripts/save/` internally (e.g. `submit_test_errors.sh` → `scripts/save/02_test_errors.sh`).

### Structure generation scripts (`scripts/save/`)
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
