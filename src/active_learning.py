"""
Active learning loop for MTP potential refinement.

Follows the MLIP-2 tutorial-2 workflow (md_al_mtp.sh):

  A. calculate_grade  — build/update active set in pot.almtp
  B. LAMMPS MD        — run MD with MLIP selection; extrapolative configs
                        written to iter_dir/preselected.cfg
  C. select_add       — pick most informative subset → data/selected.cfg
  D/E. VASP DFT       — single-point DFT on each selected structure (ASE Vasp)
  F. retrain          — append 80 % to train.cfg, 10 % each to val/test, retrain
  A. calculate_grade  — update active set with newly labeled configs

Usage:
    python src/active_learning.py
    python src/active_learning.py --pot results/potentials/pot.almtp --max-iter 10
"""

import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_TRAIN_CONFIG = "config/training.yaml"
DEFAULT_AL_CONFIG = "config/active_learning.yaml"

# Reverse of convert.py SPECIES_MAP: type_index → element symbol
_IDX_TO_SYMBOL: dict[int, str] = {0: "Ge"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _count_cfg(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text().count("BEGIN_CFG")


# ---------------------------------------------------------------------------
# MLIP commands
# ---------------------------------------------------------------------------

def calc_grade(mlp: str, pot: str, in_cfg: str, out_cfg: str) -> None:
    """Run calculate_grade to build/update the active set embedded in pot."""
    cmd = [mlp, "calculate_grade", pot, in_cfg, out_cfg]
    print("  Calc grade:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  Warning: calculate_grade exited {result.returncode}")
    Path(out_cfg).unlink(missing_ok=True)


def select_add(mlp: str, pot: str, train_cfg: str, preselected: str, selected: Path) -> int:
    cmd = [mlp, "select_add", pot, train_cfg, preselected, str(selected)]
    print("  Select:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _count_cfg(selected)


# ---------------------------------------------------------------------------
# LAMMPS MD (Step B)
# ---------------------------------------------------------------------------

def write_mlip_al_ini(
    pot_path: str,
    preselected_cfg: str,
    threshold: float,
    run_dir: Path,
) -> Path:
    """Write mlip.ini with active learning selection enabled.

    MLIP-3 reads the active set from inside the .almtp file — no
    select:load-state needed (that key is MLIP-2 / .mtp only).
    select:threshold-break is set high so LAMMPS does not abort on
    extreme extrapolation; it merely collects the structure.
    """
    ini = run_dir / "mlip.ini"
    ini.write_text(
        f"mtp-filename = {Path(pot_path).resolve()}\n"
        f"select = TRUE\n"
        f"select:threshold = {threshold}\n"
        f"select:threshold-break = {threshold * 5:.1f}\n"
        f"select:energy-weight = 1.0\n"
        f"select:force-weight = 0.0\n"
        f"select:stress-weight = 0.0\n"
        f"select:save-selected = {Path(preselected_cfg).resolve()}\n"
    )
    return ini


def run_lammps_md(
    lammps: str,
    template: Path,
    T: float,
    steps: int,
    run_dir: Path,
) -> Path:
    """Patch the LAMMPS template and run MD from run_dir.

    LAMMPS runs with cwd=run_dir so 'pair_style mlip mlip.ini' resolves
    to the mlip.ini written by write_mlip_al_ini in the same directory.
    Returns the path where MLIP writes extrapolative configs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    text = (
        template.read_text()
        .replace("${T_RUN}", str(T))
        .replace("${RUN_STEPS}", str(steps))
    )
    (run_dir / "md.in").write_text(text)
    cmd = shlex.split(lammps) + ["-in", "md.in"]
    print("  LAMMPS:", " ".join(cmd), f"(cwd={run_dir})")
    result = subprocess.run(cmd, cwd=str(run_dir))
    if result.returncode != 0:
        print(f"  Warning: LAMMPS exited {result.returncode}")
    return run_dir / "preselected.cfg"


# ---------------------------------------------------------------------------
# CFG reader  (inverse of convert.py write_cfg)
# ---------------------------------------------------------------------------

def read_cfg(path: Path) -> list[tuple]:
    """Parse MLIP CFG file → list of (ASE Atoms, feature_dict) tuples."""
    from ase import Atoms

    blocks = re.split(r"(?=BEGIN_CFG\b)", path.read_text())
    blocks = [b.strip() for b in blocks if b.strip().startswith("BEGIN_CFG")]
    results = []

    for block in blocks:
        lines = block.splitlines()
        idx = 0
        n_atoms = 0
        cell = []
        atom_types: list[int] = []
        positions: list[list[float]] = []
        forces: list[list[float]] = []
        features: dict[str, str] = {}

        while idx < len(lines):
            line = lines[idx].strip()

            if line == "Size":
                idx += 1
                n_atoms = int(lines[idx].strip())

            elif line == "Supercell":
                for _ in range(3):
                    idx += 1
                    cell.append([float(x) for x in lines[idx].split()])

            elif line.startswith("AtomData:"):
                for _ in range(n_atoms):
                    idx += 1
                    parts = lines[idx].split()
                    # id type x y z fx fy fz
                    atom_types.append(int(parts[1]))
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    forces.append([float(parts[5]), float(parts[6]), float(parts[7])])

            elif line.startswith("Feature"):
                m = re.match(r"Feature\s+(\S+)\s+(.*)", line)
                if m:
                    features[m.group(1)] = m.group(2).strip()

            idx += 1

        symbols = [_IDX_TO_SYMBOL.get(t, "X") for t in atom_types]
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )
        atoms.arrays["forces_mlip"] = np.array(forces)
        results.append((atoms, features))

    return results


# ---------------------------------------------------------------------------
# VASP DFT (Steps D/E)
# ---------------------------------------------------------------------------

def run_vasp_single_point(
    atoms,
    vasp_dir: Path,
    vasp_settings: dict,
) -> object:
    """Run a single-point VASP calculation on atoms; return atoms with results.

    VASP_PP_PATH is set from vasp_pp_path in config so ASE can find POTCAR.
    vasp_setups selects the pseudopotential variant, e.g. {'Ge': '_d'} →
    uses Ge_d/POTCAR from the VASP_PP_PATH directory.
    vasp_command supports full srun/mpirun prefix, e.g. 'srun -n 32 vasp_std'.
    """
    from ase.calculators.vasp import Vasp

    vasp_dir.mkdir(parents=True, exist_ok=True)

    pp_path = vasp_settings.get("vasp_pp_path", "")
    if pp_path:
        os.environ["VASP_PP_PATH"] = pp_path

    vasp_cmd = vasp_settings.get("vasp_command", "vasp_std")
    setups = vasp_settings.get("vasp_setups", {})

    calc = Vasp(
        command=vasp_cmd,
        xc="PBE",
        setups=setups,
        encut=vasp_settings.get("vasp_encut", 400),
        kpts=(1, 1, 1),
        gamma=True,
        ibrion=-1,
        nsw=0,
        isif=2,          # compute stress, no cell relaxation
        prec="Accurate",
        ediff=vasp_settings.get("vasp_ediff", 1e-6),
        ismear=1,
        sigma=0.2,       # Methfessel-Paxton for metals
        lwave=False,
        lcharg=False,
        directory=str(vasp_dir),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return atoms


def run_dft_labeling(
    selected_cfg: Path,
    iter_dir: Path,
    vasp_settings: dict,
) -> Path:
    """Run VASP on every structure in selected_cfg; write labelled.cfg."""
    from src.convert import write_cfg

    atoms_list = read_cfg(selected_cfg)
    labeled = []
    for i, (atoms, meta) in enumerate(atoms_list):
        vasp_dir = iter_dir / "vasp" / f"struct_{i:03d}"
        print(f"  DFT struct {i:03d} → {vasp_dir}")
        try:
            atoms = run_vasp_single_point(atoms, vasp_dir, vasp_settings)
            labeled.append((meta.get("orig_index", str(i)), atoms))
        except Exception as exc:
            print(f"  WARNING: VASP failed for struct {i}: {exc} — skipping")

    labelled_path = iter_dir / "labelled.cfg"
    write_cfg(labeled, labelled_path)
    return labelled_path


# ---------------------------------------------------------------------------
# Training-set management
# ---------------------------------------------------------------------------

def split_and_distribute_cfg(
    labelled: Path,
    train_data: Path,
    val_path: Path,
    test_path: Path,
    rng_seed: int,
) -> None:
    """Append new labeled configs 80/10/10 to train / val / test."""
    blocks = re.split(r"(?=BEGIN_CFG\b)", labelled.read_text())
    blocks = [b.strip() for b in blocks if b.strip().startswith("BEGIN_CFG")]
    if not blocks:
        print("  No configs to distribute.")
        return

    rng = random.Random(rng_seed)
    rng.shuffle(blocks)
    n = len(blocks)
    n_test = max(1, n // 10)
    n_val  = max(1, n // 10)
    test_blocks  = blocks[:n_test]
    val_blocks   = blocks[n_test : n_test + n_val]
    train_blocks = blocks[n_test + n_val :]

    def _append(path: Path, blks: list[str]) -> None:
        if not blks:
            return
        with open(path, "a") as f:
            f.write("\n\n".join(blks) + "\n")

    _append(train_data, train_blocks)
    _append(val_path,   val_blocks)
    _append(test_path,  test_blocks)
    print(
        f"  Split: {len(train_blocks)} → train | "
        f"{len(val_blocks)} → val | {len(test_blocks)} → test"
    )


def retrain(train_cfg_path: str, pot_path: str, train_config: str, iteration: int) -> None:
    from src.train import load_config, run_training
    cfg = load_config(train_config)
    overrides = {
        "train_cfg": train_cfg_path,
        "output_potential": pot_path,
        "seed": iteration,
    }
    run_training(cfg, overrides)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_active_learning(
    train_cfg: dict,
    al_cfg: dict,
    pot_path: str,
    max_iter: int,
) -> None:
    mlp            = train_cfg["mlp_binary"]
    train_data     = Path(train_cfg["train_cfg"])
    val_path       = Path("data/val.cfg")
    test_path      = Path("data/test.cfg")
    grade_threshold = al_cfg.get("grade_threshold", 2.0)
    temperature    = al_cfg.get("temperature", 300)
    md_steps       = al_cfg.get("md_steps", 10000)
    lammps         = al_cfg.get("lammps_binary", "lmp")
    md_template    = Path("config/lammps/md_nvt.in")
    al_outdir      = Path("results/active_learning")
    al_outdir.mkdir(parents=True, exist_ok=True)

    vasp_settings  = {k: v for k, v in al_cfg.items() if k.startswith("vasp_")}

    print("=== Active Learning ===")
    print(f"  Potential     : {pot_path}")
    print(f"  Training set  : {train_data}  ({_count_cfg(train_data)} configs)")
    print(f"  Temperature   : {temperature} K  |  MD steps: {md_steps}")
    print(f"  Grade threshold: {grade_threshold}  |  Max iterations: {max_iter}")

    if not train_data.exists():
        print(f"ERROR: training set not found: {train_data}", file=sys.stderr)
        sys.exit(1)

    # Step A (init): build initial active set in pot
    print("\nStep A — initializing active set (calculate_grade on training set)...")
    calc_grade(mlp, pot_path, str(train_data), str(al_outdir / "init_grade.cfg"))

    for iteration in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration}/{max_iter}")
        print(f"{'='*60}")

        iter_dir = al_outdir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        # Step B: LAMMPS MD with MLIP selection enabled
        print("\nStep B — LAMMPS MD with active learning selection...")
        write_mlip_al_ini(
            pot_path,
            str((iter_dir / "preselected.cfg").resolve()),
            grade_threshold,
            iter_dir,
        )
        preselected = run_lammps_md(lammps, md_template, temperature, md_steps, iter_dir)

        n_pre = _count_cfg(preselected)
        if n_pre == 0:
            print("\nActive learning converged: no extrapolative structures found.")
            break
        print(f"  {n_pre} preselected structures found.")

        # Step C: select_add — pick most informative subset
        print("\nStep C — select_add...")
        selected = Path("data/selected.cfg")
        n_selected = select_add(mlp, pot_path, str(train_data), str(preselected), selected)
        if n_selected == 0:
            print("\nActive learning converged: select_add chose 0 structures.")
            break
        print(f"  {n_selected} structures selected.")

        # Steps D/E: VASP single-point DFT
        print(f"\nSteps D/E — VASP DFT on {n_selected} structures...")
        labelled = run_dft_labeling(selected, iter_dir, vasp_settings)
        shutil.copy(labelled, "data/labelled.cfg")
        print(f"  Labelled configs written to data/labelled.cfg")

        # Distribute 80/10/10 → train / val / test
        split_and_distribute_cfg(labelled, train_data, val_path, test_path, rng_seed=iteration)

        # Step F: retrain
        n_train = _count_cfg(train_data)
        print(f"\nStep F — retraining on {n_train} configs...")
        retrain(str(train_data), pot_path, DEFAULT_TRAIN_CONFIG, iteration)

        # Step A (update): refresh active set with newly labeled structures
        print("\nStep A (update) — refreshing active set...")
        calc_grade(mlp, pot_path, str(selected), str(iter_dir / "grade_update.cfg"))

    print(f"\nActive learning complete.")
    print(f"  Final training set: {_count_cfg(train_data)} configs")
    print(f"  Val set           : {_count_cfg(val_path)} configs")
    print(f"  Test set          : {_count_cfg(test_path)} configs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Active learning loop for MTP refinement")
    parser.add_argument("--config",    default=DEFAULT_TRAIN_CONFIG, help="Training YAML config")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG,    help="Active learning YAML config")
    parser.add_argument("--pot",       help="Potential path (overrides config)")
    parser.add_argument("--max-iter",  type=int, help="Max AL iterations (overrides al-config)")
    args = parser.parse_args()

    train_cfg = _load(args.config)
    al_cfg    = _load(args.al_config)

    pot_path = args.pot or train_cfg["output_potential"]
    max_iter = args.max_iter or al_cfg.get("max_iterations", 20)

    run_active_learning(train_cfg, al_cfg, pot_path, max_iter)


if __name__ == "__main__":
    main()
