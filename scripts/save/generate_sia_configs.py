"""
Generate self-interstitial atom (SIA) Ge VASP input directories.

Creates 4 SIA types in a 3×3×3 conventional supercell (8×27 = 216 host atoms + 1 interstitial):
  - T  : tetrahedral interstitial
  - H  : hexagonal interstitial
  - X  : <110> dumbbell (split interstitial)
  - B  : bond-centre interstitial

For each type, generates:
  - relaxed reference structure (1 config)
  - N_perturb randomly displaced configs (default 20)

Run VASP in each directory, then use scripts/save/collect_dft_results.py.

Usage (from repo root):
    python scripts/save/generate_sia_configs.py [--outdir data/sia_dft] [--perturb 20]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.vasp import Vasp
    import yaml
except ImportError as e:
    print(f"ERROR: {e}. Activate the mtp4ge environment.", file=sys.stderr)
    sys.exit(1)


AL_CONFIG = "config/active_learning.yaml"
GE_A0 = 5.7620  # Å — PBE lattice constant for Ge (experimental ~5.658 Å, but we use the DFT value)


def _vasp_calc(vasp_cmd, setups, encut, ediff, calc_dir):
    return Vasp(
        command=vasp_cmd,
        xc="PBE",
        setups=setups,
        encut=encut,
        kspacing=0.15, 
        kgamma=True,
        ibrion=-1,
        nsw=0,
        nelm=100,
        isif=2,
        prec="Accurate",
        ediff=ediff,
        ismear=0,
        sigma=0.05,
        lasph=True,
        lwave=False,
        lcharg=False,
        directory=str(calc_dir),
    )


def _write_calc(atoms, config_type, calc_dir, vasp_cmd, setups, encut, ediff):
    calc_dir.mkdir(parents=True, exist_ok=True)
    calc = _vasp_calc(vasp_cmd, setups, encut, ediff, calc_dir)
    calc.write_input(atoms)
    (calc_dir / "config_type.txt").write_text(f"{config_type}\n")
    return calc_dir


def _perturb(atoms: "Atoms", rng, amplitude: float = 0.05) -> "Atoms":
    """Random Cartesian displacement of all atoms by up to amplitude Å."""
    a = atoms.copy()
    disp = rng.uniform(-amplitude, amplitude, size=(len(a), 3))
    a.set_positions(a.get_positions() + disp)
    return a


def build_sia_structures(n: int = 3) -> dict[str, "Atoms"]:
    """
    Build one structure per SIA type in a n×n×n conventional supercell.
    Returns dict: label -> Atoms (with interstitial inserted).
    """
    prim = bulk("Ge", crystalstructure="diamond", a=GE_A0, cubic=True)
    sup = prim.repeat(n)
    cell = sup.get_cell()

    structures = {}

    # --- Tetrahedral (T) ---
    # T void at (1/2, 1/2, 1/2) fractional of the conventional cell → (a0/2, a0/2, a0/2).
    t_pos = cell @ np.array([1 / (2 * n), 1 / (2 * n), 1 / (2 * n)])
    t_struct = sup.copy()
    t_struct.append("Ge")
    t_struct.positions[-1] = t_pos
    structures["tet"] = t_struct

    # --- Hexagonal (H) ---
    # H site at (5/8, 5/8, 5/8) fractional of the conventional cell → (5*a0/8, 5*a0/8, 5*a0/8).
    h_pos = cell @ np.array([5 / (8 * n), 5 / (8 * n), 5 / (8 * n)])
    h_struct = sup.copy()
    h_struct.append("Ge")
    h_struct.positions[-1] = h_pos
    structures["hex"] = h_struct

    # --- <110> dumbbell (X / split) ---
    # Remove the atom nearest to the cell origin and place two atoms
    # displaced by ±d/2 along [110] from that site.
    d_110 = 1.2  # Å half-separation — consistent with defect_formation.py
    origin_atom_idx = 0
    origin_pos = sup.get_positions()[origin_atom_idx].copy()
    x110_dir = np.array([1, 1, 0], dtype=float) / np.sqrt(2)
    x_struct = sup.copy()
    x_struct.positions[origin_atom_idx] = origin_pos + d_110 * x110_dir
    x_struct.append("Ge")
    x_struct.positions[-1] = origin_pos - d_110 * x110_dir
    structures["split"] = x_struct

    # --- Bond-centre (B) ---
    # Place interstitial at the midpoint of the nearest-neighbour bond
    # between atom 0 and its nearest neighbour.
    positions = sup.get_positions()
    diffs = positions[1:] - positions[0]
    dists = np.linalg.norm(diffs, axis=1)
    nn_idx = np.argmin(dists) + 1
    bond_mid = 0.5 * (positions[0] + positions[nn_idx])
    b_struct = sup.copy()
    b_struct.append("Ge")
    b_struct.positions[-1] = bond_mid
    structures["bond"] = b_struct

    return structures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/sia_dft",
                        help="Root directory for VASP calculation subdirs")
    parser.add_argument("--supercell", type=int, default=3,
                        help="Conventional supercell size n (n×n×n, default 3 → 216+1 atoms)")
    parser.add_argument("--perturb", type=int, default=20,
                        help="Number of randomly displaced configs per SIA type (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(AL_CONFIG) as f:
        al_cfg = yaml.safe_load(f)

    pp_path = al_cfg.get("vasp_pp_path", "")
    if pp_path:
        os.environ["VASP_PP_PATH"] = pp_path

    vasp_cmd = al_cfg.get("vasp_command", "vasp_std")
    setups = al_cfg.get("vasp_setups", {})
    encut = al_cfg.get("vasp_encut", 500)
    ediff = al_cfg.get("vasp_ediff", 1e-7)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    structures = build_sia_structures(args.supercell)

    total = 0
    for label, atoms in structures.items():
        # 1) Un-perturbed reference structure
        ref_dir = outdir / label / "ref"
        _write_calc(atoms, label, ref_dir, vasp_cmd, setups, encut, ediff)
        print(f"  {label}/ref → {ref_dir}")
        total += 1

        # 2) Randomly perturbed snapshots (single-point, no relaxation)
        for i in range(args.perturb):
            perturbed = _perturb(atoms, rng, amplitude=0.05)
            calc_dir = outdir / label / f"perturb_{i:03d}"
            _write_calc(perturbed, label, calc_dir, vasp_cmd, setups, encut, ediff)
            total += 1

        print(f"  {label}: wrote ref + {args.perturb} perturbed configs")

    print(f"\nWrote {total} VASP directories under {outdir}")
    print("Submit VASP calculations, then run:")
    print("  python scripts/save/collect_dft_results.py --indir data/sia_dft --out data/sia_configs.xyz")


if __name__ == "__main__":
    main()
