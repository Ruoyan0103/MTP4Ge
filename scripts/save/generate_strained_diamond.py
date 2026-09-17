"""
Generate strained diamond-cubic Ge VASP input directories.

Applies 6 Voigt strain modes at magnitudes ±1%, ±2%, ±5% to a 2×2×2
conventional supercell (64 atoms). Run VASP in each directory, then use
scripts/save/collect_dft_results.py to gather results into an XYZ file.

Usage (from repo root):
    python scripts/save/generate_strained_diamond.py [--outdir data/strained_dft]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from ase.build import bulk
    from ase.io import write as ase_write
    import yaml
except ImportError as e:
    print(f"ERROR: {e}. Activate the mtp4ge conda/venv environment.", file=sys.stderr)
    sys.exit(1)


AL_CONFIG = "config/active_learning.yaml"

# Strain modes: (label, 3×3 ε tensor — symmetric, engineering convention)
# Off-diagonal entries use ε_ij = γ/2 (tensor shear strain).
def _strain_matrix(mode: str, e: float) -> np.ndarray:
    F = np.eye(3)
    if mode == "xx":
        F[0, 0] += e
    elif mode == "yy":
        F[1, 1] += e
    elif mode == "zz":
        F[2, 2] += e
    elif mode == "yz":
        F[1, 2] += e / 2
        F[2, 1] += e / 2
    elif mode == "xz":
        F[0, 2] += e / 2
        F[2, 0] += e / 2
    elif mode == "xy":
        F[0, 1] += e / 2
        F[1, 0] += e / 2
    elif mode == "vol":
        s = (1 + e) ** (1 / 3)
        F *= s
    return F


STRAIN_MODES = ["xx", "yy", "zz", "yz", "xz", "xy", "vol"]
MAGNITUDES = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]

GE_A0 = 5.762  # Å — PBE equilibrium, consistent with existing distorted_bulk training data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/strained_dft",
                        help="Root directory for VASP calculation subdirs")
    parser.add_argument("--supercell", type=int, default=2,
                        help="Supercell size n (n×n×n conventional cell, default 2)")
    args = parser.parse_args()

    with open(AL_CONFIG) as f:
        al_cfg = yaml.safe_load(f)

    pp_path = al_cfg.get("vasp_pp_path", "/appl/soft/phys/vasp")
    setups  = al_cfg.get("vasp_setups", {"Ge": "_d"})
    encut   = al_cfg.get("vasp_encut", 500)
    ediff   = al_cfg.get("vasp_ediff", 1e-7)

    # Resolve POTCAR: pp_path/potpaw_PBE.64/<elem><suffix>/POTCAR
    pp_version = "potpaw_PBE.64"
    potcar_src = Path(pp_path) / pp_version / f"Ge{setups.get('Ge', '_d')}" / "POTCAR"
    if not potcar_src.exists():
        print(f"WARNING: POTCAR not found at {potcar_src}", file=sys.stderr)
        potcar_src = None

    # Build base structure
    n = args.supercell
    prim = bulk("Ge", crystalstructure="diamond", a=GE_A0, cubic=True)
    base = prim.repeat(n)  # 2×2×2 → 64 atoms

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    incar_template = f"""\
SYSTEM = Ge strained bulk
ENCUT  = {encut}
EDIFF  = {ediff:.1E}
IBRION = -1
NSW    = 0
NELM   = 100
ISIF   = 2
PREC   = Accurate
ISMEAR = 0
SIGMA  = 0.05
LASPH  = .TRUE.
LWAVE  = .FALSE.
LCHARG = .FALSE.
KSPACING = 0.15
KGAMMA   = .TRUE.
GGA    = PE
"""

    dirs_written = []
    for mode in STRAIN_MODES:
        for mag in MAGNITUDES:
            label = f"{mode}_{mag:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
            calc_dir = outdir / label
            calc_dir.mkdir(parents=True, exist_ok=True)

            F = _strain_matrix(mode, mag)
            strained = base.copy()
            new_cell = strained.get_cell() @ F
            strained.set_cell(new_cell, scale_atoms=True)

            # POSCAR
            ase_write(str(calc_dir / "POSCAR"), strained, format="vasp", direct=False)
            # INCAR
            (calc_dir / "INCAR").write_text(incar_template)
            # POTCAR symlink
            potcar_dst = calc_dir / "POTCAR"
            if potcar_src and not potcar_dst.exists():
                os.symlink(str(potcar_src.resolve()), str(potcar_dst))

            # Tag the config type so collect_dft_results.py can label it
            (calc_dir / "config_type.txt").write_text("strained_diamond\n")

            dirs_written.append(calc_dir)
            print(f"  Wrote {calc_dir}")

    print(f"\nWrote {len(dirs_written)} VASP directories under {outdir}")
    print("Submit VASP calculations, then run:")
    print("  python scripts/save/collect_dft_results.py --indir data/strained_dft --out data/strained_diamond.xyz")


if __name__ == "__main__":
    main()
