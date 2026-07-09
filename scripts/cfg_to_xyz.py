"""Convert MLIP-3 CFG files to extended XYZ (readable by VESTA and OVITO).

Usage:
    python scripts/cfg_to_xyz.py path/to/file.cfg [path/to/out.xyz]
    python scripts/cfg_to_xyz.py path/to/file.cfg --poscar   # write POSCAR instead

Multiple CFG blocks in one file → multiple frames written to a single XYZ.
Default output: same directory as input, same stem, .xyz extension.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

# Must match SPECIES_MAP in src/convert.py
TYPE_TO_SYMBOL = {0: "Ge"}


def _parse_all_cfgs(path: Path) -> list[Atoms]:
    """Parse every CFG block in *path* and return a list of ASE Atoms objects."""
    text = path.read_text()
    blocks = re.findall(r"BEGIN_CFG(.*?)END_CFG", text, re.DOTALL)
    if not blocks:
        raise ValueError(f"No CFG block found in {path}")

    atoms_list = []
    for block in blocks:
        lines = block.splitlines()
        n_atoms, cell, positions, types, forces, energy = 0, [], [], [], [], None
        mode = None

        for line in lines:
            s = line.strip()
            if s == "Size":
                mode = "size"
            elif mode == "size":
                n_atoms = int(s); mode = None
            elif s == "Supercell":
                mode = "supercell"
            elif mode == "supercell":
                cell.append([float(x) for x in s.split()])
                if len(cell) == 3:
                    mode = None
            elif s.startswith("AtomData:"):
                header = s.split()[1:]  # id type cartes_x cartes_y cartes_z [fx fy fz]
                has_forces = "fx" in header
                mode = "atoms"
            elif mode == "atoms" and s and not s.startswith(
                ("Feature", "Energy", "PlusStress", "BEGIN", "END")
            ):
                parts = s.split()
                if len(parts) >= 5:
                    types.append(int(parts[1]))
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    if has_forces and len(parts) >= 8:
                        forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
                    if len(positions) == n_atoms:
                        mode = None
            elif s.startswith("Energy"):
                parts = s.split()
                if len(parts) >= 2:
                    energy = float(parts[1])

        symbols = [TYPE_TO_SYMBOL.get(t, f"X{t}") for t in types]
        atoms = Atoms(
            symbols=symbols,
            positions=np.array(positions),
            cell=np.array(cell),
            pbc=True,
        )
        info = {}
        if energy is not None:
            info["energy"] = energy
        atoms.info.update(info)
        if forces and len(forces) == n_atoms:
            atoms.arrays["forces"] = np.array(forces)
        atoms_list.append(atoms)

    return atoms_list


def main():
    parser = argparse.ArgumentParser(description="Convert MLIP CFG to XYZ or POSCAR")
    parser.add_argument("cfg", help="Input .cfg file")
    parser.add_argument("out", nargs="?", help="Output file (default: <stem>.xyz next to input)")
    parser.add_argument("--poscar", action="store_true", help="Write VASP POSCAR instead of XYZ")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        sys.exit(f"Error: {cfg_path} does not exist")

    atoms_list = _parse_all_cfgs(cfg_path)
    n = len(atoms_list)

    if args.out:
        out_path = Path(args.out)
    elif args.poscar:
        out_path = cfg_path.with_suffix(".POSCAR")
    else:
        out_path = cfg_path.with_suffix(".xyz")

    fmt = "vasp" if args.poscar else "extxyz"
    if args.poscar and n > 1:
        print(f"Warning: POSCAR supports only one frame; writing first of {n} blocks.")
        atoms_list = [atoms_list[0]]

    ase_write(str(out_path), atoms_list if len(atoms_list) > 1 else atoms_list[0], format=fmt)
    print(f"Written {n} frame(s) → {out_path}")


if __name__ == "__main__":
    main()
