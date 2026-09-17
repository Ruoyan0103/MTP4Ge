"""Convert VASP POSCAR/CONTCAR structure files to bare MLIP-3 CFG (no energy/forces).

Recursively finds files named POSCAR or CONTCAR under the given root and writes
a sibling <name>.cfg for each (e.g. POSCAR -> POSCAR.cfg, CONTCAR -> CONTCAR.cfg).

Usage:
    python scripts/save/poscar_to_cfg.py data/AL/Int
    python scripts/save/poscar_to_cfg.py data/AL/Int/B/POSCAR
"""

import argparse
import sys
from pathlib import Path

from ase.io import read as ase_read

# Must match SPECIES_MAP in src/utils/convert.py
SPECIES_MAP: dict[str, int] = {"Ge": 0}


def write_bare_cfg(cell, positions, types, path: Path) -> None:
    """Write a single geometry-only CFG block (no energy/forces)."""
    n = len(positions)
    with open(path, "w") as f:
        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        f.write(f"    {n}\n")
        f.write(" Supercell\n")
        for row in cell:
            f.write(f"    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n")
        f.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z\n")
        for i, (t, pos) in enumerate(zip(types, positions), start=1):
            f.write(
                f"    {i:8d}  {t:3d}    "
                f"{pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}\n"
            )
        f.write("END_CFG\n")


def convert(poscar_path: Path, out_path: Path | None = None) -> Path:
    atoms = ase_read(str(poscar_path), format="vasp")
    symbols = atoms.get_chemical_symbols()
    unknown = sorted(set(symbols) - SPECIES_MAP.keys())
    if unknown:
        raise ValueError(f"{poscar_path}: unmapped species {unknown} (add to SPECIES_MAP)")

    cell = atoms.get_cell().array
    positions = atoms.get_positions()
    types = [SPECIES_MAP[s] for s in symbols]

    if out_path is None:
        out_path = poscar_path.with_name(poscar_path.name + ".cfg")
    write_bare_cfg(cell, positions, types, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="POSCAR/CONTCAR file or a directory to search recursively")
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_file():
        targets = [root]
    else:
        targets = sorted(
            p for p in root.rglob("*") if p.is_file() and p.name in ("POSCAR", "CONTCAR")
        )

    if not targets:
        print(f"No POSCAR/CONTCAR files found under {root}", file=sys.stderr)
        sys.exit(1)

    for src in targets:
        out = convert(src)
        print(f"  {src} -> {out}")


if __name__ == "__main__":
    main()
