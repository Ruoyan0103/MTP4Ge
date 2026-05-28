"""
Convert extended XYZ training data to MLIP-3 CFG format.

Usage:
    python src/convert.py --input <file.xyz> --outdir data/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from ase.io import read as ase_read
except ImportError:
    print("ERROR: ASE not found. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)


def _get_config_type(atoms) -> str:
    return atoms.info.get("config_type", atoms.info.get("Config_type", "unknown"))


# Map element symbol → integer type index (extend for multi-species systems)
SPECIES_MAP: dict[str, int] = {"Ge": 0}


def _parse_virial_from_atoms(atoms) -> list[float] | None:
    """Extract Voigt stress components [xx, yy, zz, yz, xz, xy] in eV."""
    info = atoms.info
    for key in ("virial", "Virial", "stress", "Stress"):
        if key in info:
            v = np.asarray(info[key]).flatten()
            if v.size == 9:
                return [v[0], v[4], v[8], v[5], v[2], v[1]]
            if v.size == 6:
                return list(v)
    try:
        vol = atoms.get_volume()
        stress_voigt = atoms.get_stress(voigt=True)  # eV/Å³
        virial = -stress_voigt * vol                  # eV
        return list(virial)
    except Exception:
        return None


def write_cfg(frames: list, path: Path) -> None:
    """Write a list of ASE Atoms objects to MLIP-3 CFG format."""
    species_map = SPECIES_MAP.copy()
    next_idx = max(species_map.values(), default=-1) + 1

    with open(path, "w") as f:
        for orig_idx, atoms in frames:
            cell = atoms.get_cell()
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()

            try:
                forces = atoms.get_forces()
            except Exception:
                forces = np.zeros_like(positions)

            calc_results = atoms.calc.results if atoms.calc else {}
            energy = calc_results.get("free_energy")
            if energy is None:
                continue
            virial = _parse_virial_from_atoms(atoms)
            config_type = _get_config_type(atoms)

            types = []
            for sym in symbols:
                if sym not in species_map:
                    species_map[sym] = next_idx
                    next_idx += 1
                types.append(species_map[sym])

            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"    {len(atoms)}\n")
            f.write(" Supercell\n")
            for row in cell:
                f.write(f"    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n")
            f.write(
                " AtomData:  id type       cartes_x      cartes_y"
                "      cartes_z           fx          fy          fz\n"
            )
            for i, (t, pos, frc) in enumerate(zip(types, positions, forces), start=1):
                f.write(
                    f"    {i:8d}  {t:3d}  "
                    f"  {pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}"
                    f"  {frc[0]:12.6f}  {frc[1]:12.6f}  {frc[2]:12.6f}\n"
                )
            f.write(" Energy\n")
            f.write(f"    {energy:.12f}\n")
            if virial is not None:
                f.write(" PlusStress:  xx          yy          zz"
                        "          yz          xz          xy\n")
                f.write(
                    "    "
                    + "  ".join(f"{v:12.5f}" for v in virial)
                    + "\n"
                )
            f.write(f" Feature   orig_index\t {orig_idx}\n")
            f.write(f" Feature   type\t {config_type}\n")
            f.write("END_CFG\n\n")

    print(f"  Wrote {len(frames)} configs → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extended XYZ to MLIP-3 CFG")
    parser.add_argument("--input", required=True, help="Input .xyz file")
    parser.add_argument("--outdir", default="data/", help="Output directory (default: data/)")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path} ...")
    all_frames = list(enumerate(ase_read(str(input_path), index=":")))
    print(f"  Loaded {len(all_frames)} configurations.")

    skipped = sum(1 for _, a in all_frames if not (a.calc and a.calc.results.get("free_energy")))
    if skipped:
        print(f"  WARNING: {skipped} configs have no free_energy label and will be skipped.")

    frames = [(i, a) for i, a in all_frames if a.calc and a.calc.results.get("free_energy")]
    out_name = input_path.stem + ".cfg"
    write_cfg(frames, outdir / out_name)


if __name__ == "__main__":
    main()
