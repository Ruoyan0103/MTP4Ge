"""
Collect finished VASP single-point calculations into an extended XYZ file.

For each subdirectory under --indir that contains a completed vasprun.xml
(or OUTCAR), read the structure + energy/forces/stress and write to --out.
A config_type.txt file in each directory sets the config_type label.

Usage (from repo root):
    python scripts/save/collect_dft_results.py --indir data/strained_dft --out data/strained_diamond.xyz
    python scripts/save/collect_dft_results.py --indir data/sia_dft      --out data/sia_configs.xyz
    python scripts/save/collect_dft_results.py --indir data/aimd_snapshots --out data/aimd.xyz
"""

import argparse
import sys
from pathlib import Path

try:
    from ase.io import read as ase_read, write as ase_write
    from ase.calculators.vasp import Vasp
except ImportError as e:
    print(f"ERROR: {e}. Activate the mtp4ge environment.", file=sys.stderr)
    sys.exit(1)


def read_vasp_dir(calc_dir: Path):
    """
    Read a completed VASP directory. Returns (atoms, config_type) or None on failure.
    Prefer vasprun.xml (full precision); fall back to OUTCAR.
    """
    vasprun = calc_dir / "vasprun.xml"
    outcar = calc_dir / "OUTCAR"

    config_type_file = calc_dir / "config_type.txt"
    config_type = config_type_file.read_text().strip() if config_type_file.exists() else "unknown"

    source = None
    if vasprun.exists():
        source = str(vasprun)
    elif outcar.exists():
        source = str(outcar)
    else:
        return None

    try:
        atoms = ase_read(source, format="vasp-xml" if "xml" in source else "vasp-out")
    except Exception as exc:
        print(f"  WARNING: could not read {source}: {exc}")
        return None

    # Check that energy is available
    try:
        _ = atoms.get_potential_energy()
    except Exception:
        print(f"  WARNING: no energy in {calc_dir} — skipping")
        return None

    atoms.info["config_type"] = config_type
    return atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True,
                        help="Root directory containing VASP calculation subdirs (searched recursively)")
    parser.add_argument("--out", required=True,
                        help="Output extended XYZ file")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    args = parser.parse_args()

    indir = Path(args.indir)
    if not indir.is_dir():
        print(f"ERROR: {indir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all leaf directories that have a POSCAR (i.e., were written by Vasp.write_input)
    calc_dirs = sorted(p.parent for p in indir.rglob("POSCAR"))

    collected = []
    for calc_dir in calc_dirs:
        result = read_vasp_dir(calc_dir)
        if result is not None:
            collected.append(result)
            print(f"  OK  {calc_dir.relative_to(indir)}  ({len(result)} atoms)")
        else:
            print(f"  SKIP {calc_dir.relative_to(indir)}")

    if not collected:
        print("No completed VASP calculations found.")
        sys.exit(0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    ase_write(str(out), collected, format="extxyz", append=(mode == "a"))

    print(f"\nCollected {len(collected)} / {len(calc_dirs)} configs → {out}")
    print("Next step: add to training set by running:")
    print(f"  bash scripts/save/00_convert.sh {out}")


if __name__ == "__main__":
    main()
