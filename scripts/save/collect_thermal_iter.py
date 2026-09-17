"""Collect completed VASP single-points from struct_NNN/ dirs under a thermal_dft
iteration folder and write them into one labelled.cfg.

Usage:
    python scripts/save/collect_thermal_iter.py <iter_dir> [--config-type thermal_dft]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "utils"))

from ase.io import read as ase_read
from convert import write_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iter_dir", type=Path)
    parser.add_argument("--config-type", default="strained")
    parser.add_argument("--out", default=None, help="Output path (default: <iter_dir>/labelled.cfg)")
    args = parser.parse_args()

    struct_dirs = sorted(args.iter_dir.glob("struct_*"))
    frames = []
    for i, struct_dir in enumerate(struct_dirs):
        outcar = struct_dir / "OUTCAR"
        if not outcar.exists():
            print(f"  WARNING: {struct_dir.name} has no OUTCAR — skipping")
            continue
        try:
            atoms = ase_read(str(outcar))
        except Exception as exc:
            print(f"  WARNING: failed to read {outcar}: {exc} — skipping")
            continue

        calc = atoms.calc
        if calc is not None and "free_energy" not in calc.results:
            calc.results["free_energy"] = calc.results.get("energy")
        atoms.info["config_type"] = args.config_type
        frames.append((i, atoms))

    out_path = args.out or (args.iter_dir / "labelled.cfg")
    write_cfg(frames, Path(out_path))


if __name__ == "__main__":
    main()
