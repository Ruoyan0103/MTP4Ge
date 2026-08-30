"""Convert MLIP-3 CFG files to a VASP XDATCAR trajectory (readable by VESTA/OVITO).

Multiple CFG blocks in one file become successive XDATCAR configurations.
All blocks must share the same species and atom count (a real trajectory);
use --allow-varying-cell if the cell changes frame to frame (e.g. NPT runs) —
ASE writes a fresh lattice header for each frame in that case.

Usage:
    python scripts/cfg_to_xdatcar.py path/to/file.cfg [path/to/XDATCAR]

Default output: same directory as input, named XDATCAR.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg_to_xyz import _parse_all_cfgs  # noqa: E402

from ase.io import write as ase_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MLIP CFG to VASP XDATCAR")
    parser.add_argument("cfg", help="Input .cfg file")
    parser.add_argument("out", nargs="?", help="Output XDATCAR path (default: <cfg_dir>/XDATCAR)")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        sys.exit(f"Error: {cfg_path} does not exist")

    atoms_list = _parse_all_cfgs(cfg_path)
    n = len(atoms_list)

    out_path = Path(args.out) if args.out else cfg_path.parent / "cvt_XDATCAR"

    ase_write(str(out_path), atoms_list, format="vasp-xdatcar")
    print(f"Written {n} frame(s) -> {out_path}")


if __name__ == "__main__":
    main()
