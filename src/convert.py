"""
Convert extended XYZ training data to MLIP-3 CFG format.

Usage:
    python src/convert.py --input <file.xyz> --outdir data/ --split 0.8 0.1 0.1
    python src/convert.py --input <file.xyz> --outdir data/ --no-split
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np

try:
    from ase.io import read as ase_read
except ImportError:
    print("ERROR: ASE not found. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)


# Map element symbol → integer type index (extend for multi-species systems)
SPECIES_MAP: dict[str, int] = {"Ge": 0}


def _parse_virial_from_atoms(atoms) -> list[float] | None:
    """Extract Voigt stress components [xx, yy, zz, yz, xz, xy] in eV."""
    info = atoms.info
    # ASE stores virial (= -stress * volume) in various keys
    for key in ("virial", "Virial", "stress", "Stress"):
        if key in info:
            v = np.asarray(info[key]).flatten()
            if v.size == 9:
                return [v[0], v[4], v[8], v[5], v[2], v[1]]
            if v.size == 6:
                return list(v)
    # ASE atoms.get_stress() returns [xx, yy, zz, yz, xz, xy] in eV/Å³
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
        for atoms in frames:
            cell = atoms.get_cell()
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()

            try:
                forces = atoms.get_forces()
            except Exception:
                forces = np.zeros_like(positions)

            energy = atoms.get_potential_energy()
            virial = _parse_virial_from_atoms(atoms)

            # Build type index for each symbol
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
            f.write("END_CFG\n\n")

    print(f"  Wrote {len(frames)} configs → {path}")


def convert(
    input_xyz: Path,
    outdir: Path,
    split: tuple[float, float, float] | None,
    seed: int = 42,
) -> None:
    print(f"Reading {input_xyz} ...")
    frames = ase_read(str(input_xyz), index=":")
    print(f"  Loaded {len(frames)} configurations.")

    outdir.mkdir(parents=True, exist_ok=True)

    if split is None:
        write_cfg(frames, outdir / "train.cfg")
        return

    random.seed(seed)
    idx = list(range(len(frames)))
    random.shuffle(idx)

    n = len(idx)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    write_cfg([frames[i] for i in train_idx], outdir / "train.cfg")
    write_cfg([frames[i] for i in val_idx], outdir / "val.cfg")
    write_cfg([frames[i] for i in test_idx], outdir / "test.cfg")

    # Save index files for reproducibility
    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        (outdir / f"{name}_indices.txt").write_text("\n".join(map(str, indices)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extended XYZ to MLIP-3 CFG")
    parser.add_argument("--input", required=True, help="Input .xyz file")
    parser.add_argument("--outdir", default="data/", help="Output directory")
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument("--no-split", action="store_true", help="Write single train.cfg without splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    split = None if args.no_split else tuple(args.split)
    if split is not None and abs(sum(split) - 1.0) > 1e-6:
        parser.error(f"Split ratios must sum to 1.0, got {sum(split):.4f}")

    convert(
        input_xyz=Path(args.input),
        outdir=Path(args.outdir),
        split=split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
