"""
Convert extended XYZ training data to MLIP-3 CFG format.

Modes:
    --mode split   Train/val/test random split (default, use for held-out test set)
    --mode no-split  Write everything as a single train.cfg
    --mode pool    Stratified split: one rep per config_type → seed.cfg,
                   remainder → candidate_pool.cfg (for AL from existing data)

Usage:
    python src/convert.py --input <file.xyz> --outdir data/ --mode split
    python src/convert.py --input <file.xyz> --outdir data/ --mode pool
    python src/convert.py --input <file.xyz> --outdir data/ --mode pool \\
        --seed-per-type 1 --max-per-type 5 --always-include dimer
"""

import argparse
import random
import sys
from collections import defaultdict
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

            energy = atoms.info.get("energy") or atoms.get_potential_energy()
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


def _has_energy(atoms) -> bool:
    try:
        atoms.get_potential_energy()
        return True
    except Exception:
        return False


def convert(
    input_xyz: Path,
    outdir: Path,
    split: tuple[float, float, float] | None,
    seed: int = 42,
) -> None:
    print(f"Reading {input_xyz} ...")
    all_frames = ase_read(str(input_xyz), index=":")
    frames = [f for f in all_frames if _has_energy(f)]
    skipped = len(all_frames) - len(frames)
    if skipped:
        print(f"  WARNING: skipped {skipped} configs with no energy label.")
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


def split_pool(
    input_xyz: Path,
    outdir: Path,
    seed_per_type: int = 1,
    max_per_type: int | None = None,
    always_include: list[str] | None = None,
    always_include_contains: list[str] | None = None,
    always_include_contains_except: list[str] | None = None,
    exclude: list[str] | None = None,
    rng_seed: int = 42,
) -> None:
    """Stratified split: seed set (N per config_type) + candidate pool (remainder).

    Outputs:
        data/seed.cfg           — minimal diverse starting set for MTP training
        data/candidate_pool.cfg — remaining labeled configs for AL selection
        data/train.cfg          — symlink / copy of seed.cfg (used by training scripts)
    """
    print(f"Reading {input_xyz} ...")
    frames = ase_read(str(input_xyz), index=":")
    print(f"  Loaded {len(frames)} configurations.")
    outdir.mkdir(parents=True, exist_ok=True)

    always_include = set(always_include or [])
    always_include_contains = list(always_include_contains or [])
    always_include_contains_except = set(always_include_contains_except or [])
    exclude = set(exclude or [])
    rng = random.Random(rng_seed)

    # Group by config_type
    by_type: dict[str, list] = defaultdict(list)
    for atoms in frames:
        by_type[_get_config_type(atoms)].append(atoms)

    seed_frames: list = []
    pool_frames: list = []

    print(f"\n  {'config_type':<30} {'total':>6} {'→seed':>6} {'→pool':>6}")
    print("  " + "-" * 52)

    for ctype, group in sorted(by_type.items()):
        if ctype in exclude:
            print(f"  {ctype:<30} {len(group):>6}  EXCLUDED")
            continue
        rng.shuffle(group)
        in_always = (
            ctype in always_include
            or (any(s in ctype for s in always_include_contains) and ctype not in always_include_contains_except)
        )
        if in_always:
            n_seed = len(group)
        else:
            cap = max_per_type if max_per_type is not None else len(group)
            n_seed = min(seed_per_type, cap, len(group))
        seed_frames.extend(group[:n_seed])
        pool_frames.extend(group[n_seed:])
        print(f"  {ctype:<30} {len(group):>6} {n_seed:>6} {len(group)-n_seed:>6}")

    print("  " + "-" * 52)
    print(f"  {'TOTAL':<30} {len(frames):>6} {len(seed_frames):>6} {len(pool_frames):>6}")

    write_cfg(seed_frames, outdir / "seed.cfg")
    write_cfg(pool_frames, outdir / "candidate_pool.cfg")

    # train.cfg starts as a copy of seed.cfg; the AL loop appends to it
    import shutil
    shutil.copy(outdir / "seed.cfg", outdir / "train.cfg")
    print(f"  Copied seed.cfg → train.cfg (AL loop will append to train.cfg)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extended XYZ to MLIP-3 CFG")
    parser.add_argument("--input", required=True, help="Input .xyz file")
    parser.add_argument("--outdir", default="data/", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["split", "no-split", "pool"],
        default="split",
        help=(
            "split: random train/val/test split; "
            "no-split: single train.cfg; "
            "pool: stratified seed + candidate_pool for AL (default: split)"
        ),
    )
    # split-mode options
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios for --mode split (default: 0.8 0.1 0.1)",
    )
    # pool-mode options
    parser.add_argument(
        "--seed-per-type",
        type=int,
        default=1,
        help="Configs per config_type to put in seed set (default: 1)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Cap seed selection at N per type even when seed-per-type is higher",
    )
    parser.add_argument(
        "--always-include",
        nargs="+",
        default=[],
        metavar="TYPE",
        help="config_types (exact match) where ALL configs go to seed (default: none)",
    )
    parser.add_argument(
        "--always-include-contains",
        nargs="+",
        default=[],
        metavar="STR",
        help="All config_types whose name contains any of these substrings go entirely to seed",
    )
    parser.add_argument(
        "--always-include-contains-except",
        nargs="+",
        default=[],
        metavar="TYPE",
        help="Exact config_types to exclude from --always-include-contains (fall back to seed-per-type)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="TYPE",
        help="config_types to completely skip (not in seed, not in pool)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.mode == "pool":
        split_pool(
            input_xyz=Path(args.input),
            outdir=outdir,
            seed_per_type=args.seed_per_type,
            max_per_type=args.max_per_type,
            always_include=args.always_include,
            always_include_contains=args.always_include_contains,
            always_include_contains_except=args.always_include_contains_except,
            exclude=args.exclude,
            rng_seed=args.seed,
        )
    elif args.mode == "no-split":
        frames = ase_read(str(args.input), index=":")
        print(f"  Loaded {len(frames)} configurations.")
        outdir.mkdir(parents=True, exist_ok=True)
        write_cfg(frames, outdir / "train.cfg")
    else:  # split
        ratios = tuple(args.split)
        if abs(sum(ratios) - 1.0) > 1e-6:
            parser.error(f"Split ratios must sum to 1.0, got {sum(ratios):.4f}")
        convert(
            input_xyz=Path(args.input),
            outdir=outdir,
            split=ratios,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
