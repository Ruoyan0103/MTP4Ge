"""
Convert between MLIP-3 CFG format and extended XYZ / VASP POSCAR.

Every invocation must state both directions explicitly with --from/--to.
Supported (--from, --to) pairs:
  xyz,    cfg    - extended XYZ training data with energy/forces labels;
                   writes MLIP-3 CFG to --outdir.
  dump,   cfg    - LAMMPS dump trajectory (.lammpstrj); forces are set to
                   zero (not available from MD dump) and energy/stress are
                   omitted. Output CFG is suitable for:
                       mlp calculate_grade pot.almtp dump.cfg graded.cfg
  cfg,    xyz    - MLIP-3 CFG (reverse of xyz,cfg); writes extended XYZ
                   (readable by VESTA/OVITO) to --output.
  cfg,    poscar - MLIP-3 CFG; writes a VASP POSCAR to --output (first
                   frame only if the CFG has more than one block).
  poscar, cfg    - VASP POSCAR/CONTCAR (geometry only, no energy/forces);
                   writes a bare CFG. --input may be a single file or a
                   directory, searched recursively for files named POSCAR
                   or CONTCAR.

Usage:
    python src/utils/convert_format.py --from xyz --to cfg --input <file.xyz> --outdir data/
    python src/utils/convert_format.py --from dump --to cfg --input results/active_learning/iter_001/dump.lammpstrj
    python src/utils/convert_format.py --from dump --to cfg --input dump.lammpstrj --output candidates.cfg --stride 10
    python src/utils/convert_format.py --from cfg --to xyz --input data/train.cfg [--output out.xyz]
    python src/utils/convert_format.py --from cfg --to poscar --input data/train.cfg
    python src/utils/convert_format.py --from poscar --to cfg --input data/AL/Int
    python src/utils/convert_format.py --from poscar --to cfg --input data/AL/Int/B/POSCAR
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except ImportError:
    print("ERROR: ASE not found. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)

from utils import write_bare_cfg


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


def _convert_xyz(input_path: Path, outdir: Path) -> None:
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


# ---------------------------------------------------------------------------
# LAMMPS dump → CFG (zero forces, no energy/stress; for calculate_grade)
# ---------------------------------------------------------------------------

# LAMMPS type (1-based) → MLIP type index (0-based), must match SPECIES_MAP above
_LAMMPS_TO_MLIP = {1: 0}  # Ge


def parse_dump(path):
    """Yield (timestep, cell, positions, types) for each frame in a LAMMPS dump file.

    Supports both orthogonal and triclinic boxes.
    """
    with open(path) as f:
        lines = f.readlines()

    i = 0
    timestep = 0
    n_atoms = 0
    cell = []

    while i < len(lines):
        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2

        elif line == "ITEM: NUMBER OF ATOMS":
            n_atoms = int(lines[i + 1].strip())
            i += 2

        elif line.startswith("ITEM: BOX BOUNDS"):
            tokens = line.split()
            triclinic = "xy" in tokens  # triclinic header: ITEM: BOX BOUNDS xy xz yz ...
            if triclinic:
                xlo_b, xhi_b, xy = map(float, lines[i + 1].split())
                ylo_b, yhi_b, xz = map(float, lines[i + 2].split())
                zlo_b, zhi_b, yz = map(float, lines[i + 3].split())
                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo, zhi = zlo_b, zhi_b
                cell = [
                    [xhi - xlo, 0.0, 0.0],
                    [xy, yhi - ylo, 0.0],
                    [xz, yz, zhi - zlo],
                ]
                i += 4
            else:
                xlo, xhi = map(float, lines[i + 1].split())
                ylo, yhi = map(float, lines[i + 2].split())
                zlo, zhi = map(float, lines[i + 3].split())
                cell = [
                    [xhi - xlo, 0.0, 0.0],
                    [0.0, yhi - ylo, 0.0],
                    [0.0, 0.0, zhi - zlo],
                ]
                i += 4

        elif line.startswith("ITEM: ATOMS"):
            col_names = line.split()[2:]  # skip "ITEM:" and "ATOMS"
            id_col   = col_names.index("id")
            type_col = col_names.index("type")
            x_col    = col_names.index("x")
            y_col    = col_names.index("y")
            z_col    = col_names.index("z")

            positions = []
            types = []
            for _ in range(n_atoms):
                i += 1
                vals = lines[i].split()
                lammps_type = int(vals[type_col])
                types.append(_LAMMPS_TO_MLIP.get(lammps_type, lammps_type - 1))
                positions.append([
                    float(vals[x_col]),
                    float(vals[y_col]),
                    float(vals[z_col]),
                ])

            yield timestep, cell, positions, types
            i += 1

        else:
            i += 1


def write_dump_cfg(frames, path) -> int:
    """Write LAMMPS-dump frames as bare CFG (zero forces, no energy/stress)
    for `mlp calculate_grade`."""
    count = 0
    with open(path, "w") as f:
        for timestep, cell, positions, types in frames:
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"    {len(positions)}\n")
            f.write(" Supercell\n")
            for row in cell:
                f.write(f"    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n")
            f.write(
                " AtomData:  id type       cartes_x      cartes_y"
                "      cartes_z           fx          fy          fz\n"
            )
            for atom_id, (t, pos) in enumerate(zip(types, positions), start=1):
                f.write(
                    f"    {atom_id:8d}  {t:3d}  "
                    f"  {pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}"
                    f"  {'0.000000':>12}  {'0.000000':>12}  {'0.000000':>12}\n"
                )
            f.write(f" Feature   timestep\t {timestep}\n")
            f.write("END_CFG\n\n")
            count += 1
    return count


def _convert_dump(input_path: Path, output: str | None, stride: int) -> None:
    output_path = Path(output) if output else input_path.with_suffix(".cfg")

    all_frames = parse_dump(input_path)
    strided = (frame for i, frame in enumerate(all_frames) if i % stride == 0)
    count = write_dump_cfg(strided, output_path)
    print(f"Wrote {count} configs → {output_path}")


# ---------------------------------------------------------------------------
# CFG → XYZ / POSCAR (reverse of _convert_xyz; for VESTA/OVITO/VASP)
# ---------------------------------------------------------------------------

def _parse_all_cfgs(path: Path) -> list:
    """Parse every CFG block in *path* and return a list of ASE Atoms objects."""
    idx_to_species = {v: k for k, v in SPECIES_MAP.items()}

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

        symbols = [idx_to_species.get(t, f"X{t}") for t in types]
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


def _convert_cfg(cfg_path: Path, output: str | None, poscar: bool) -> None:
    if not cfg_path.exists():
        sys.exit(f"Error: {cfg_path} does not exist")

    atoms_list = _parse_all_cfgs(cfg_path)
    n = len(atoms_list)

    if output:
        out_path = Path(output)
    elif poscar:
        out_path = cfg_path.with_suffix(".POSCAR")
    else:
        out_path = cfg_path.with_suffix(".xyz")

    fmt = "vasp" if poscar else "extxyz"
    if poscar and n > 1:
        print(f"Warning: POSCAR supports only one frame; writing first of {n} blocks.")
        atoms_list = [atoms_list[0]]

    ase_write(str(out_path), atoms_list if len(atoms_list) > 1 else atoms_list[0], format=fmt)
    print(f"Written {n} frame(s) → {out_path}")


# ---------------------------------------------------------------------------
# POSCAR/CONTCAR → CFG (geometry only, no energy/forces)
# ---------------------------------------------------------------------------

def _convert_poscar(path: Path, output: str | None) -> None:
    if path.is_file():
        targets = [path]
    else:
        targets = sorted(
            p for p in path.rglob("*") if p.is_file() and p.name in ("POSCAR", "CONTCAR")
        )
    if not targets:
        sys.exit(f"No POSCAR/CONTCAR files found under {path}")
    if output and len(targets) > 1:
        sys.exit("--output only supported for a single POSCAR/CONTCAR file")

    for src in targets:
        atoms = ase_read(str(src), format="vasp")
        symbols = atoms.get_chemical_symbols()
        unknown = sorted(set(symbols) - SPECIES_MAP.keys())
        if unknown:
            sys.exit(f"{src}: unmapped species {unknown} (add to SPECIES_MAP)")

        cell = atoms.get_cell().array
        positions = atoms.get_positions()
        types = [SPECIES_MAP[s] for s in symbols]

        out_path = Path(output) if output else src.with_name(src.name + ".cfg")
        write_bare_cfg([(cell, positions, types)], out_path)
        print(f"  {src} -> {out_path}")


_SUPPORTED_CONVERSIONS = {
    ("xyz", "cfg"),
    ("dump", "cfg"),
    ("cfg", "xyz"),
    ("cfg", "poscar"),
    ("poscar", "cfg"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert between training/candidate data and MLIP-3 CFG format. "
                     "Both --from and --to must be given explicitly."
    )
    parser.add_argument("--from", dest="from_format", required=True,
                        choices=["xyz", "dump", "cfg", "poscar"],
                        help="Input format: 'xyz' extended XYZ with energy/forces, "
                             "'dump' LAMMPS trajectory dump (zero forces, for "
                             "calculate_grade), 'cfg' MLIP-3 CFG, 'poscar' VASP "
                             "POSCAR/CONTCAR (geometry only)")
    parser.add_argument("--to", dest="to_format", required=True,
                        choices=["cfg", "xyz", "poscar"],
                        help="Output format. Supported --from/--to pairs: "
                             + ", ".join(f"{a}->{b}" for a, b in sorted(_SUPPORTED_CONVERSIONS)))
    parser.add_argument("--input", required=True,
                        help="Input file (.xyz, .lammpstrj, .cfg) or, for "
                             "--from poscar, a POSCAR/CONTCAR file or a "
                             "directory to search recursively")
    parser.add_argument("--outdir", default="data/",
                        help="Output directory (--from xyz only; default: data/)")
    parser.add_argument("--output", default=None,
                        help="Output file (--from dump/cfg/poscar only, and for "
                             "poscar only with a single-file --input); default: "
                             "same name as input with .cfg/.xyz/.POSCAR extension")
    parser.add_argument("--stride", type=int, default=1,
                        help="Write every N-th frame (--from dump only; default: "
                             "1 = all frames)")
    args = parser.parse_args()

    pair = (args.from_format, args.to_format)
    if pair not in _SUPPORTED_CONVERSIONS:
        parser.error(
            f"Unsupported conversion --from {args.from_format} --to {args.to_format}. "
            "Supported pairs: " + ", ".join(f"{a}->{b}" for a, b in sorted(_SUPPORTED_CONVERSIONS))
        )

    input_path = Path(args.input)
    if pair == ("xyz", "cfg"):
        _convert_xyz(input_path, Path(args.outdir))
    elif pair == ("dump", "cfg"):
        _convert_dump(input_path, args.output, args.stride)
    elif pair == ("cfg", "xyz"):
        _convert_cfg(input_path, args.output, poscar=False)
    elif pair == ("cfg", "poscar"):
        _convert_cfg(input_path, args.output, poscar=True)
    else:  # ("poscar", "cfg")
        _convert_poscar(input_path, args.output)


if __name__ == "__main__":
    main()
