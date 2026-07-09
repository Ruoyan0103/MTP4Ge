"""
Convert LAMMPS dump trajectory to MLIP-3 CFG format for calculate_grade.

Forces are set to zero (not available from MD dump); energy/stress omitted.
The output CFG is suitable for:
    mlp calculate_grade pot.almtp dump.cfg graded.cfg

Usage:
    python src/dump_to_cfg.py --input results/active_learning/iter_001/dump.lammpstrj
    python src/dump_to_cfg.py --input dump.lammpstrj --output candidates.cfg --stride 10
"""

import argparse
from pathlib import Path

# LAMMPS type (1-based) → MLIP type index (0-based), must match convert.py SPECIES_MAP
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


def write_cfg(frames, path):
    """Write frames as bare CFG (zero forces, no energy/stress) for calculate_grade."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS dump trajectory to MLIP-3 CFG for calculate_grade"
    )
    parser.add_argument("--input", required=True, help="LAMMPS dump file (.lammpstrj)")
    parser.add_argument(
        "--output", default=None,
        help="Output CFG file (default: same name as input with .cfg extension)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Write every N-th frame (default: 1 = all frames)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".cfg")

    all_frames = parse_dump(input_path)
    strided = (frame for i, frame in enumerate(all_frames) if i % args.stride == 0)
    count = write_cfg(strided, output_path)
    print(f"Wrote {count} configs → {output_path}")


if __name__ == "__main__":
    main()
