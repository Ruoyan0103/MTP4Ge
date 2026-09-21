"""Quasi-static drag test for Ge MTP potential.

Displaces one atom stepwise along [100], [110], [111], [031] crystal directions
and computes ΔE and the parallel force component at each step.

Evaluates via --backend mlp (default: all displacements for a given direction
in a single batched `mlp calculate_efs` call) or --backend lammps (LAMMPS,
pair_style hybrid/overlay mtp nlh, via src/utils/mtp_calculator.py's
MTPLammpsCalculator — one LAMMPS subprocess call per step). The LAMMPS
backend is needed for potentials trained with a radial basis type
`mlp calculate_efs` in the mlip-3-prune build cannot load (fails with "Wrong
radial basis type") — see src/physical_validation/elastic_constant/potential.mod for the same
LAMMPS pair_style workaround used for elastic constants.

Expected results:
  ΔE > 0 everywhere (moving off lattice site costs energy)
  Force near zero at equilibrium, peaks near interstitial sites

Usage:
    python src/physical_validation/quasi_static_drag.py --pot results/potentials/pot.almtp
    python src/physical_validation/quasi_static_drag.py --pot results/potentials/pot.almtp \\
        --a0 5.76 --n-steps 80 --step-size 0.05
    python src/physical_validation/quasi_static_drag.py --pot results/potentials/20.mtp --backend lammps \\
        --a0 5.7547 --lammps "srun /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import calc_efs
from mtp_calculator import MTPLammpsCalculator
from energy_volume import DEFAULT_LAMMPS

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"

REF_DFT_DIR = Path(__file__).parent / "reference" / "quasi_static_displacement_DFT"
DFT_REF_MAP = {
    "[100]": REF_DFT_DIR / "qsd_100.dat",
    "[110]": REF_DFT_DIR / "qsd_110.dat",
    "[111]": REF_DFT_DIR / "qsd_111.dat",
    "[031]": REF_DFT_DIR / "qsd_031.dat",
}


def load_ref_dft(path: Path):
    if not path.exists():
        return None, None, None, None, None
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]


def _dft_max_disp(label: str):
    path = DFT_REF_MAP.get(label)
    if path is None or not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",")
    return float(data[-1, 0])

# Minimum allowed interatomic distance (A) during a drag step; guards only
# against true numerical singularities. The DFT reference dips as low as
# ~1.44 A near the end of each direction, so this must stay well below that
# or the MTP scan gets truncated far short of the DFT drag distance.
MIN_SAFE_DIST = 1.2

# Drag directions: label → fractional Miller index (will be normalised)
DRAG_DIRECTIONS = {
    "[100]": np.array([1.0, 0.0, 0.0]),
    "[110]": np.array([1.0, 1.0, 0.0]),
    "[111]": np.array([1.0, 1.0, 1.0]),
    "[031]": np.array([0.0, 3.0, 1.0]),
}


# ---------------------------------------------------------------------------
# Structure builder
# ---------------------------------------------------------------------------

def build_drag_supercell(a0: float, repeat: tuple = (2, 2, 2)):
    """2×2×2 diamond supercell (64 atoms); return (cell, positions, types)."""
    from ase.build import bulk
    atoms = bulk("Ge", "diamond", a=a0, cubic=True).repeat(repeat)
    return (
        np.array(atoms.get_cell()),
        atoms.get_positions(),
        [0] * len(atoms),
    )


# ---------------------------------------------------------------------------
# EFS multi-block parser — energy + forces for each block
# ---------------------------------------------------------------------------

def _parse_all_efs(path: Path) -> list[tuple[float, np.ndarray]]:
    """Return list of (energy, forces) from all CFG blocks in an EFS file."""
    results = []
    for block in re.split(r"(?=BEGIN_CFG\b)", path.read_text()):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        em = re.search(r"\bEnergy\b\s*\n\s*([-\d.eE+]+)", block)
        if not em:
            continue
        energy = float(em.group(1))

        forces = []
        in_atoms = False
        for line in block.splitlines():
            s = line.strip()
            if s.startswith("AtomData:"):
                in_atoms = True
                continue
            if in_atoms:
                if not s or (s[0].isalpha() and not s[0].isdigit()):
                    in_atoms = False
                    continue
                parts = s.split()
                if len(parts) >= 8:
                    forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
        results.append((energy, np.array(forces)))
    return results


# ---------------------------------------------------------------------------
# Drag scan
# ---------------------------------------------------------------------------

_TYPE_TO_SYMBOL = {0: "Ge"}  # mirrors SPECIES_MAP in src/utils/convert.py


def _write_cfg(configs: list, disp_vals: np.ndarray, path: Path) -> None:
    """Write each drag-step structure to a batch CFG with Feature disp tag (mlp input)."""
    with open(path, "w") as f:
        for d, (cell, positions, types) in zip(disp_vals, configs):
            n = len(positions)
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"    {n}\n")
            f.write(f" Feature disp {d:.6f}\n")
            f.write(" Supercell\n")
            for row in cell:
                f.write(f"    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n")
            f.write(
                " AtomData:  id type       cartes_x      cartes_y      cartes_z\n"
            )
            for i, (t, pos) in enumerate(zip(types, positions), start=1):
                f.write(
                    f"    {i:8d}  {t:3d}    "
                    f"{pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}\n"
                )
            f.write("END_CFG\n\n")


def _write_xyz(configs: list, disp_vals: np.ndarray, path: Path,
               efs_data: list | None = None) -> None:
    """Write drag-step structures as extended XYZ.

    efs_data: list of (energy, forces) from _parse_all_efs; when provided,
    energy and per-atom forces are written into each frame.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    frames = []
    efs_iter = iter(efs_data) if efs_data is not None else None
    for d, (cell, positions, types) in zip(disp_vals, configs):
        symbols = [_TYPE_TO_SYMBOL.get(t, "X") for t in types]
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.info["disp"] = float(d)
        if efs_iter is not None:
            energy, forces = next(efs_iter)
            atoms.info["energy"] = energy
            atoms.arrays["forces"] = forces
        frames.append(atoms)
    ase_write(str(path), frames, format="extxyz")


def _drag_scan_mlp(
    cell, pos, types, drag_idx, direction, n_steps, step_size,
    mlp, pot, outdir, label, max_disp=2.0, dft_max_disp=None,
):
    """Displace atom along direction; return (disp, dE, Fx, Fy, Fz)."""
    uvec = direction / np.linalg.norm(direction)
    pos0 = pos[drag_idx].copy()

    effective_max = dft_max_disp if dft_max_disp is not None else max_disp
    if dft_max_disp is None:
        effective_max = min(effective_max, n_steps * step_size)

    disp_vals_all = np.arange(0.0, effective_max, step_size)
    if disp_vals_all.size == 0 or not np.isclose(disp_vals_all[-1], effective_max):
        disp_vals_all = np.append(disp_vals_all, effective_max)
    print(f"    [{label}] {len(disp_vals_all)} steps x {step_size} A "
          f"(nominal) = {disp_vals_all[-1]:.4f} A")

    configs = []
    cell_inv = np.linalg.inv(cell)

    safe_disp_vals = []
    for d in disp_vals_all:
        new_pos = pos.copy()
        new_pos[drag_idx] = pos0 + uvec * d
        diff = new_pos[drag_idx] - np.delete(new_pos, drag_idx, axis=0)
        frac = diff @ cell_inv
        frac -= np.round(frac)
        min_dist = np.linalg.norm(frac @ cell, axis=1).min()
        if min_dist < MIN_SAFE_DIST:
            print(f"    [{label}] skipping d={d:.3f} A: min dist {min_dist:.3f} A < {MIN_SAFE_DIST} A")
            continue
        configs.append((cell, new_pos, types))
        safe_disp_vals.append(d)
    disp_vals = np.array(safe_disp_vals)

    subdir = outdir / label
    subdir.mkdir(parents=True, exist_ok=True)
    mlp_in = subdir / "_mlp_in.cfg"
    mlp_out = subdir / "_mlp_out.cfg"
    _write_cfg(configs, disp_vals, mlp_in)
    mlp_out.unlink(missing_ok=True)
    rc = calc_efs(mlp, pot, mlp_in, mlp_out, quiet=True)
    if rc != 0:
        raise RuntimeError(f"mlp calculate_efs returned {rc} for {label}")

    efs_data = _parse_all_efs(mlp_out)
    _write_xyz(configs, disp_vals, subdir / "drag_in.xyz")
    _write_xyz(configs, disp_vals, subdir / "drag_out.xyz", efs_data=efs_data)
    energies = np.array([e for e, _ in efs_data])
    Fx = np.array([forces[drag_idx,0] for _,forces in efs_data if len(forces)>drag_idx])
    Fy = np.array([forces[drag_idx,1] for _,forces in efs_data if len(forces)>drag_idx])
    Fz = np.array([forces[drag_idx,2] for _,forces in efs_data if len(forces)>drag_idx])

    n = min(len(disp_vals), len(energies), len(Fx), len(Fy), len(Fz))
    return disp_vals[:n], (energies[:n] - energies[0]), Fx[:n], Fy[:n], Fz[:n]


def _drag_scan_lammps(
    cell, pos, types, drag_idx, direction, n_steps, step_size,
    lammps_cmd, pot, outdir, label, max_disp=2.0, dft_max_disp=None,
):
    """Same as _drag_scan_mlp but evaluates each step via LAMMPS
    (MTPLammpsCalculator, pair_style hybrid/overlay mtp nlh)."""
    from ase import Atoms

    uvec = direction / np.linalg.norm(direction)
    pos0 = pos[drag_idx].copy()

    effective_max = dft_max_disp if dft_max_disp is not None else max_disp
    if dft_max_disp is None:
        effective_max = min(effective_max, n_steps * step_size)

    disp_vals_all = np.arange(0.0, effective_max, step_size)
    if disp_vals_all.size == 0 or not np.isclose(disp_vals_all[-1], effective_max):
        disp_vals_all = np.append(disp_vals_all, effective_max)
    print(f"    [{label}] {len(disp_vals_all)} steps x {step_size} A "
          f"(nominal) = {disp_vals_all[-1]:.4f} A")

    configs = []
    cell_inv = np.linalg.inv(cell)

    safe_disp_vals = []
    for d in disp_vals_all:
        new_pos = pos.copy()
        new_pos[drag_idx] = pos0 + uvec * d
        diff = new_pos[drag_idx] - np.delete(new_pos, drag_idx, axis=0)
        frac = diff @ cell_inv
        frac -= np.round(frac)
        min_dist = np.linalg.norm(frac @ cell, axis=1).min()
        if min_dist < MIN_SAFE_DIST:
            print(f"    [{label}] skipping d={d:.3f} A: min dist {min_dist:.3f} A < {MIN_SAFE_DIST} A")
            continue
        configs.append((cell, new_pos, types))
        safe_disp_vals.append(d)
    disp_vals = np.array(safe_disp_vals)

    subdir = outdir / label
    subdir.mkdir(parents=True, exist_ok=True)

    efs_data = []
    for i, (c, p, t) in enumerate(configs):
        symbols = [_TYPE_TO_SYMBOL.get(ti, "X") for ti in t]
        atoms = Atoms(symbols=symbols, positions=p, cell=c, pbc=True)
        atoms.calc = MTPLammpsCalculator(lammps_cmd=lammps_cmd, pot=pot)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        efs_data.append((energy, forces))
        print(f"    [{label}] [{i + 1}/{len(configs)}] d={disp_vals[i]:.3f} A  E={energy:.6f} eV")

    _write_xyz(configs, disp_vals, subdir / "drag_in.xyz")
    _write_xyz(configs, disp_vals, subdir / "drag_out.xyz", efs_data=efs_data)
    energies = np.array([e for e, _ in efs_data])
    Fx = np.array([forces[drag_idx, 0] for _, forces in efs_data if len(forces) > drag_idx])
    Fy = np.array([forces[drag_idx, 1] for _, forces in efs_data if len(forces) > drag_idx])
    Fz = np.array([forces[drag_idx, 2] for _, forces in efs_data if len(forces) > drag_idx])

    n = min(len(disp_vals), len(energies), len(Fx), len(Fy), len(Fz))
    return disp_vals[:n], (energies[:n] - energies[0]), Fx[:n], Fy[:n], Fz[:n]


def run(
    pot: str,
    outdir: Path,
    a0: float,
    n_steps: int,
    step_size: float,
    max_disp: float = 2.0,
    backend: str = "mlp",
    mlp: str | None = None,
    lammps_cmd: str = DEFAULT_LAMMPS,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    scan_fn = _drag_scan_mlp if backend == "mlp" else _drag_scan_lammps
    engine = mlp if backend == "mlp" else lammps_cmd

    cell, pos, types = build_drag_supercell(a0)
    n_atoms = len(pos)
    # [111] uses atom 1 (B-sublattice at a/4,a/4,a/4): bonded neighbours are NOT
    # in [111], giving open space toward the next A-sublattice atom.
    # All other directions use atom 0 (A-sublattice at origin).
    DRAG_ATOM = {"[100]": 0, "[110]": 0, "[111]": 1, "[031]": 0}
    print(f"  {n_atoms}-atom 2×2×2 supercell  |  a0 = {a0:.4f} Å  |  {n_steps} steps × {step_size} Å")

    all_results = {}
    for label, direction in DRAG_DIRECTIONS.items():
        drag_idx = DRAG_ATOM[label]
        print(f"\n  [{label}]  direction = {direction / np.linalg.norm(direction)}"
              f"  atom {drag_idx} @ {pos[drag_idx]}")
        dft_max = _dft_max_disp(label)
        disp, dE, Fx, Fy, Fz = scan_fn(
            cell, pos, types, drag_idx, direction, n_steps, step_size,
            engine, pot, outdir, label.strip("[]"),
            max_disp=max_disp, dft_max_disp=dft_max,
        )
        all_results[label] = (disp, dE, Fx, Fy, Fz)
        i_max = int(np.argmax(dE))
        print(f"  Max ΔE = {dE[i_max]:.4f} eV at d = {disp[i_max]:.2f} Å")

    # --- Write data ---
    txt = outdir / "qsd_data.txt"
    with open(txt, "w") as f:
        f.write(f"# Quasi-static drag  |  pot: {pot}  a0={a0:.4f} Å\n")
        f.write(f"# {n_atoms} atoms  |  {n_steps} steps × {step_size} Å\n\n")
        for label, (disp, dE, Fx, Fy, Fz) in all_results.items():
            f.write(f"# direction {label}\n")
            f.write("# disp[A]  dE[eV]  Fx[eV/A]  Fy[eV/A]  Fz[eV/A]\n")
            for d, de, x, y, z in zip(disp, dE, Fx, Fy, Fz):
                f.write(f"  {d:.4f}  {de:.6f}  {x:.6f}  {y:.6f}  {z:.6f}\n")
            f.write("\n")
    print(f"\n  Data written to {txt}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_dir = len(all_results)
        fig, axes = plt.subplots(2, n_dir, figsize=(4 * n_dir, 7))
        if n_dir == 1:
            axes = axes[:, np.newaxis]

        fc = {"Fx": "tab:red", "Fy": "tab:blue", "Fz": "tab:green"}

        for col, (label, (disp, dE, Fx, Fy, Fz)) in enumerate(all_results.items()):
            ax_e = axes[0, col]
            ax_f = axes[1, col]

            # --- Energy ---
            ax_e.plot(disp, dE, '-o', color="tab:blue", lw=1.5, label="MTP")
            ax_e.axhline(0, color="gray", lw=0.8, ls="--")

            dft_path = DFT_REF_MAP.get(label)
            if dft_path:
                rd = load_ref_dft(dft_path)
                if rd[0] is not None:
                    ax_e.plot(rd[0], rd[1], "s", color="black", ms=3,
                              alpha=0.6, label="DFT", zorder=0)

            ax_e.set_ylabel("dE (eV)" if col == 0 else "")
            ax_e.set_title(label, fontsize=11)
            ax_e.set_xlabel("")
            ax_e.legend(fontsize=7, markerscale=0.8)

            # --- Force components ---
            mtp_f = {"Fx": Fx, "Fy": Fy, "Fz": Fz}
            for fl in ["Fx", "Fy", "Fz"]:
                ax_f.plot(disp, mtp_f[fl], '-o', color=fc[fl], lw=1.0, label=f"MTP {fl}")
            ax_f.axhline(0, color="gray", lw=0.8, ls="--")

            if dft_path:
                rd = load_ref_dft(dft_path)
                if rd[0] is not None:
                    dft_f = {"Fx": rd[2], "Fy": rd[3], "Fz": rd[4]}
                    for fl in ["Fx", "Fy", "Fz"]:
                        ax_f.plot(rd[0], dft_f[fl], "s", color=fc[fl], ms=2.5, 
                                  alpha=0.5, zorder=0)

            ax_f.set_xlabel("Displacement (A)")
            ax_f.set_ylabel("Force (eV/A)" if col == 0 else "")
            ax_f.legend(fontsize=6, markerscale=0.8, ncol=2)

        backend_tag = "" if backend == "mlp" else " (LAMMPS)"
        fig.suptitle(f"Quasi-static drag - Ge MTP{backend_tag}  (a0={a0:.3f} A)", fontsize=13)
        fig.tight_layout()
        png = outdir / "qsd.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png}")
    except ImportError as e:
        print(f"  Matplotlib not available ({e}); skipping plot.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quasi-static drag test for Ge MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--a0", type=float, default=5.76,
                        help="Lattice constant Å (default: 5.76)")
    parser.add_argument("--n-steps", type=int, default=15,
                        help="Number of drag steps per direction (default: 15)")
    parser.add_argument("--step-size", type=float, default=0.3,
                        help="Displacement per step in Å (default: 0.3)")
    parser.add_argument("--max-disp", type=float, default=2.0,
                        help="Max displacement per direction in Å (default: 2.0)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/tests/<pot's parent dir "
                             "name>/quasi_static_drag, e.g. --pot results/potentials/pot_660277/pot.almtp "
                             "-> results/tests/pot_660277/quasi_static_drag)")
    parser.add_argument("--backend", choices=["mlp", "lammps"], default="mlp",
                        help="Evaluation engine: 'mlp' calculate_efs (default) or "
                             "'lammps' (pair_style mtp+nlh — for potentials 'mlp' can't load)")
    parser.add_argument("--mlp", default=MLP_DEFAULT,
                        help="mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(--backend lammps only; default: {DEFAULT_LAMMPS})")
    args = parser.parse_args()

    if args.outdir is None:
        run_name = Path(args.pot).resolve().parent.name
        outdir = Path("results/tests") / run_name / "quasi_static_drag"
    else:
        outdir = Path(args.outdir)

    run(
        pot=args.pot,
        outdir=outdir,
        a0=args.a0,
        n_steps=args.n_steps,
        step_size=args.step_size,
        max_disp=args.max_disp,
        backend=args.backend,
        mlp=args.mlp,
        lammps_cmd=args.lammps,
    )


if __name__ == "__main__":
    main()
