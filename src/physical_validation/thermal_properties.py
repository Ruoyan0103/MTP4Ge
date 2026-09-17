"""
Quasi-harmonic approximation (QHA) quality test for MTP potentials.

Reproduces GAP Germanium paper Fig. 3(c)(d)(e) for diamond cubic Ge:
  (c) Linear thermal expansion coefficient alphaL (1/K)
  (d) Isobaric heat capacity Cp (J/K/mol)
  (e) Bulk modulus B(T) (GPa)

Force constants are evaluated at N volumes, either via `mlp calculate_efs`
(--backend mlp, default) or via LAMMPS (--backend lammps: pair_style
hybrid/overlay mtp nlh — static per-volume energies via
src/physical_validation/energy_volume.py's single-point evaluator, config/lammps/energy_volume.in;
per-volume force constants via src/physical_validation/phonon_dispersion.py's single-point
force evaluator, config/lammps/phonon_dispersion_mtp.in). The LAMMPS backend
is needed for potentials trained with a radial basis type that
`mlp calculate_efs` in the mlip-3-prune build cannot load (fails with
"Wrong radial basis type") — see src/physical_validation/elastic_constant/potential.mod for the
same LAMMPS pair_style workaround used for elastic constants.
PhonopyQHA fits the free-energy surface to extract thermodynamic properties
vs temperature.

Usage:
    python src/physical_validation/thermal_properties.py --pot results/potentials/pot_al.almtp
    python src/physical_validation/thermal_properties.py --pot results/potentials/pot_al.almtp \\
        --supercell 2 --n-volumes 11 --t-max 1000 --t-step 10
    python src/physical_validation/thermal_properties.py --pot results/potentials/20.mtp --backend lammps \\
        --supercell 2 --n-volumes 11 --t-max 1000 --t-step 10 \\
        --lammps "srun /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import calc_efs, write_bare_cfg
from energy_volume import DEFAULT_LAMMPS, _run_lammps_energy
from phonon_dispersion import _run_lammps_forces_nlh

DEFAULT_TRAIN_CONFIG = "config/training.yaml"


# ---------------------------------------------------------------------------
# Structure conversion helpers (duplicated from phonon_dispersion.py)
# ---------------------------------------------------------------------------

def _ase_to_phonopy(atoms):
    """Convert ASE Atoms -> PhonopyAtoms."""
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(
        symbols=list(atoms.get_chemical_symbols()),
        cell=atoms.get_cell().array.copy(),
        scaled_positions=atoms.get_scaled_positions(),
    )


def _phonopy_to_tuple(ph_atoms):
    """Convert PhonopyAtoms -> (cell, positions, types) for write_bare_cfg."""
    symbols = ph_atoms.symbols
    unique = sorted(set(symbols))
    sym_idx = {s: i for i, s in enumerate(unique)}
    types = [sym_idx[s] for s in symbols]
    return ph_atoms.cell.copy(), ph_atoms.positions.copy(), types


# ---------------------------------------------------------------------------
# CFG parsers
# ---------------------------------------------------------------------------

def _parse_forces_from_cfg(path: Path) -> list[np.ndarray]:
    """Extract per-atom forces from an EFS CFG output file."""
    forces_list = []
    text = path.read_text()
    for block in re.split(r"(?=BEGIN_CFG\b)", text):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        header_m = re.search(r"AtomData:\s*(.*)", block)
        if not header_m:
            continue
        cols = header_m.group(1).split()
        try:
            ix = cols.index("fx")
            iy = cols.index("fy")
            iz = cols.index("fz")
        except ValueError:
            continue
        atom_forces = []
        in_atoms = False
        for line in block.splitlines():
            s = line.strip()
            if "AtomData:" in s:
                in_atoms = True
                continue
            if in_atoms:
                if not s or re.match(
                    r"(Feature|Energy|PlusStress|BEGIN_CFG|END_CFG)", s
                ):
                    break
                parts = s.split()
                if len(parts) > max(ix, iy, iz):
                    atom_forces.append(
                        [float(parts[ix]), float(parts[iy]), float(parts[iz])]
                    )
        if atom_forces:
            forces_list.append(np.array(atom_forces))
    return forces_list


def _parse_energies_from_cfg(path: Path) -> list[float]:
    """Extract the Energy value from each CFG block."""
    energies = []
    for block in re.split(r"(?=BEGIN_CFG\b)", path.read_text()):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        m = re.search(r"^\s*Energy\s*\n\s*([-\d.eE+]+)", block, re.MULTILINE)
        if m:
            energies.append(float(m.group(1)))
    return energies


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_thermal(
    te_temps: np.ndarray,
    alpha_V: np.ndarray,
    cp_temps: np.ndarray,
    cp_vals: np.ndarray,
    bm_temps: np.ndarray,
    bm_vals: np.ndarray,
    outdir: Path,
    ref_dir: Path,
) -> None:
    """Save thermal_properties.png: 3-panel figure matching combined.py style."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    plt.rcParams['font.family'] = 'DejaVu Serif'
    exp_dir = ref_dir / "thermal_properties_EXP"
    dft_sub_dir = ref_dir / "thermal_properties_DFT"

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

    def _load_ref(path, skiprows=0):
        if not path.exists():
            return None
        try:
            return np.loadtxt(path, skiprows=skiprows)
        except Exception:
            return None

    # ---- (c) Linear thermal expansion coefficient ----
    ax = axes[0]
    dft_te = _load_ref(dft_sub_dir / 'thermal_expansion_coefficient.dat')
    if dft_te is not None:
        ax.plot(dft_te[:, 0], dft_te[:, 1] / 3.0,
                label='DFT', color='black', linewidth=1)
    ax.plot(te_temps, alpha_V / 3.0, label='MTP', color='red', linewidth=1)
    exp_te = _load_ref(exp_dir / 'thermal_expansion_Reeber1996.dat', skiprows=1)
    if exp_te is not None:
        ax.scatter(exp_te[:, 0], exp_te[:, 1] / 1e6, marker='*',
                   facecolors='none', edgecolors='blue',
                   label='Exp. Reeber et al.', s=50, linewidth=1)
    ax.legend(loc='best', framealpha=1, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$\alpha_L$' + r' (K$^{-1}$)', fontsize=10)
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes,
            fontsize=14, va='top', ha='left')

    # ---- (d) Isobaric heat capacity ----
    ax = axes[1]
    dft_cp = _load_ref(dft_sub_dir / 'heat_capacity_vs_temperature.dat')
    if dft_cp is not None:
        ax.plot(dft_cp[:, 0], dft_cp[:, 1] / 2.0,
                label='DFT', color='black', linewidth=1)
    ax.plot(cp_temps, cp_vals, label='MTP', color='red', linewidth=1)
    cp_exp_cfg = [
        ('heat_capacity_Estermann1952.dat', 'Exp. Estermann et al.', 'blue',   '^'),
        ('heat_capacity_Flubacher1953.dat', 'Exp. Flubacher et al.', 'green',  's'),
        ('heat_capacity_Leadbetter1969.dat', 'Exp. Leadbetter et al.', 'orange', 'o'),
    ]
    for fname, lbl, col, mrk in cp_exp_cfg:
        d = _load_ref(exp_dir / fname, skiprows=1)
        if d is not None:
            ax.scatter(d[:, 0], d[:, 1], marker=mrk,
                       facecolors='none', edgecolors=col,
                       label=lbl, s=50, linewidth=1)
    ax.legend(loc='best', framealpha=1, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$C_{p}$' + r' (J/mol/K)', fontsize=10)
    ax.text(0.05, 0.95, '(d)', transform=ax.transAxes,
            fontsize=14, va='top', ha='left')

    # ---- (e) Bulk modulus ----
    ax = axes[2]
    dft_bm = _load_ref(dft_sub_dir / 'bulk_modulus_vs_temperature.dat')
    if dft_bm is not None:
        ax.plot(dft_bm[:, 0], dft_bm[:, 1],
                label='DFT', color='black', linewidth=1)
    ax.plot(bm_temps, bm_vals, label='MTP', color='red', linewidth=1)
    bm_exp = _load_ref(exp_dir / 'bulk_modulus_Yin1982.dat', skiprows=1)
    if bm_exp is not None:
        if len(bm_exp.shape) == 1:
            temp_bm, val_bm = 77, 77
        else:
            temp_bm, val_bm = bm_exp[:, 0], bm_exp[:, 1]
        # ax.scatter(temp_bm, val_bm, marker='*',
        #            facecolors='none', edgecolors='blue',
        #            label='Exp. Yin et al.', s=50, linewidth=1)
    ax.legend(loc='best', framealpha=1, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=5)
    ax.tick_params(axis='both', which='minor', length=3)
    ax.grid(linestyle='--', alpha=0.7)
    ax.set_ylabel(r'$B$ (GPa)', fontsize=10)
    ax.text(0.05, 0.95, '(e)', transform=ax.transAxes,
            fontsize=14, va='top', ha='left')

    axes[-1].set_xlabel('Temperature (K)', fontsize=12)
    plt.tight_layout()

    out_path = outdir / "thermal_properties.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    outdir: Path,
    alat: float,
    supercell_size: int,
    n_volumes: int,
    vol_range_start: float,
    vol_range_end: float,
    t_max: float,
    t_step: int,
    dos_mesh: int,
    displacement: float,
    backend: str = "mlp",
    mlp: str | None = None,
    lammps_cmd: str = DEFAULT_LAMMPS,
) -> None:
    from ase.build import bulk
    from phonopy import Phonopy, PhonopyQHA

    outdir.mkdir(parents=True, exist_ok=True)

    vol_rates = np.linspace(vol_range_start, vol_range_end, n_volumes)
    lat_rates = (1 + vol_rates) ** (1 / 3) - 1
    S = supercell_size
    print(f"  QHA: {n_volumes} volumes, V = V0 * (1 + r) for V in "
          f"[{vol_range_start}, {vol_range_end}]")
    print(f"  Supercell: {S}x{S}x{S} cubic = {S**3 * 8} atoms per displaced config")

    # Build unit cells and collect volumes (conventional cubic cell, 8 atoms,
    # so the supercell box is cubic and its edge scales cleanly with S)
    unit_cells = [
        bulk("Ge", crystalstructure="diamond", a=alat * (1 + r), cubic=True)
        for r in lat_rates
    ]
    volumes = [float(uc.get_volume()) for uc in unit_cells]

    # --- Static energies ---
    if backend == "mlp":
        uc_cfg = outdir / "unit_cells.cfg"
        uc_efs = outdir / "unit_cells_efs.cfg"
        if uc_efs.exists():
            print(f"  Found existing unit_cells_efs.cfg — skipping static energy calculation")
        else:
            uc_configs = [
                (uc.get_cell().array.copy(), uc.get_positions().copy(), [0] * len(uc))
                for uc in unit_cells
            ]
            write_bare_cfg(uc_configs, uc_cfg)
            rc = calc_efs(mlp, pot, uc_cfg, uc_efs)
            if rc != 0:
                print(f"  Warning: mlp calculate_efs returned {rc} for unit cells")
            if not uc_efs.exists():
                print("  Error: no EFS output for unit cells.", file=sys.stderr)
                sys.exit(1)

        energies = _parse_energies_from_cfg(uc_efs)
        if len(energies) != n_volumes:
            print(
                f"  Error: expected {n_volumes} energies, got {len(energies)}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        energy_dir = outdir / "static_energies"
        energies = []
        for i, uc in enumerate(unit_cells):
            cell = uc.get_cell().array.copy()
            pos = uc.get_positions().copy()
            e = _run_lammps_energy(lammps_cmd, pot, cell, pos, energy_dir / f"vol_{i:02d}")
            energies.append(e)
    print(f"  Static energies: {energies[0]:.4f} ... {energies[-1]:.4f} eV")

    # --- Per-volume force constants and thermal properties ---
    fe_list: list[list[float]] = []
    cv_list: list[list[float]] = []
    entropy_list: list[list[float]] = []
    temps = None

    for i, (rate, uc) in enumerate(zip(lat_rates, unit_cells)):
        vol_dir = outdir / f"vol_{i:02d}"
        vol_dir.mkdir(exist_ok=True)
        li = alat * (1 + rate)

        unitcell = _ase_to_phonopy(uc)
        phonon = Phonopy(
            unitcell,
            supercell_matrix=[[S, 0, 0], [0, S, 0], [0, 0, S]],
            primitive_matrix=[[0, 0.5, 0.5],
                                [0.5, 0, 0.5],
                                [0.5, 0.5, 0]],
            calculator='lammps'
        )
        phonon.generate_displacements(distance=displacement)
        supercells = phonon.supercells_with_displacements

        if backend == "mlp":
            in_cfg = vol_dir / "displaced.cfg"
            out_cfg = vol_dir / "displaced_efs.cfg"
            if out_cfg.exists():
                print(f"  vol {i:02d} (a={li:.4f} A): found cached displaced_efs.cfg")
            else:
                configs = [_phonopy_to_tuple(sc) for sc in supercells]
                write_bare_cfg(configs, in_cfg)
                rc = calc_efs(mlp, pot, in_cfg, out_cfg)
                if rc != 0:
                    print(f"  Warning: mlp returned {rc} for vol {i:02d}")
                if not out_cfg.exists():
                    print(f"  Error: no EFS output for vol {i:02d}.", file=sys.stderr)
                    sys.exit(1)
                print(f"  vol {i:02d} (a={li:.4f} A) (vol={li**3:4f}): forces computed")
            forces_list = _parse_forces_from_cfg(out_cfg)
        else:
            cache = vol_dir / "lammps_forces.npz"
            if cache.exists():
                print(f"  vol {i:02d} (a={li:.4f} A): found cached {cache.name}")
                data = np.load(cache)
                forces_list = [data[f"arr_{j}"] for j in range(len(supercells))]
            else:
                forces_list = _run_lammps_forces_nlh(lammps_cmd, pot, supercells, vol_dir)
                np.savez(cache, *forces_list)
                print(f"  vol {i:02d} (a={li:.4f} A) (vol={li**3:4f}): forces computed")

        if len(forces_list) != len(supercells):
            print(
                f"  Error: vol {i:02d}: expected {len(supercells)} force blocks, "
                f"got {len(forces_list)}",
                file=sys.stderr,
            )
            sys.exit(1)

        phonon.forces = forces_list
        phonon.produce_force_constants()

        phonon.run_mesh([dos_mesh, dos_mesh, dos_mesh], is_gamma_center=True)
        phonon.run_thermal_properties(t_step=t_step, t_max=int(t_max), t_min=0)
        tp = phonon.get_thermal_properties_dict()

        if temps is None:
            temps = np.array(tp["temperatures"])
            n_atoms_prim = len(phonon.primitive)

        fe_list.append(list(tp["free_energy"]))       # kJ/mol
        cv_list.append(list(tp["heat_capacity"]))     # J/K/mol
        entropy_list.append(list(tp["entropy"]))      # J/K/mol

    print(f"  All {n_volumes} volumes done.")

    # Normalize to per-atom molar quantities matching the experimental/DFT-
    # reference convention. Static energies/volumes come from the conventional
    # cubic cell (8 atoms), but phonopy's thermal properties are reported per
    # mole of the *primitive* cell it reduced to via primitive_matrix (2 atoms
    # for diamond Ge) -- these two must be normalized by different atom counts.
    n_atoms_per_cell = len(unit_cells[0])
    fe_list = [[v / n_atoms_prim for v in row] for row in fe_list]
    cv_list = [[v / n_atoms_prim for v in row] for row in cv_list]
    entropy_list = [[v / n_atoms_prim for v in row] for row in entropy_list]
    energies = [e / n_atoms_per_cell for e in energies]
    volumes = [v / n_atoms_per_cell for v in volumes]

    # --- PhonopyQHA ---
    print("  Running PhonopyQHA ...")
    qha = PhonopyQHA(
        volumes=volumes,
        electronic_energies=energies,
        temperatures=temps,
        free_energy=np.array(fe_list).T,     # shape (T, V), kJ/mol
        cv=np.array(cv_list).T,              # shape (T, V), J/K/mol
        entropy=np.array(entropy_list).T,    # shape (T, V), J/K/mol
        t_max=int(t_max),
        eos="vinet",
        verbose=False,
    )

    te_file = outdir / "thermal_expansion.dat"
    cp_file = outdir / "heat_capacity.dat"
    bm_file = outdir / "bulk_modulus.dat"
    ev_file = outdir / "entropy-volume.dat"
    cvv_file = outdir / "Cv-volume.dat"
    dsdvt_file = outdir / "dsdv-temperature.dat"

    qha.write_thermal_expansion(str(te_file))
    qha.write_heat_capacity_P_polyfit(
        str(cp_file),
        filename_ev=str(ev_file),
        filename_cvv=str(cvv_file),
        filename_dsdvt=str(dsdvt_file),
    )
    qha.write_bulk_modulus_temperature(str(bm_file))
    print(f"  Written: {te_file.name}, {cp_file.name}, {bm_file.name}, "
          f"{ev_file.name}, {cvv_file.name}, {dsdvt_file.name}")

    fe_volume_file = outdir / "free_energy-volume.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        qha.plot_helmholtz_volume().savefig(str(fe_volume_file))
        plt.close("all")
        print(f"  Written: {fe_volume_file.name}")
    except ImportError:
        print("  matplotlib not available — skipping free energy-volume plot")

    # --- Parse output files ---
    def _read_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(path, comments="#")
        return data[:, 0], data[:, 1]

    te_temps, alpha_V = _read_dat(te_file)
    cp_temps, cp_vals = _read_dat(cp_file)
    bm_temps, bm_vals = _read_dat(bm_file)

    # Volumetric -> linear expansion coefficient (for sanity check below)
    alpha_L = alpha_V / 3.0

    # --- Plot ---
    _plot_thermal(
        te_temps, alpha_V,
        cp_temps, cp_vals,
        bm_temps, bm_vals,
        outdir,
        Path("src/physical_validation/reference"),
    )

    # --- Sanity check at 300 K ---
    idx300 = int(np.argmin(np.abs(te_temps - 300.0)))
    print(f"\n  At T = {te_temps[idx300]:.0f} K:")
    print(f"    alphaL = {alpha_L[idx300]*1e6:.2f} x 10^-6 K^-1  "
          f"(Ge exp: ~5.9 x 10^-6 K^-1)")
    if idx300 < len(cp_vals):
        print(f"    Cp     = {cp_vals[idx300]:.2f} J/K/mol/atom  "
              f"(Ge exp: ~23 J/K/mol/atom; Dulong-Petit 3R = 24.9)")
    if idx300 < len(bm_vals):
        print(f"    B      = {bm_vals[idx300]:.2f} GPa  (Ge exp: ~70-75 GPa)")
    print(f"  Results in {outdir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "QHA thermal properties test for MTP potential (Ge diamond): "
            "alphaL, Cp, B(T) — Fig. 3(c)(d)(e)"
        )
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--alat", type=float, default=5.76,
                        help="Equilibrium lattice constant in A (default: 5.76)")
    parser.add_argument("--supercell", type=int, default=1,
                        help="Primitive supercell size S for force constant calc (default: 1)")
    parser.add_argument("--n-volumes", type=int, default=11,
                        help="Number of volume grid points for QHA (default: 11)")
    parser.add_argument("--vol-range-start", type=float, default=-0.15,
                        help="Min volume scale rate (default: -0.15 = -15%%)")
    parser.add_argument("--vol-range-end", type=float, default=0.15,
                        help="Max volume scale rate (default: +0.15 = +15%%)")
    parser.add_argument("--t-max", type=float, default=1000.0,
                        help="Max temperature in K (default: 1000)")
    parser.add_argument("--t-step", type=int, default=10,
                        help="Temperature step in K (default: 10)")
    parser.add_argument("--dos-mesh", type=int, default=30,
                        help="Gamma-centered q-mesh for thermal integration (default: 30)")
    parser.add_argument("--displacement", type=float, default=0.01,
                        help="Phonopy displacement distance in A (default: 0.01)")
    parser.add_argument("--outdir", default="results/tests/thermal_properties",
                        help="Output directory")
    parser.add_argument("--backend", choices=["mlp", "lammps"], default="mlp",
                        help="Evaluation engine: 'mlp' calculate_efs (default) or "
                             "'lammps' (pair_style mtp+nlh — for potentials 'mlp' can't load)")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML config (provides mlp_binary; --backend mlp only)")
    parser.add_argument("--mlp", default=None,
                        help="Override mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(--backend lammps only; default: {DEFAULT_LAMMPS})")
    args = parser.parse_args()

    mlp = args.mlp
    if args.backend == "mlp" and mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        alat=args.alat,
        supercell_size=args.supercell,
        n_volumes=args.n_volumes,
        vol_range_start=args.vol_range_start,
        vol_range_end=args.vol_range_end,
        t_max=args.t_max,
        t_step=args.t_step,
        dos_mesh=args.dos_mesh,
        displacement=args.displacement,
        backend=args.backend,
        mlp=mlp,
        lammps_cmd=args.lammps,
    )


if __name__ == "__main__":
    main()
