"""Multi-phase energy-volume curves for Ge crystal phases.

Computes E vs V/atom for 8 Ge crystal phases (diamond, hd, hcp, fcc, bcc,
beta-Sn, bc8, st12) using MTP, then overlays DFT reference data from the
ilearn reference dataset for direct comparison.

Structure definitions match the ilearn DFT reference exactly:
  /scratch/project_2012355/Paper_3/ilearn/dft-files/structures/bulk/*.py

DFT reference data from:
  /scratch/project_2012355/Paper_3/ilearn/code4plots/Fig3/Fig2/*/DFT

Evaluates energy via `mlp calculate_efs` (--backend mlp, default) or LAMMPS
(--backend lammps: pair_style hybrid/overlay mtp nlh,
config/lammps/energy_volume.in). The LAMMPS backend is needed for potentials
trained with a radial basis type that `mlp calculate_efs` in the
mlip-3-prune build cannot load (fails with "Wrong radial basis type") — see
src/physical_validation/elastic_constant/potential.mod for the same LAMMPS pair_style
workaround used for elastic constants.

Usage:
    python src/physical_validation/multiphase_ev.py --pot results/potentials/pot.almtp
    python src/physical_validation/multiphase_ev.py --pot results/potentials/pot.almtp \
        --outdir results/tests/multiphase_ev --n-volumes 21 --vol-range 0.40
    python src/physical_validation/multiphase_ev.py --pot results/potentials/20.mtp --backend lammps \
        --lammps "srun /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import _mlp_env, calc_efs, write_bare_cfg
from energy_volume import DEFAULT_LAMMPS, _run_lammps_energy

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"

# Default path to DFT reference data (local copy in src/physical_validation/reference/)
DFT_DATA_DEFAULT = str(
    Path(__file__).resolve().parent / "reference" / "multiphase_ev_DFT"
)


# ---------------------------------------------------------------------------
# Structure builders — return (cell [3×3], positions [N×3], types [list])
# Parameters match the ilearn DFT reference exactly.
# ---------------------------------------------------------------------------

def build_phases() -> dict:
    """Build 8 Ge crystal phases at reference volumes.

    Lattice constants and internal parameters are taken from the ilearn
    DFT reference (dft-files/structures/bulk/*.py) so that MTP and DFT
    E-V curves are directly comparable.
    """
    from ase import Atoms

    phases = {}

    # --- diamond (Fd-3m) — a0 = 5.7620 ---
    from ase.build import bulk
    atoms = bulk("Ge", "diamond", a=5.7620)
    phases["diamond"] = (
        np.array(atoms.get_cell()),
        atoms.get_positions(),
        [0] * len(atoms),
    )

    # --- hexagonal diamond / lonsdaleite (P6_3/mmc) ---
    # a = 4.0561, c/a = 1.6496, z = 0.06281, 4 atoms
    a_hd = 4.0561
    c_hd = a_hd * 1.6496
    z = 0.06281
    hd_lattice = np.array([
        [0.5 * a_hd, -0.5 * np.sqrt(3) * a_hd, 0.0],
        [0.5 * a_hd,  0.5 * np.sqrt(3) * a_hd, 0.0],
        [0.0, 0.0, c_hd],
    ])
    hd_basis = np.array([
        [0.5 * a_hd,  a_hd / np.sqrt(12),  z * c_hd],
        [0.5 * a_hd, -a_hd / np.sqrt(12),  (0.5 + z) * c_hd],
        [0.5 * a_hd,  a_hd / np.sqrt(12),  (0.5 - z) * c_hd],
        [0.5 * a_hd, -a_hd / np.sqrt(12), -z * c_hd],
    ])
    phases["hd"] = (hd_lattice, hd_basis, [0] * 4)

    # --- hcp (P6_3/mmc) — a = 3.0118, ideal c/a ---
    atoms_hcp = bulk("Ge", "hcp", a=3.0118)
    phases["hcp"] = (
        np.array(atoms_hcp.get_cell()),
        atoms_hcp.get_positions(),
        [0] * len(atoms_hcp),
    )

    # --- fcc (Fm-3m) — a = 4.2749 ---
    atoms_fcc = bulk("Ge", "fcc", a=4.2749)
    phases["fcc"] = (
        np.array(atoms_fcc.get_cell()),
        atoms_fcc.get_positions(),
        [0] * len(atoms_fcc),
    )

    # --- bcc (Im-3m) — a = 3.3858 ---
    atoms_bcc = bulk("Ge", "bcc", a=3.3858)
    phases["bcc"] = (
        np.array(atoms_bcc.get_cell()),
        atoms_bcc.get_positions(),
        [0] * len(atoms_bcc),
    )

    # --- beta-Sn / Ge-II (I4_1/amd) ---
    # a = 5.1856, c/a = 0.5527, 2 atoms in primitive cell
    a_sn = 5.1856
    c_sn = a_sn * 0.5527
    sn_lattice = np.array([
        [a_sn, 0.0, 0.0],
        [0.0, a_sn, 0.0],
        [0.5 * a_sn, 0.5 * a_sn, 0.5 * c_sn],
    ])
    sn_basis = np.array([
        [0.0, -0.25 * a_sn,  0.125 * c_sn],
        [0.0,  0.25 * a_sn, -0.125 * c_sn],
    ])
    phases["beta_sn"] = (sn_lattice, sn_basis, [0] * 2)

    # --- bc8 / Ge-III (Ia-3) ---
    # a = 7.053, x = 0.1013, 8 atoms in primitive cell
    a_bc8 = 7.053
    x = 0.1013
    bc8_lattice = np.array([
        [-0.5 * a_bc8,  0.5 * a_bc8,  0.5 * a_bc8],
        [ 0.5 * a_bc8, -0.5 * a_bc8,  0.5 * a_bc8],
        [ 0.5 * a_bc8,  0.5 * a_bc8, -0.5 * a_bc8],
    ])
    bc8_basis = np.array([
        [ x * a_bc8,  x * a_bc8,  x * a_bc8],
        [-x * a_bc8, -x * a_bc8, -x * a_bc8],
        [ x * a_bc8, -x * a_bc8,  (0.5 - x) * a_bc8],
        [-x * a_bc8,  x * a_bc8, -(0.5 - x) * a_bc8],
        [ (0.5 - x) * a_bc8,  x * a_bc8, -x * a_bc8],
        [-(0.5 - x) * a_bc8, -x * a_bc8,  x * a_bc8],
        [-x * a_bc8,  (0.5 - x) * a_bc8,  x * a_bc8],
        [ x * a_bc8, -(0.5 - x) * a_bc8, -x * a_bc8],
    ])
    phases["bc8"] = (bc8_lattice, bc8_basis, [0] * 8)

    # --- st12 (P4_3_2_12) ---
    # a = 6.0177, c/a = 1.182, 12 atoms
    a_st12 = 6.0177
    c_st12 = a_st12 * 1.182
    x1 = 0.0874
    x2 = 0.1709
    y2 = 0.3704
    z2 = 0.2525
    st12_lattice = np.array([
        [a_st12, 0.0, 0.0],
        [0.0, a_st12, 0.0],
        [0.0, 0.0, c_st12],
    ])
    st12_basis = np.array([
        [ x1 * a_st12,  x1 * a_st12, 0.0],
        [-x1 * a_st12, -x1 * a_st12, 0.5 * c_st12],
        [ (0.5 - x1) * a_st12,  (0.5 + x1) * a_st12, 0.75 * c_st12],
        [ (0.5 + x1) * a_st12,  (0.5 - x1) * a_st12, 0.25 * c_st12],
        [ x2 * a_st12,  y2 * a_st12,  z2 * c_st12],
        [-x2 * a_st12, -y2 * a_st12,  (0.5 + z2) * c_st12],
        [ (0.5 - y2) * a_st12,  (0.5 + x2) * a_st12,  (0.75 + z2) * c_st12],
        [ (0.5 + y2) * a_st12,  (0.5 - x2) * a_st12,  (0.25 + z2) * c_st12],
        [ y2 * a_st12,  x2 * a_st12, -z2 * c_st12],
        [-y2 * a_st12, -x2 * a_st12,  (0.5 - z2) * c_st12],
        [ (0.5 - x2) * a_st12,  (0.5 + y2) * a_st12,  (0.75 - z2) * c_st12],
        [ (0.5 + x2) * a_st12,  (0.5 - y2) * a_st12,  (0.25 - z2) * c_st12],
    ])
    phases["st12"] = (st12_lattice, st12_basis, [0] * 12)

    return phases


# ---------------------------------------------------------------------------
# EFS multi-block parser
# ---------------------------------------------------------------------------

def _parse_energies_volumes(path: Path) -> list[tuple[float, float]]:
    """Return list of (volume [A^3], energy [eV]) from a multi-block EFS CFG."""
    ev = []
    for block in re.split(r"(?=BEGIN_CFG\b)", path.read_text()):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        sm = re.search(
            r"Supercell\s*\n((?:\s*[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\n){3})",
            block,
        )
        em = re.search(r"\bEnergy\b\s*\n\s*([-\d.eE+]+)", block)
        if sm and em:
            rows = [[float(x) for x in row.split()]
                    for row in sm.group(1).strip().splitlines()]
            cell = np.array(rows)
            vol = abs(np.linalg.det(cell))
            ev.append((vol, float(em.group(1))))
    return ev


# ---------------------------------------------------------------------------
# DFT reference data loader
# ---------------------------------------------------------------------------

# Map DFT reference file names (src/physical_validation/reference/multiphase_ev_DFT/<name>)
# to phase names used in build_phases()
_FIG2_TO_PHASE = {
    "dia": "diamond",
    "fcc": "fcc",
    "bcc": "bcc",
    "bc8": "bc8",
    "hcp": "hcp",
    "hd": "hd",
    "st12": "st12",
    "beta": "beta_sn",
}


def load_dft_reference(dft_dir: Path) -> dict:
    """Load DFT reference E-V data from flat files in dft_dir.

    Returns dict: phase_name -> (vol_per_atom array, energy_per_atom array)
    """
    dft_data = {}
    for prefix, phase_name in _FIG2_TO_PHASE.items():
        dft_file = dft_dir / f"{prefix}.dat"
        if not dft_file.exists():
            print(f"  Warning: DFT file not found: {dft_file}")
            continue
        vols, energies = [], []
        for line in dft_file.read_text().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                # Format: lattice_const, volume_per_atom, energy_per_atom
                vols.append(float(parts[1]))
                energies.append(float(parts[2]))
        if vols:
            dft_data[phase_name] = (np.array(vols), np.array(energies))
            print(f"  Loaded DFT: {phase_name} ({len(vols)} points)")
    return dft_data


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    outdir: Path,
    n_volumes: int,
    vol_range: float,
    dft_dir: str | None = None,
    backend: str = "mlp",
    mlp: str | None = None,
    lammps_cmd: str = DEFAULT_LAMMPS,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    phases = build_phases()
    vol_fracs = np.linspace(1.0 - vol_range / 2.0, 1.0 + vol_range / 2.0, n_volumes)

    # --- Load DFT reference if available ---
    dft_data = {}
    if dft_dir:
        dft_path = Path(dft_dir)
        if dft_path.exists():
            print(f"Loading DFT reference from {dft_path}")
            dft_data = load_dft_reference(dft_path)
        else:
            print(f"DFT directory not found: {dft_path}")

    # --- Energy computation ---
    lammps_dir = outdir / "lammps"
    results = {}  # name -> (vol_per_atom array, energy_per_atom array)

    for name, (cell0, pos0, types) in phases.items():
        n_atom = len(pos0)
        vol0 = abs(np.linalg.det(cell0))
        print(f"\n  [{name}]  {n_atom} atoms  V0 = {vol0:.2f} A^3")

        if backend == "mlp":
            configs = [(cell0 * s ** (1.0 / 3.0), pos0 * s ** (1.0 / 3.0), types)
                       for s in vol_fracs]

            in_cfg = outdir / f"{name}_in.cfg"
            out_cfg = outdir / f"{name}_out.cfg"
            write_bare_cfg(configs, in_cfg)
            rc = calc_efs(mlp, pot, in_cfg, out_cfg, quiet=True)
            if rc != 0:
                print(f"  Warning: mlp returned {rc} for {name}")
                continue
            if not out_cfg.exists():
                print(f"  Error: no output for {name}")
                continue

            ev = _parse_energies_volumes(out_cfg)
            if len(ev) != n_volumes:
                print(f"  Warning: expected {n_volumes} blocks, got {len(ev)} for {name}")
            if not ev:
                continue
        else:
            ev = []
            for i, s in enumerate(vol_fracs):
                scale = s ** (1.0 / 3.0)
                cell = cell0 * scale
                pos = pos0 * scale
                vol = abs(np.linalg.det(cell))
                e = _run_lammps_energy(lammps_cmd, pot, cell, pos,
                                        lammps_dir / name / f"point_{i:03d}")
                ev.append((vol, e))

        vols, energies = zip(*sorted(ev))
        results[name] = (
            np.array(vols) / n_atom,
            np.array(energies) / n_atom,
        )
        e_min = np.min(results[name][1])
        print(f"    E_min/atom (MTP) = {e_min:.4f} eV  ({len(ev)} points)")

    # --- Write text data ---
    txt = outdir / "multiphase_ev.txt"
    with open(txt, "w") as f:
        f.write("# phase  V_per_atom[A^3]  E_per_atom[eV]\n")
        for name, (vpa, epa) in results.items():
            for v, e in zip(vpa, epa):
                f.write(f"{name:15s}  {v:10.4f}  {e:12.6f}\n")
    print(f"\n  Data written to {txt}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Phase order consistent with the reference figure
        phase_order = [
            "diamond", "hd", "bc8", "st12", "beta_sn",
            "hcp", "fcc", "bcc",
        ]
        # Color palette matching reference all.png (same 8 colors)
        colors = [
            "#377eb8", "#ff7f00", "#4daf4a", "#f781bf",
            "#a65628", "#984ea3", "#999999", "#e41a1c",
        ]

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot each phase: MTP line + DFT diamond markers
        for idx, name in enumerate(phase_order):
            if name not in results:
                continue
            color = colors[idx % len(colors)]
            vpa, epa = results[name]

            # MTP: line with circle markers
            ax.plot(
                vpa, epa, "-", color=color, markersize=4, lw=1.5,
                label=f"{name} MTP",
            )

            # DFT: diamond markers, same color, slightly transparent
            if name in dft_data:
                v_dft, e_dft = dft_data[name]
                ax.plot(
                    v_dft, e_dft, "D", color=color,
                    markersize=5, markeredgewidth=0.5,
                    markeredgecolor="white", alpha=0.9,
                    label=f"{name} DFT",
                )

        backend_tag = "" if backend == "mlp" else " (LAMMPS)"
        ax.set_xlabel("Volume per atom (Å³)", fontsize=12)
        ax.set_ylabel("Energy per atom (eV)", fontsize=12)
        ax.set_title(f"MTP{backend_tag} vs DFT Multi-Phase E-V Curves — Ge", fontsize=13)
        ax.legend(fontsize=8, ncol=2, loc="lower left",
                  frameon=True, facecolor="white", framealpha=0.8)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_xlim(15, 32)
        ax.set_ylim(-4.55, -3.95)
        fig.tight_layout()
        png = outdir / "multiphase_ev.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png}")
    except ImportError as e:
        print(f"  Matplotlib not available ({e}); skipping plot.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-phase E-V curves for Ge MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: results/tests/<pot's parent dir "
             "name>/multiphase_ev, e.g. --pot results/potentials/pot_660277/pot.almtp "
             "-> results/tests/pot_660277/multiphase_ev)",
    )
    parser.add_argument("--n-volumes", type=int, default=21,
                        help="Number of volume points per phase (default: 21)")
    parser.add_argument("--vol-range", type=float, default=0.70,
                        help="Fractional volume range (default: 0.70 -> +/-35%%)")
    parser.add_argument("--backend", choices=["mlp", "lammps"], default="mlp",
                        help="Evaluation engine: 'mlp' calculate_efs (default) or "
                             "'lammps' (pair_style mtp+nlh — for potentials 'mlp' can't load)")
    parser.add_argument("--mlp", default=MLP_DEFAULT,
                        help="mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(--backend lammps only; default: {DEFAULT_LAMMPS})")
    parser.add_argument(
        "--dft-data",
        default=DFT_DATA_DEFAULT,
        help="Path to DFT reference directory (default: ilearn Fig2)",
    )
    args = parser.parse_args()

    if args.outdir is None:
        run_name = Path(args.pot).resolve().parent.name
        outdir = Path("results/tests") / run_name / "multiphase_ev"
    else:
        outdir = Path(args.outdir)

    run(
        pot=args.pot,
        outdir=outdir,
        n_volumes=args.n_volumes,
        vol_range=args.vol_range,
        dft_dir=args.dft_data,
        backend=args.backend,
        mlp=args.mlp,
        lammps_cmd=args.lammps,
    )


if __name__ == "__main__":
    main()
