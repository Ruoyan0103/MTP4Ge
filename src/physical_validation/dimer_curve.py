"""Ge-Ge dimer curve test for MTP potentials, via LAMMPS.

Scans an isolated Ge-Ge dimer over a range of separations and evaluates the
total energy with LAMMPS (pair_style hybrid/overlay mtp nlh, via
src/physical_validation/energy_volume.py's single-point evaluator — same workaround as
the rest of the LAMMPS test family, for potentials trained with a radial
basis type `mlp calculate_efs` cannot load). Reproduces the join.in /
plot-poteng.py workflow from
/scratch/project_2012355/Paper_3/00-subsets/07-short_range/01-SW_joining/,
embedded into this project:
  - short-range reference: src/physical_validation/reference/dimer_energy/dimer_NLH.dat
    (from tools/nlhpot-Ge-Ge — the standalone NLH short-range table)
  - DFT reference: src/physical_validation/reference/dimer_energy/dimer_DFT.dat
    (from ../dimer_data/method1/dimer_dft/dimer_curve.dat)

The two atoms sit in a large, mostly-empty periodic box (default 40 Å) so
there is no periodic-image interaction — same isolation trick as
build_isolated_atom() in src/physical_validation/defect_formation.py.

Usage:
    python src/physical_validation/dimer_curve.py --pot results/potentials/20.mtp
    python src/physical_validation/dimer_curve.py --pot results/potentials/20.mtp \\
        --r-min 0.7 --r-max 6.0 --r-step 0.05 \\
        --lammps "srun /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from energy_volume import DEFAULT_LAMMPS, _run_lammps_energy  # noqa: E402

REF_DIR = Path(__file__).parent / "reference" / "dimer_energy"
NLH_REF = REF_DIR / "dimer_NLH.dat"
DFT_REF = REF_DIR / "dimer_DFT.dat"


# ---------------------------------------------------------------------------
# Dimer scan
# ---------------------------------------------------------------------------

def dimer_scan(
    lammps_cmd: str, pot: str, distances: np.ndarray, box_size: float, outdir: Path,
) -> np.ndarray:
    """Return per-distance total energy [eV] for an isolated Ge-Ge dimer."""
    cell = np.eye(3) * box_size
    scan_dir = outdir / "lammps"

    energies = np.zeros_like(distances)
    for i, r in enumerate(distances):
        pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        e = _run_lammps_energy(lammps_cmd, pot, cell, pos, scan_dir / f"r_{r:.3f}")
        energies[i] = e
        print(f"  [{i + 1}/{len(distances)}] r={r:.3f} A  E={e:.6f} eV")
    return energies


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    outdir: Path,
    lammps_cmd: str,
    r_min: float,
    r_max: float,
    r_step: float,
    box_size: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    distances = np.arange(r_min, r_max + 1e-9, r_step)
    print(f"  Ge-Ge dimer scan: {len(distances)} points, "
          f"r = [{distances[0]:.3f}, {distances[-1]:.3f}] A, box = {box_size} A")

    energies = dimer_scan(lammps_cmd, pot, distances, box_size, outdir)

    # --- Write data (same 2-column format as the original join.out) ---
    txt = outdir / "dimer_curve_mtp.txt"
    with open(txt, "w") as f:
        f.write(f"# Ge-Ge dimer curve (LAMMPS, pair_style mtp+nlh)  |  pot: {pot}\n")
        f.write("# distance[A]  energy[eV]\n")
        for r, e in zip(distances, energies):
            f.write(f"  {r:.4f}  {e:.6f}\n")
    print(f"  Data written to {txt}")

    r_min_e = distances[int(np.argmin(energies))]
    print(f"  Minimum energy {energies.min():.4f} eV at r = {r_min_e:.3f} A")

    # --- Plot: matches the original plot-poteng.py layout ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color = ['#1f77b4', '#d62728', '#ff7f0e']

        data_table = np.loadtxt(NLH_REF) if NLH_REF.exists() else None
        data_dft = np.loadtxt(DFT_REF) if DFT_REF.exists() else None

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # --- Left panel: short range (repulsive wall), log scale ---
        if data_table is not None:
            m = (data_table[:, 0] >= 0) & (data_table[:, 0] <= 1)
            ax[0].plot(data_table[m, 0], data_table[m, 1], color=color[0], label="NLH")
        m = (distances >= 0) & (distances <= 1)
        ax[0].plot(distances[m], energies[m], '--', color=color[1], label="NLH+MTP (LAMMPS)")
        ax[0].set_xlabel("Distance (Å)", fontsize=14)
        ax[0].set_ylabel("Potential Energy (eV)", fontsize=14)
        ax[0].set_xlim(0, 1)
        ax[0].set_yscale("log")
        ax[0].grid()
        ax[0].text(0.1, 0.92, "(a)", transform=ax[0].transAxes, fontsize=17, va="top")

        # --- Right panel: bonding region + DFT overlay ---
        if data_table is not None:
            m = (data_table[:, 0] >= 1) & (data_table[:, 0] <= 3)
            ax[1].plot(data_table[m, 0], data_table[m, 1], color=color[0], label="NLH")
        m = (distances >= 1) & (distances <= 3)
        ax[1].plot(distances[m], energies[m], '--', color=color[1], label="NLH+MTP (LAMMPS)")
        if data_dft is not None:
            m = (data_dft[:, 0] >= 1.0) & (data_dft[:, 0] <= 3.0)
            ax[1].plot(data_dft[m, 0], data_dft[m, 1], 'o', ms=4, color=color[2], label="DFT")
        ax[1].axvline(x=1.5, color="grey", ls="--", label=r"$r_1=1.5\ \mathrm{\AA}$")
        ax[1].axvline(x=2.3, color="grey", ls="--", label=r"$r_2=2.3\ \mathrm{\AA}$")
        ax[1].set_xlabel("Distance (Å)", fontsize=14)
        ax[1].set_xlim(1, 3)
        ax[1].grid()
        ax[1].text(0.1, 0.92, "(b)", transform=ax[1].transAxes, fontsize=17, va="top")

        # --- Inset: zoom on the join region ---
        ax_inset = ax[1].inset_axes([0.46, 0.46, 0.52, 0.52])
        if data_table is not None:
            m = (data_table[:, 0] >= 1.5) & (data_table[:, 0] <= 3)
            ax_inset.plot(data_table[m, 0], data_table[m, 1], color=color[0])
        m = (distances >= 1.5) & (distances <= 3)
        ax_inset.plot(distances[m], energies[m], '--', color=color[1])
        if data_dft is not None:
            m = (data_dft[:, 0] >= 1.5) & (data_dft[:, 0] <= 3)
            ax_inset.plot(data_dft[m, 0], data_dft[m, 1], 'o', ms=4, color=color[2])
        ax_inset.axvline(x=1.5, color="grey", ls="--")
        ax_inset.axvline(x=2.3, color="grey", ls="--")
        ax_inset.set_xlim(1.5, 3)
        ax_inset.grid()

        handles, labels = ax[1].get_legend_handles_labels()
        ax[0].legend(handles, labels, loc="upper right", fontsize=11)
        fig.tight_layout()
        png = outdir / "dimer_curve.png"
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
        description="Ge-Ge dimer curve test for MTP potential via LAMMPS"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.mtp / .almtp)")
    parser.add_argument("--r-min", type=float, default=0.001, help="Min separation Å (default: 0.001)")
    parser.add_argument("--r-max", type=float, default=5.0, help="Max separation Å (default: 5.0)")
    parser.add_argument("--r-step", type=float, default=0.005, help="Step size Å (default: 0.005)")
    parser.add_argument("--box-size", type=float, default=40.0,
                        help="Cubic box edge length Å — must be well beyond the MTP cutoff "
                             "so periodic images don't interact (default: 40.0)")
    parser.add_argument("--outdir", default="results/tests/dimer_curve", help="Output directory")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(default: {DEFAULT_LAMMPS})")
    args = parser.parse_args()

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        lammps_cmd=args.lammps,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
        box_size=args.box_size,
    )


if __name__ == "__main__":
    main()
