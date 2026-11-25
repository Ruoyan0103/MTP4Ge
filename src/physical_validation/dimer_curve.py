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
    python src/physical_validation/dimer_curve.py --pot results/potentials/20.mtp --backend lammps
    # replot an existing scan with GAP overlaid (no LAMMPS):
    python src/physical_validation/dimer_curve.py --pot results/potentials/pot_1023/28vib.mtp --plot-only \\
        --outdir results/tests/pot_1023/28vib/dimer_curve \\
        --extra GAP=results/tests/ref/gap_ge_v10/dimer_curve/GAP.txt
    python src/physical_validation/dimer_curve.py --pot results/potentials/20.mtp --backend lammps \\
        --r-min 0.7 --r-max 6.0 --r-step 0.05

The LAMMPS command defaults to energy_volume.DEFAULT_LAMMPS, prefixed with
`srun` automatically when run inside a SLURM job; --lammps overrides it.
"""

import argparse
import os
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
    extra: dict[str, Path] | None = None,
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
    plot(distances, energies, outdir, extra)


def _load_curve(path: Path) -> np.ndarray:
    """Two-column (r [Å], E [eV]) file, whitespace- or comma-separated, '#' comments."""
    rows = [l.replace(",", " ").split() for l in path.read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    return np.array([[float(a), float(b)] for a, b, *_ in rows])


def plot(distances: np.ndarray, energies: np.ndarray, outdir: Path,
         extra: dict[str, Path] | None = None) -> None:
    """dimer_curve.png; `extra` = {label: file} curves (e.g. GAP) overlaid on every panel."""
    curves = {k: _load_curve(Path(v)) for k, v in (extra or {}).items()}
    # --- Plot: matches the original plot-poteng.py layout ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color = ['#1f77b4', '#d62728', '#ff7f0e']
        extra_color = ['#2ca02c', '#9467bd', '#8c564b']

        def overlay(axis, lo, hi, label=True):
            for (name, c), col in zip(curves.items(), extra_color):
                m = (c[:, 0] >= lo) & (c[:, 0] <= hi)
                axis.plot(c[m, 0], c[m, 1], ':', lw=2, color=col, label=name if label else None)

        data_table = np.loadtxt(NLH_REF) if NLH_REF.exists() else None
        data_dft = np.loadtxt(DFT_REF) if DFT_REF.exists() else None

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # --- Left panel: short range (repulsive wall), log scale ---
        if data_table is not None:
            m = (data_table[:, 0] >= 0) & (data_table[:, 0] <= 1)
            ax[0].plot(data_table[m, 0], data_table[m, 1], color=color[0], label="NLH")
        m = (distances >= 0) & (distances <= 1)
        ax[0].plot(distances[m], energies[m], '--', color=color[1], label="NLH+MTP (LAMMPS)")
        overlay(ax[0], 0, 1, label=False)
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
        overlay(ax[1], 1, 3)
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
        overlay(ax_inset, 1.5, 3, label=False)
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
    parser.add_argument("--r-max", type=float, default=4.0, help="Max separation Å (default: 4.0)")
    parser.add_argument("--r-step", type=float, default=0.01, help="Step size Å (default: 0.01)")
    parser.add_argument("--box-size", type=float, default=40.0,
                        help="Cubic box edge length Å — must be well beyond the MTP cutoff "
                             "so periodic images don't interact (default: 40.0)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/tests/<pot's parent dir "
                             "name>/dimer_curve, e.g. --pot results/potentials/pot_660277/pot.almtp "
                             "-> results/tests/pot_660277/dimer_curve)")
    parser.add_argument("--backend", choices=["lammps"], default="lammps",
                        help="Evaluation backend; only LAMMPS (pair_style mtp+nlh) is "
                             "supported. Runs DEFAULT_LAMMPS, prefixed with 'srun' when "
                             "inside a SLURM job (default: lammps)")
    parser.add_argument("--lammps", default=None,
                        help="Override the LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(default: [srun] {DEFAULT_LAMMPS})")
    parser.add_argument("--extra", action="append", default=[], metavar="LABEL=FILE",
                        help="Overlay another dimer curve (2 columns r[Å] E[eV], whitespace or "
                             "comma), e.g. GAP=results/tests/ref/gap_ge_v10/dimer_curve/GAP.txt; "
                             "absolute energies, as plotted for MTP/DFT. Repeatable.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip the LAMMPS scan; replot from <outdir>/dimer_curve_mtp.txt")
    args = parser.parse_args()
    extra = dict(e.split("=", 1) for e in args.extra)

    if args.lammps is None:
        args.lammps = f"srun {DEFAULT_LAMMPS}" if "SLURM_JOB_ID" in os.environ else DEFAULT_LAMMPS

    if args.outdir is None:
        run_name = Path(args.pot).resolve().parent.name
        outdir = Path("results/tests") / run_name / "dimer_curve"
    else:
        outdir = Path(args.outdir)

    if args.plot_only:
        r, e = _load_curve(outdir / "dimer_curve_mtp.txt").T
        plot(r, e, outdir, extra)
        return

    run(
        pot=args.pot,
        outdir=outdir,
        lammps_cmd=args.lammps,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
        box_size=args.box_size,
        extra=extra,
    )


if __name__ == "__main__":
    main()
