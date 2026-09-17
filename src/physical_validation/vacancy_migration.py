"""Vacancy migration barriers via NEB for Ge MTP potential.

Computes 1st- and 2nd-nearest-neighbour vacancy hop barriers using ASE NEB.

Evaluates via --backend mlp (default: MTPCalculator, mlp calculate_efs per
image) or --backend lammps (MTPLammpsCalculator, LAMMPS pair_style
hybrid/overlay mtp nlh per image). The LAMMPS backend is needed for
potentials trained with a radial basis type `mlp calculate_efs` in the
mlip-3-prune build cannot load (fails with "Wrong radial basis type") — see
src/physical_validation/elastic_constant/potential.mod for the same LAMMPS pair_style
workaround used for elastic constants.

Reference (GAP-Ge / DFT-GGA):
  1NN barrier: 0.22 eV (DFT),  0.21 eV (GAP)
  2NN barrier: 1.77 eV (DFT),  1.71 eV (GAP)

Usage:
    python src/physical_validation/vacancy_migration.py --pot results/potentials/pot_al.almtp
    python src/physical_validation/vacancy_migration.py --pot results/potentials/pot_al.almtp \\
        --a0 5.779 --n-images 7 --fmax 0.05
    python src/physical_validation/vacancy_migration.py --pot results/potentials/20.mtp --backend lammps \\
        --a0 5.7547 --lammps "srun /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from mtp_calculator import MTPCalculator, MTPLammpsCalculator
from utils import write_bare_cfg
from energy_volume import DEFAULT_LAMMPS

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"

DFT_REF = {"1NN": 0.22, "2NN": 1.77}
GAP_REF = {"1NN": 0.21, "2NN": 1.71}


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------

def build_vacancy_supercell(a0: float, repeat: tuple = (2, 2, 2)):
    """2×2×2 diamond supercell with one vacancy.

    Returns (atoms_with_vacancy, vac_pos_cartesian) as ASE Atoms.
    The vacancy is created by removing the atom nearest to the origin.
    """
    from ase.build import bulk
    atoms = bulk("Ge", "diamond", a=a0, cubic=True).repeat(repeat)
    pos = atoms.get_positions()
    dists = np.linalg.norm(pos, axis=1)
    vac_idx = int(np.argmin(dists))
    vac_pos = pos[vac_idx].copy()
    del atoms[vac_idx]
    return atoms, vac_pos


def _nn_distances(atoms, vac_pos: np.ndarray) -> np.ndarray:
    """Return distances from all atoms to vac_pos with minimum image convention."""
    pos = atoms.get_positions()
    diff = pos - vac_pos
    cell = np.array(atoms.get_cell())
    cell_inv = np.linalg.inv(cell)
    frac = diff @ cell_inv
    frac -= np.round(frac)
    diff_mic = frac @ cell
    return np.linalg.norm(diff_mic, axis=1)


def build_neb_endpoints(a0: float, hop_shell: int, repeat: tuple = (2, 2, 2)):
    """Build initial and final ASE Atoms for a vacancy hop.

    hop_shell=1 → 1NN hop (distance ≈ a0·√3/4)
    hop_shell=2 → 2NN hop (distance ≈ a0/√2)

    Returns (initial, final, hop_atom_idx, vac_pos).
    Both structures have the same number of atoms (vacancy excluded).
    """
    atoms, vac_pos = build_vacancy_supercell(a0, repeat)

    dists = _nn_distances(atoms, vac_pos)

    # Expected distances for diamond structure
    d1nn = a0 * np.sqrt(3.0) / 4.0  # ≈ 2.5 Å
    d2nn = a0 / np.sqrt(2.0)         # ≈ 4.1 Å

    if hop_shell == 1:
        target_dist = d1nn
    elif hop_shell == 2:
        target_dist = d2nn
    else:
        raise ValueError(f"hop_shell must be 1 or 2, got {hop_shell}")

    # Find candidate atoms within tolerance
    candidates = np.where(np.abs(dists - target_dist) < 0.4)[0]
    if len(candidates) == 0:
        raise RuntimeError(
            f"No {hop_shell}NN atoms found near vac_pos={vac_pos}. "
            f"Closest distances: {sorted(dists)[:8]}"
        )
    hop_idx = int(candidates[0])
    hop_pos_initial = atoms.get_positions()[hop_idx].copy()

    # initial: atom at hop_pos_initial, vacancy at vac_pos
    initial = atoms.copy()

    # final: hop atom moved to vac_pos, new vacancy at hop_pos_initial
    final = atoms.copy()
    new_pos = final.get_positions()
    new_pos[hop_idx] = vac_pos.copy()
    final.set_positions(new_pos)

    return initial, final, hop_idx, vac_pos


# ---------------------------------------------------------------------------
# NEB runner
# ---------------------------------------------------------------------------

def _save_neb_images(images, outdir: Path) -> None:
    """Write each NEB image (initial, intermediates, final) to xyz+POSCAR+lammps-data+cfg."""
    from ase.io import write as ase_write
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(images)
    for i, img in enumerate(images):
        tag = "initial" if i == 0 else "final" if i == n - 1 else f"image{i:02d}"
        base = outdir / f"{i:02d}_{tag}"
        ase_write(str(base.with_suffix(".xyz")), img, format="extxyz")
        ase_write(str(base.with_suffix(".POSCAR")), img, format="vasp")
        ase_write(str(base.with_suffix(".data")), img,
                  format="lammps-data", atom_style="atomic")
        cell = img.get_cell().array
        pos = img.get_positions()
        types = [0] * len(img)  # single-species Ge (SPECIES_MAP: {"Ge": 0})
        write_bare_cfg([(cell, pos, types)], base.with_suffix(".cfg"))


def _relax_endpoint(atoms, make_calc, fmax: float = 0.02, max_steps: int = 300):
    """Relax a vacancy supercell to its local minimum before NEB.

    Uses a tighter fmax than the NEB convergence criterion so the endpoint
    sits at a genuine local minimum; ASE NEB freezes endpoints during band
    optimisation, so pre-relaxation is required for accurate barriers.
    """
    from ase.optimize import BFGS
    atoms = atoms.copy()
    atoms.calc = make_calc()
    opt = BFGS(atoms, trajectory=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    if not converged:
        print(f"    Warning: endpoint relaxation did not converge in {max_steps} steps")
    return atoms


def run_neb(
    initial,
    final,
    make_calc,
    n_images: int = 7,
    fmax: float = 0.05,
    fmax_relax: float = 0.02,
    max_steps: int = 200,
    image_dir: Path | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run climbing-image NEB; return (barrier [eV], distances, energies).

    Relaxes both endpoints to their local minima before building the band
    (ASE NEB freezes endpoints during optimisation, so pre-relaxation is
    required for accurate barrier heights).

    `make_calc` is a zero-arg factory returning a fresh calculator (MTPCalculator
    or MTPLammpsCalculator) — a fresh instance is created for each image to
    avoid shared-state issues. If image_dir is given, writes every image
    (initial, intermediates, final) to image_dir/before/ right after
    interpolation and to image_dir/after/ once the band optimisation finishes.
    """
    from ase.mep.neb import NEB
    from ase.optimize import BFGS

    print(f"  Relaxing endpoints (fmax={fmax_relax} eV/Å) ...")
    initial = _relax_endpoint(initial, make_calc, fmax=fmax_relax)
    final   = _relax_endpoint(final,   make_calc, fmax=fmax_relax)

    images = [initial.copy()]
    for i in range(n_images):
        img = initial.copy()
        img.calc = make_calc()
        images.append(img)
    images.append(final.copy())

    # Set calculators on endpoints too (required for energy queries)
    images[0].calc = make_calc()
    images[-1].calc = make_calc()

    neb = NEB(images, climb=True, k=1.0)
    neb.interpolate()

    if image_dir is not None:
        _save_neb_images(images, image_dir / "before")
        print(f"  Pre-NEB images written to {image_dir / 'before'}")

    optimizer = BFGS(neb, trajectory=None)
    converged = optimizer.run(fmax=fmax, steps=max_steps)
    if not converged:
        print(f"  Warning: NEB did not converge in {max_steps} steps")

    if image_dir is not None:
        _save_neb_images(images, image_dir / "after")
        print(f"  Post-NEB images written to {image_dir / 'after'}")

    # Extract energies along the path
    energies = np.array([img.get_potential_energy() for img in images])

    # Arc-length coordinate (cumulative displacement of the hopping atom)
    hop_idx = _find_moving_atom(initial, final)
    coords = np.array([img.get_positions()[hop_idx] for img in images])
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])

    barrier = float(np.max(energies) - energies[0])
    return barrier, arc, energies


def _find_moving_atom(initial, final) -> int:
    """Return the index of the atom that moves most between initial and final."""
    disp = np.linalg.norm(
        final.get_positions() - initial.get_positions(), axis=1
    )
    return int(np.argmax(disp))


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(pot: str, outdir: Path, a0: float,
        n_images: int, fmax: float, fmax_relax: float, max_steps: int,
        backend: str = "mlp", mlp: str | None = None,
        lammps_cmd: str = DEFAULT_LAMMPS) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if backend == "mlp":
        make_calc = lambda: MTPCalculator(mlp=mlp, pot=pot)
    else:
        make_calc = lambda: MTPLammpsCalculator(lammps_cmd=lammps_cmd, pot=pot)

    neb_results = {}
    for shell, label in [(1, "1NN"), (2, "2NN")]:
        print(f"\n  [{label} vacancy hop]")
        try:
            initial, final, hop_idx, vac_pos = build_neb_endpoints(a0, shell)
        except RuntimeError as e:
            print(f"  Error building endpoints: {e}")
            continue

        hop_dist = np.linalg.norm(
            initial.get_positions()[hop_idx] - vac_pos
        )
        print(f"  Hop distance = {hop_dist:.3f} Å  |  {len(initial)} atoms")
        print(f"  Running NEB ({n_images} images, fmax={fmax} eV/Å) ...")

        barrier, arc, energies = run_neb(
            initial, final, make_calc,
            n_images=n_images, fmax=fmax, fmax_relax=fmax_relax,
            max_steps=max_steps, image_dir=outdir / label,
        )
        neb_results[label] = (arc, energies, barrier)

        dft = DFT_REF.get(label, float("nan"))
        gap = GAP_REF.get(label, float("nan"))
        print(
            f"  Barrier = {barrier:.3f} eV  "
            f"(DFT ref: {dft:.2f} eV,  GAP ref: {gap:.2f} eV)"
        )

    if not neb_results:
        print("No NEB results. Exiting.")
        return

    # --- Write text data ---
    txt = outdir / "vacancy_migration.txt"
    with open(txt, "w") as f:
        f.write(f"Vacancy migration barriers — pot: {pot}  a0={a0:.4f} Å\n\n")
        f.write(f"{'Hop':<6}  {'Barrier (eV)':>13}  "
                f"{'DFT ref (eV)':>13}  {'GAP ref (eV)':>13}\n")
        f.write("-" * 52 + "\n")
        for label, (_, _, barrier) in neb_results.items():
            dft = f"{DFT_REF.get(label, float('nan')):.2f}"
            gap = f"{GAP_REF.get(label, float('nan')):.2f}"
            f.write(f"{label:<6}  {barrier:>13.3f}  {dft:>13}  {gap:>13}\n")

        for label, (arc, energies, _) in neb_results.items():
            f.write(f"\n# {label} NEB path\n")
            f.write("# arc[A]  E[eV]  dE[eV]\n")
            for s, e in zip(arc, energies):
                f.write(f"  {s:.4f}  {e:.6f}  {e - energies[0]:.6f}\n")
    print(f"\n  Results written to {txt}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(neb_results), figsize=(5 * len(neb_results), 4),
                                 squeeze=False)
        for ax, (label, (arc, energies, barrier)) in zip(axes[0], neb_results.items()):
            dE = (energies - energies[0]) * 1000  # convert to meV
            ax.plot(arc, dE, "o-", color="tab:blue", markersize=5)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Arc length (Å)")
            ax.set_ylabel("ΔE (meV)")
            ax.set_title(f"{label} hop  [barrier = {barrier * 1000:.0f} meV]")
            ax.text(0.98, 0.05, f"DFT: {DFT_REF.get(label, 0) * 1000:.0f} meV",
                    transform=ax.transAxes, ha="right", fontsize=9, color="gray")
        backend_tag = "" if backend == "mlp" else " (LAMMPS)"
        fig.suptitle(f"Ge Vacancy Migration — MTP{backend_tag}", fontsize=12)
        fig.tight_layout()
        png = outdir / "vacancy_migration.png"
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
        description="Vacancy migration NEB test for Ge MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--a0", type=float, default=5.76,
                        help="Lattice constant Å — use MTP equilibrium (default: 5.76)")
    parser.add_argument("--n-images", type=int, default=7,
                        help="Number of NEB intermediate images (default: 7)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence threshold eV/Å (default: 0.05)")
    parser.add_argument("--fmax-relax", type=float, default=0.02,
                        help="Force threshold for endpoint pre-relaxation eV/Å (default: 0.02)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max BFGS steps (default: 200)")
    parser.add_argument("--outdir", default="results/tests/vacancy_migration")
    parser.add_argument("--backend", choices=["mlp", "lammps"], default="mlp",
                        help="Evaluation engine: 'mlp' calculate_efs (default) or "
                             "'lammps' (pair_style mtp+nlh — for potentials 'mlp' can't load)")
    parser.add_argument("--mlp", default=MLP_DEFAULT,
                        help="mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(--backend lammps only; default: {DEFAULT_LAMMPS})")
    args = parser.parse_args()

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        a0=args.a0,
        n_images=args.n_images,
        fmax=args.fmax,
        fmax_relax=args.fmax_relax,
        max_steps=args.max_steps,
        backend=args.backend,
        mlp=args.mlp,
        lammps_cmd=args.lammps,
    )


if __name__ == "__main__":
    main()
