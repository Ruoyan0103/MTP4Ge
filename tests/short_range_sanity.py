"""
Short-range sanity checks for MTP potentials.

Diagnoses whether a potential has a physically sane repulsive wall at short
Ge-Ge separations, or a spurious bounded/unbounded false minimum (as found
for the bond-centre interstitial: results/tests/defect_formation/B_bond_center).

Three checks, all single-point evaluations via `mlp calculate_efs` (no
relaxation, so each probes the raw PES rather than where an optimiser ends up):

  1. Dimer curve      - two isolated Ge atoms, energy vs separation 0.6-5.0 A.
  2. Bulk compression  - 2x2x2 diamond supercell, uniformly scaled well past
                         equilibrium, down to ~0.9 A nearest-neighbour distance.
  3. Bond-centre scan  - the actual 217-atom bond-centre geometry used by
                         defect_formation.py, with the interstitial swept along
                         the bond axis from near one host atom out to the exact
                         bond-centre midpoint. Reports E_f(t) = E_total - N*mu_Ge
                         so it's directly comparable to defect_formation_energies.txt.

Usage:
    python tests/short_range_sanity.py --pot /path/to/pot.almtp
    python tests/short_range_sanity.py --pot /path/to/pot.almtp --a0 5.76
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import calc_efs, write_bare_cfg

DEFAULT_TRAIN_CONFIG = "config/training.yaml"


def _parse_energies(path: Path) -> list[float]:
    """Extract Energy from every CFG block, in file order."""
    energies = []
    for block in re.split(r"(?=BEGIN_CFG\b)", path.read_text()):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        em = re.search(r"\bEnergy\b\s*\n\s*([-\d.eE+]+)", block)
        if em:
            energies.append(float(em.group(1)))
    return energies


def _eval(mlp: str, pot: str, configs: list, workdir: Path, tag: str) -> list[float]:
    in_cfg = workdir / f"{tag}.cfg"
    out_cfg = workdir / f"{tag}_efs.cfg"
    write_bare_cfg(configs, in_cfg)
    rc = calc_efs(mlp, pot, in_cfg, out_cfg, quiet=True)
    if rc != 0:
        print(f"  Warning: mlp calculate_efs returned {rc} for {tag}")
    energies = _parse_energies(out_cfg)
    if len(energies) != len(configs):
        print(f"  Warning: expected {len(configs)} energies, parsed {len(energies)} for {tag}")
    return energies


# ---------------------------------------------------------------------------
# Check 1 - dimer curve
# ---------------------------------------------------------------------------

def dimer_curve(mlp: str, pot: str, outdir: Path,
                 r_min: float = 0.6, r_max: float = 5.0, npoints: int = 60) -> None:
    print("\n  [1/3] Dimer curve (isolated pair, vacuum-like box)")
    box = 25.0  # >> cutoff, avoids periodic self-interaction
    cell = np.diag([box, box, box])
    rs = np.linspace(r_min, r_max, npoints)
    center = np.array([box / 2, box / 2, box / 2])
    configs = [
        (cell, np.array([center - [r / 2, 0, 0], center + [r / 2, 0, 0]]), [0, 0])
        for r in rs
    ]
    energies = _eval(mlp, pot, configs, outdir, "dimer")

    txt = outdir / "dimer_curve.txt"
    with open(txt, "w") as f:
        f.write("# r[A]  E_total[eV]\n")
        for r, e in zip(rs, energies):
            f.write(f"  {r:.4f}  {e:.6f}\n")
    print(f"    written to {txt}")

    imin = int(np.argmin(energies))
    print(f"    global min: r={rs[imin]:.3f} A, E={energies[imin]:.4f} eV")
    _plot_xy(rs, energies, "Separation r (A)", "Total energy (eV)",
             "Dimer curve (isolated pair)", outdir / "dimer_curve.png",
             vline=rs[imin])


# ---------------------------------------------------------------------------
# Check 2 - bulk uniform compression
# ---------------------------------------------------------------------------

def bulk_compression(mlp: str, pot: str, outdir: Path, a0: float,
                      scale_min: float = 0.35, scale_max: float = 1.05,
                      npoints: int = 60) -> tuple[float, float]:
    print("\n  [2/3] Bulk uniform compression (2x2x2 diamond supercell)")
    from ase.build import bulk
    atoms0 = bulk("Ge", crystalstructure="diamond", a=a0, cubic=True).repeat((2, 2, 2))
    cell0 = atoms0.get_cell().array.copy()
    pos0 = atoms0.get_positions()
    n = len(pos0)
    types = [0] * n

    scales = np.linspace(scale_min, scale_max, npoints)
    d1nn0 = a0 * np.sqrt(3.0) / 4.0
    configs = [(cell0 * s, pos0 * s, types) for s in scales]
    energies = _eval(mlp, pot, configs, outdir, "bulk_compress")
    epa = np.array(energies) / n
    d1nn = scales * d1nn0

    txt = outdir / "bulk_compression.txt"
    with open(txt, "w") as f:
        f.write("# scale  d1NN[A]  E_per_atom[eV]\n")
        for s, d, e in zip(scales, d1nn, epa):
            f.write(f"  {s:.4f}  {d:.4f}  {e:.6f}\n")
    print(f"    written to {txt}")

    imin = int(np.argmin(epa))
    mu_Ge = float(epa[-1])  # least-compressed point ~ equilibrium-ish reference
    print(f"    global min: d1NN={d1nn[imin]:.3f} A, E/atom={epa[imin]:.4f} eV/atom "
          f"(reference at largest scale: {mu_Ge:.4f} eV/atom)")
    if epa[imin] < epa[-1] and d1nn[imin] < d1nn0 * 0.9:
        print("    FLAG: compressed configuration is lower energy than the "
              "least-compressed reference -> spurious short-range attraction")

    _plot_xy(d1nn, epa, "Nearest-neighbour distance (A)", "Energy per atom (eV)",
             "Bulk uniform compression", outdir / "bulk_compression.png",
             vline=d1nn0, vline_label="equilibrium d1NN")
    return d1nn[imin], epa[imin]


# ---------------------------------------------------------------------------
# Check 3 - bond-centre local environment scan
# ---------------------------------------------------------------------------

def bond_centre_scan(mlp: str, pot: str, outdir: Path, a0: float,
                      t_min: float = 0.2, t_max: float = 0.5, npoints: int = 40) -> None:
    print("\n  [3/3] Bond-centre local environment scan (217-atom supercell)")
    from ase.build import bulk
    atoms0 = bulk("Ge", crystalstructure="diamond", a=a0, cubic=True).repeat((3, 3, 3))
    cell = atoms0.get_cell().array.copy()
    pos0 = atoms0.get_positions()
    n_bulk = len(pos0)

    host0 = np.array([0.0, 0.0, 0.0])
    host1 = np.array([a0 / 4, a0 / 4, a0 / 4])
    d_bond = float(np.linalg.norm(host1 - host0))

    ts = np.linspace(t_min, t_max, npoints)
    configs = []
    for t in ts:
        interstitial = host0 + t * (host1 - host0)
        pos = np.vstack([pos0, interstitial])
        configs.append((cell, pos, [0] * (n_bulk + 1)))

    # mu_Ge from the same potential, same supercell size, no defect
    mu_configs = [(cell, pos0, [0] * n_bulk)]
    mu_energy = _eval(mlp, pot, mu_configs, outdir, "bulk_ref")[0]
    mu_Ge = mu_energy / n_bulk

    energies = _eval(mlp, pot, configs, outdir, "bond_centre_scan")
    n_defect = n_bulk + 1
    dists = ts * d_bond
    Ef = np.array(energies) - n_defect * mu_Ge

    txt = outdir / "bond_centre_scan.txt"
    with open(txt, "w") as f:
        f.write(f"# mu_Ge = {mu_Ge:.6f} eV/atom  (from {n_bulk}-atom bulk cell)\n")
        f.write(f"# bond length d_bond = {d_bond:.4f} A, exact bond-centre at t=0.5 "
                f"(dist={d_bond/2:.4f} A)\n")
        f.write("# t  dist_to_host0[A]  E_total[eV]  E_f[eV]\n")
        for t, d, e, ef in zip(ts, dists, energies, Ef):
            f.write(f"  {t:.4f}  {d:.4f}  {e:.6f}  {ef:.6f}\n")
    print(f"    written to {txt}")

    imin = int(np.argmin(Ef))
    print(f"    min E_f = {Ef[imin]:.3f} eV at dist={dists[imin]:.3f} A "
          f"(exact bond-centre dist={d_bond/2:.3f} A, E_f there = {Ef[-1]:.3f} eV)")
    if Ef[imin] < 0:
        print(f"    FLAG: negative formation energy at dist={dists[imin]:.3f} A "
              f"-> spurious false minimum below bulk reference")

    _plot_xy(dists, Ef, "Interstitial distance to nearest host atom (A)",
             "Formation energy E_f (eV)", "Bond-centre local environment scan",
             outdir / "bond_centre_scan.png",
             vline=d_bond / 2, vline_label="exact bond-centre", hline=0.0)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_xy(x, y, xlabel, ylabel, title, path, vline=None, vline_label=None, hline=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y, "o-", ms=4, color="tab:blue")
        if hline is not None:
            ax.axhline(hline, color="gray", lw=0.8, ls="--")
        if vline is not None:
            ax.axvline(vline, color="tab:red", lw=1.0, ls="--",
                       label=vline_label or f"{vline:.3f}")
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"    plot saved to {path}")
    except ImportError as e:
        print(f"    matplotlib not available ({e}); skipping plot")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(pot: str, outdir: Path, mlp: str, a0: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Short-range sanity checks for potential: {pot}")
    print(f"a0 = {a0} A, outdir = {outdir}")

    dimer_curve(mlp, pot, outdir)
    bulk_compression(mlp, pot, outdir, a0)
    bond_centre_scan(mlp, pot, outdir, a0)

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short-range sanity checks (dimer, bulk compression, "
                     "bond-centre scan) for an MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--a0", type=float, default=5.76, help="Lattice constant A")
    parser.add_argument("--outdir", default="results/tests/defect_formation/short_range_sanity")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML config (provides mlp_binary)")
    parser.add_argument("--mlp", default=None, help="Override mlp binary path")
    args = parser.parse_args()

    mlp = args.mlp
    if mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    run(pot=args.pot, outdir=Path(args.outdir), mlp=mlp, a0=args.a0)


if __name__ == "__main__":
    main()
