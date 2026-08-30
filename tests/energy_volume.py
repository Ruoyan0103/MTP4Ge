"""
Energy-volume curve quality test for MTP potentials.

Generates scaled copies of a reference structure, evaluates energy via
mlp calculate_efs, outputs E(V) data, and optionally fits a Birch-Murnaghan EOS.

Usage:
    python tests/energy_volume.py --pot results/potentials/pot.almtp
    python tests/energy_volume.py --pot results/potentials/pot.almtp \\
        --struct data/relaxed.cfg --npoints 25 --vrange 0.3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import calc_efs, load_structure, write_bare_cfg

DEFAULT_TRAIN_CONFIG = "config/training.yaml"
EV_A3_TO_GPA = 160.21766   # 1 eV/Å³ = 160.21766 GPa


# ---------------------------------------------------------------------------
# CFG parsing
# ---------------------------------------------------------------------------

def _parse_ev_from_cfg(path: Path) -> list[tuple[float, float]]:
    """Extract (volume [Å³], energy [eV]) pairs from an EFS CFG file."""
    import re
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
            a = np.array([[float(x) for x in row.split()]
                          for row in sm.group(1).strip().splitlines()]).flatten()
            vol = abs(
                a[0] * (a[4] * a[8] - a[5] * a[7])
                - a[1] * (a[3] * a[8] - a[5] * a[6])
                + a[2] * (a[3] * a[7] - a[4] * a[6])
            )
            ev.append((vol, float(em.group(1))))
    return ev


# ---------------------------------------------------------------------------
# Birch-Murnaghan EOS
# ---------------------------------------------------------------------------

def _bm3(V, E0, V0, B0, B0p):
    """Third-order Birch-Murnaghan EOS."""
    eta = (V0 / V) ** (2 / 3)
    return E0 + (9 * V0 * B0 / 16) * (
        (eta - 1) ** 3 * B0p + (eta - 1) ** 2 * (6 - 4 * eta)
    )


def _fit_bm_eos(volumes, energies):
    """Fit BM EOS; return (E0, V0, B0 [eV/Å³], B0p) or None on failure."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    v, e = np.array(volumes), np.array(energies)
    i0 = np.argmin(e)
    p0 = [e[i0], v[i0], 75 / EV_A3_TO_GPA, 4.0]
    try:
        popt, _ = curve_fit(_bm3, v, e, p0=p0, maxfev=10000)
        return popt
    except Exception as err:
        print(f"  EOS fit failed: {err}")
        return None


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    struct: str | None,
    outdir: Path,
    npoints: int,
    vrange: float,
    mlp: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cell0, pos0, types = load_structure(struct)
    n_atoms = len(pos0)
    V0_guess = abs(np.linalg.det(cell0))
    print(f"  Structure: {n_atoms} atoms  |  V₀ ≈ {V0_guess:.4f} Å³")

    # Lattice-parameter scales so volumes span ±vrange/2 around V₀
    scale_min = (1 - vrange / 2) ** (1 / 3)
    scale_max = (1 + vrange / 2) ** (1 / 3)
    scales = np.linspace(scale_min, scale_max, npoints)

    configs = [(cell0 * s, pos0 * s, types) for s in scales]

    in_cfg  = outdir / "deformed.cfg"
    out_cfg = outdir / "deformed_efs.cfg"
    write_bare_cfg(configs, in_cfg)

    rc = calc_efs(mlp, pot, in_cfg, out_cfg)
    if rc != 0:
        print(f"  Warning: mlp calculate_efs returned {rc}")
    if not out_cfg.exists():
        print("  Error: no EFS output produced.", file=sys.stderr)
        sys.exit(1)

    ev = _parse_ev_from_cfg(out_cfg)
    if not ev:
        print("  Error: no E-V pairs extracted.", file=sys.stderr)
        sys.exit(1)
    ev.sort()
    volumes, energies = zip(*ev)
    volumes = list(volumes)
    energies = list(energies)

    # Write E_V.txt
    ev_txt = outdir / "E_V.txt"
    with open(ev_txt, "w") as f:
        f.write("# Volume[A^3]  Energy[eV]  Energy_per_atom[eV/atom]\n")
        for v, e in zip(volumes, energies):
            f.write(f"  {v:12.6f}  {e:14.8f}  {e/n_atoms:14.8f}\n")
    print(f"  E-V data ({len(ev)} points) written to {ev_txt}")

    # Fit Birch-Murnaghan EOS
    eos = _fit_bm_eos(volumes, energies)
    if eos is not None:
        E0, V0, B0, B0p = eos
        B0_GPa = B0 * EV_A3_TO_GPA
        # Diamond cubic: 8 atoms per conventional cell → a = (V0 * 8/n_atoms)^(1/3)
        a0 = (V0 * 8 / n_atoms) ** (1 / 3) if n_atoms <= 8 else None

        eos_txt = outdir / "eos_fit.txt"
        with open(eos_txt, "w") as f:
            f.write("Birch-Murnaghan 3rd-order EOS fit\n")
            f.write(f"Potential: {pot}\n")
            f.write(f"E0  = {E0:.6f} eV\n")
            f.write(f"V0  = {V0:.4f} Å³  ({V0/n_atoms:.4f} Å³/atom)\n")
            if a0:
                f.write(f"a0  = {a0:.4f} Å  (diamond cubic: a = (4V_cell/N)^1/3)\n")
            f.write(f"B0  = {B0_GPa:.2f} GPa\n")
            f.write(f"B0' = {B0p:.4f}\n")
        print(
            f"  BM EOS: V0 = {V0:.2f} Å³, B0 = {B0_GPa:.1f} GPa, B0' = {B0p:.2f}"
        )
        if a0:
            print(f"         a0 = {a0:.4f} Å  (exp. Ge: 5.658 Å)")
        print(f"  EOS parameters written to {eos_txt}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vpa = [v / n_atoms for v in volumes]
        epa = [e / n_atoms for e in energies]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(vpa, epa, "o", ms=5, color="tab:blue", label="MTP")
        if eos is not None:
            v_fit = np.linspace(min(volumes), max(volumes), 300)
            e_fit = _bm3(v_fit, *eos)
            ax.plot(
                v_fit / n_atoms, e_fit / n_atoms,
                "--", color="tab:red", lw=1.5, label="BM EOS fit",
            )
        ax.set_xlabel("Volume per atom (Å³)")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title("Energy-Volume Curve")
        ax.legend()
        fig.tight_layout()
        plot_path = outdir / "E_V.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {plot_path}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Energy-volume curve test for MTP potential"
    )
    parser.add_argument("--pot",     required=True,
                        help="Path to potential (.almtp)")
    parser.add_argument("--struct",  default=None,
                        help="Reference structure (CFG / XYZ / LAMMPS data). "
                             "Default: 2-atom ASE diamond Ge (a=5.658 Å)")
    parser.add_argument("--outdir",  default="results/tests/energy_volume",
                        help="Output directory")
    parser.add_argument("--npoints", type=int,   default=20,
                        help="Number of volume points (default: 20)")
    parser.add_argument("--vrange",  type=float, default=0.30,
                        help="Fractional volume range (default: 0.30 → ±15%%)")
    parser.add_argument("--config",  default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML config (provides mlp_binary)")
    parser.add_argument("--mlp",     default=None,
                        help="Override mlp binary path")
    args = parser.parse_args()

    mlp = args.mlp
    if mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    run(
        pot=args.pot,
        struct=args.struct,
        outdir=Path(args.outdir),
        npoints=args.npoints,
        vrange=args.vrange,
        mlp=mlp,
    )


if __name__ == "__main__":
    import yaml
    main()
