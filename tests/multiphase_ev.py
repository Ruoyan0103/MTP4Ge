"""Multi-phase energy-volume curves for Ge crystal phases.

Computes E vs V/atom for 7 Ge crystal phases (Fig 3a of GAP-Ge paper):
diamond, hexagonal diamond (lonsdaleite), hcp, fcc, bcc, beta-Sn, bc8.

Usage:
    python tests/multiphase_ev.py --pot results/potentials/pot.almtp
    python tests/multiphase_ev.py --pot results/potentials/pot.almtp \\
        --outdir results/tests/multiphase_ev --n-volumes 21 --vol-range 0.40
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import _mlp_env, calc_efs, write_bare_cfg

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"


# ---------------------------------------------------------------------------
# Structure builders — return (cell [3×3], positions [N×3], types [list])
# ---------------------------------------------------------------------------

def build_phases(a0_diamond: float = 5.658) -> dict:
    """Build 7 Ge crystal phases at reference volumes (a0 for diamond)."""
    from ase.build import bulk

    phases = {}

    # --- diamond (Fd-3m, 8 atoms conventional cell) ---
    atoms = bulk("Ge", "diamond", a=a0_diamond, cubic=True)
    phases["diamond"] = (
        np.array(atoms.get_cell()),
        atoms.get_positions(),
        [0] * len(atoms),
    )

    # --- lonsdaleite / hexagonal diamond (P6_3/mmc, 4 atoms) ---
    # a = a0/√2, c = a·√(8/3); Wyckoff 2c and 2d
    a_hd = a0_diamond / np.sqrt(2)
    c_hd = a_hd * np.sqrt(8.0 / 3.0)
    hd_cell = np.array([
        [a_hd, 0.0, 0.0],
        [-a_hd / 2.0, a_hd * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c_hd],
    ])
    # 4f Wyckoff positions for lonsdaleite (P6_3/mmc), z ≈ 1/16
    hd_frac = np.array([
        [1 / 3, 2 / 3, 1 / 16],
        [2 / 3, 1 / 3, 9 / 16],
        [1 / 3, 2 / 3, 7 / 16],
        [2 / 3, 1 / 3, 15 / 16],
    ])
    phases["lonsdaleite"] = (hd_cell, hd_frac @ hd_cell, [0] * 4)

    # --- hcp (P6_3/mmc, 2 atoms) ---
    # ideal c/a = √(8/3); a ≈ 1NN distance in diamond / √2
    a_hcp = a0_diamond * np.sqrt(3.0) / 4.0 * np.sqrt(2.0)  # ≈ 2.83 Å
    c_hcp = a_hcp * np.sqrt(8.0 / 3.0)
    atoms_hcp = bulk("Ge", "hcp", a=a_hcp, c=c_hcp)
    phases["hcp"] = (
        np.array(atoms_hcp.get_cell()),
        atoms_hcp.get_positions(),
        [0] * len(atoms_hcp),
    )

    # --- fcc (Fm-3m, 1 atom primitive cell) ---
    atoms_fcc = bulk("Ge", "fcc", a=a0_diamond * 0.72)
    phases["fcc"] = (
        np.array(atoms_fcc.get_cell()),
        atoms_fcc.get_positions(),
        [0] * len(atoms_fcc),
    )

    # --- bcc (Im-3m, 1 atom primitive cell) ---
    atoms_bcc = bulk("Ge", "bcc", a=a0_diamond * 0.57)
    phases["bcc"] = (
        np.array(atoms_bcc.get_cell()),
        atoms_bcc.get_positions(),
        [0] * len(atoms_bcc),
    )

    # --- beta-Sn / Ge-II (I4_1/amd, 4 atoms) ---
    # High-pressure phase; a ≈ 4.87 Å, c ≈ 2.59 Å
    a_sn = 4.87
    c_sn = 2.59
    sn_cell = np.array([[a_sn, 0.0, 0.0], [0.0, a_sn, 0.0], [0.0, 0.0, c_sn]])
    sn_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.25],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.75],
    ])
    phases["beta_sn"] = (sn_cell, sn_frac @ sn_cell, [0] * 4)

    # --- bc8 / Ge-III (Ia-3, 8 atoms) ---
    # High-pressure phase; a ≈ 6.675 Å, x ≈ 0.100
    a_bc8 = 6.675
    x = 0.100
    bc8_frac = np.array([
        [x, x, x],
        [-x + 0.5, -x, x + 0.5],
        [-x, x + 0.5, -x + 0.5],
        [x + 0.5, -x + 0.5, -x],
        [-x, -x, -x],
        [x + 0.5, x, -x + 0.5],
        [x, -x + 0.5, x + 0.5],
        [-x + 0.5, x + 0.5, x],
    ])
    bc8_cell = np.diag([a_bc8, a_bc8, a_bc8])
    phases["bc8"] = (bc8_cell, bc8_frac @ bc8_cell, [0] * 8)

    return phases


# ---------------------------------------------------------------------------
# EFS multi-block parser
# ---------------------------------------------------------------------------

def _parse_energies_volumes(path: Path) -> list[tuple[float, float]]:
    """Return list of (volume [Å³], energy [eV]) from a multi-block EFS CFG."""
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
# Main workflow
# ---------------------------------------------------------------------------

def run(pot: str, outdir: Path, mlp: str, n_volumes: int, vol_range: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    phases = build_phases()
    vol_fracs = np.linspace(1.0 - vol_range / 2.0, 1.0 + vol_range / 2.0, n_volumes)

    results = {}  # name → (vol_per_atom array, energy_per_atom array)

    for name, (cell0, pos0, types) in phases.items():
        n_atom = len(pos0)
        vol0 = abs(np.linalg.det(cell0))
        print(f"\n  [{name}]  {n_atom} atoms  V₀ = {vol0:.2f} Å³")

        configs = []
        for s in vol_fracs:
            scale = s ** (1.0 / 3.0)
            configs.append((cell0 * scale, pos0 * scale, types))

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

        vols, energies = zip(*sorted(ev))
        results[name] = (
            np.array(vols) / n_atom,
            np.array(energies) / n_atom,
        )
        e_min = np.min(results[name][1])
        print(f"    E_min/atom = {e_min:.4f} eV  ({len(ev)} points)")

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

        # Shift each phase so its minimum is at 0 (relative energy)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for (name, (vpa, epa)), color in zip(results.items(), colors):
            epa_rel = epa - epa.min()
            ax.plot(vpa, epa_rel, "o-", label=name, color=color, markersize=4, lw=1.5)

        ax.set_xlabel("Volume per atom (Å³)", fontsize=12)
        ax.set_ylabel("Energy per atom relative to minimum (eV)", fontsize=12)
        ax.set_title("MTP Multi-Phase E-V Curves — Ge", fontsize=13)
        ax.legend(fontsize=9, ncol=2)
        ax.set_ylim(-0.1, 3.0)
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
    parser.add_argument("--outdir", default="results/tests/multiphase_ev")
    parser.add_argument("--n-volumes", type=int, default=21,
                        help="Number of volume points per phase (default: 21)")
    parser.add_argument("--vol-range", type=float, default=0.40,
                        help="Fractional volume range (default: 0.40 → ±20%%)")
    parser.add_argument("--mlp", default=MLP_DEFAULT, help="mlp binary path")
    args = parser.parse_args()

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        mlp=args.mlp,
        n_volumes=args.n_volumes,
        vol_range=args.vol_range,
    )


if __name__ == "__main__":
    main()
