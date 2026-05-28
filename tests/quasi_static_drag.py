"""Quasi-static drag test for Ge MTP potential.

Displaces one atom stepwise along [100], [110], [111], [031] crystal directions
and computes ΔE and the parallel force component at each step.
All displacements for a given direction are evaluated in a single mlp call
(batch CFG) for efficiency.

Expected results:
  ΔE > 0 everywhere (moving off lattice site costs energy)
  Force near zero at equilibrium, peaks near interstitial sites

Usage:
    python tests/quasi_static_drag.py --pot results/potentials/pot.almtp
    python tests/quasi_static_drag.py --pot results/potentials/pot.almtp \\
        --a0 5.779 --n-steps 80 --step-size 0.05
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import _mlp_env, calc_efs, write_bare_cfg

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"

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

def _max_safe_disp(pos: np.ndarray, cell: np.ndarray, drag_idx: int,
                   uvec: np.ndarray, min_sep: float = 1.5) -> float:
    """Return the max displacement before any inter-atom distance drops below min_sep.

    Uses straight-line drag geometry: dist²(d) = perp² + (proj - d)²
    Collision when dist(d) = min_sep → d = proj - sqrt(min_sep² - perp²).
    Only fires if the drag path passes within min_sep of another atom (perp < min_sep)
    and the atom is ahead (proj > 0).
    """
    pos0 = pos[drag_idx]
    others = np.delete(pos, drag_idx, axis=0)
    cell_inv = np.linalg.inv(cell)
    min_d = np.inf
    for other in others:
        diff = other - pos0
        frac = diff @ cell_inv
        frac -= np.round(frac)
        diff_mic = frac @ cell
        proj = float(np.dot(diff_mic, uvec))
        if proj <= 0:
            continue  # atom is behind or at start — no collision ahead
        perp2 = float(np.dot(diff_mic, diff_mic)) - proj ** 2
        if perp2 < min_sep ** 2:
            # Drag path passes within min_sep of this atom
            lim = proj - np.sqrt(max(0.0, min_sep ** 2 - perp2))
            min_d = min(min_d, max(0.1, lim))
    return float(min_d)


def drag_scan(
    cell: np.ndarray,
    pos: np.ndarray,
    types: list,
    drag_idx: int,
    direction: np.ndarray,
    n_steps: int,
    step_size: float,
    mlp: str,
    pot: str,
    outdir: Path,
    label: str,
    max_disp: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displace atom drag_idx along direction; return (displacements, dE, F_parallel).

    All n_steps configs are written to a single CFG and evaluated in one mlp call.
    Displacement is automatically capped at max_disp or the safe collision limit.
    """
    uvec = direction / np.linalg.norm(direction)
    pos0 = pos[drag_idx].copy()

    # Limit displacement to avoid atom-atom overlap
    safe = _max_safe_disp(pos, cell, drag_idx, uvec, min_sep=1.5)
    effective_max = min(max_disp, safe, n_steps * step_size)
    actual_steps = max(2, int(effective_max / step_size))
    if actual_steps < n_steps:
        print(f"    [{label}] capped at {effective_max:.2f} Å ({actual_steps} steps) "
              f"to avoid atom overlap")

    configs = []
    disp_vals = np.arange(actual_steps) * step_size
    for d in disp_vals:
        new_pos = pos.copy()
        new_pos[drag_idx] = pos0 + uvec * d
        configs.append((cell, new_pos, types))

    in_cfg = outdir / f"drag_{label}_in.cfg"
    out_cfg = outdir / f"drag_{label}_out.cfg"
    write_bare_cfg(configs, in_cfg)
    rc = calc_efs(mlp, pot, in_cfg, out_cfg, quiet=True)
    if rc != 0:
        raise RuntimeError(f"mlp calculate_efs returned {rc} for direction {label}")

    efs_data = _parse_all_efs(out_cfg)
    if len(efs_data) != actual_steps:
        raise RuntimeError(
            f"Expected {n_steps} EFS blocks, got {len(efs_data)} for {label}"
        )

    energies = np.array([e for e, _ in efs_data])
    f_parallel = np.array([
        float(np.dot(forces[drag_idx], uvec))
        for _, forces in efs_data
        if len(forces) > drag_idx
    ])
    # Trim disp_vals to match actual efs output length
    disp_vals = disp_vals[:len(energies)]

    dE = energies - energies[0]
    return disp_vals, dE, f_parallel


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    outdir: Path,
    mlp: str,
    a0: float,
    n_steps: int,
    step_size: float,
    max_disp: float = 2.0,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    cell, pos, types = build_drag_supercell(a0)
    n_atoms = len(pos)
    drag_idx = 0  # drag the first atom (at/near origin)
    print(f"  {n_atoms}-atom 2×2×2 supercell  |  a0 = {a0:.4f} Å")
    print(f"  Dragging atom {drag_idx} at {pos[drag_idx]}  |  {n_steps} steps × {step_size} Å")

    all_results = {}
    for label, direction in DRAG_DIRECTIONS.items():
        print(f"\n  [{label}]  direction = {direction / np.linalg.norm(direction)}")
        disp, dE, F_par = drag_scan(
            cell, pos, types, drag_idx, direction,
            n_steps=n_steps, step_size=step_size,
            mlp=mlp, pot=pot, outdir=outdir, label=label.strip("[]"),
            max_disp=max_disp,
        )
        all_results[label] = (disp, dE, F_par)
        i_max = int(np.argmax(dE))
        print(f"  Max ΔE = {dE[i_max]:.4f} eV at d = {disp[i_max]:.2f} Å")

    # --- Write data ---
    txt = outdir / "qsd_data.txt"
    with open(txt, "w") as f:
        f.write(f"# Quasi-static drag  |  pot: {pot}  a0={a0:.4f} Å\n")
        f.write(f"# {n_atoms} atoms  |  {n_steps} steps × {step_size} Å\n\n")
        for label, (disp, dE, F_par) in all_results.items():
            f.write(f"# direction {label}\n")
            f.write("# disp[A]  dE[eV]  F_par[eV/A]\n")
            for d, de, fp in zip(disp, dE, F_par):
                f.write(f"  {d:.4f}  {de:.6f}  {fp:.6f}\n")
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

        for col, (label, (disp, dE, F_par)) in enumerate(all_results.items()):
            ax_e = axes[0, col]
            ax_f = axes[1, col]

            ax_e.plot(disp, dE * 1000, color="tab:blue", lw=1.5)
            ax_e.axhline(0, color="gray", lw=0.8, ls="--")
            ax_e.set_ylabel("ΔE (meV)" if col == 0 else "")
            ax_e.set_title(label, fontsize=11)
            ax_e.set_xlabel("")

            ax_f.plot(disp, F_par, color="tab:orange", lw=1.5)
            ax_f.axhline(0, color="gray", lw=0.8, ls="--")
            ax_f.set_xlabel("Displacement (Å)")
            ax_f.set_ylabel("F‖ (eV/Å)" if col == 0 else "")

        fig.suptitle(f"Quasi-static drag — Ge MTP  (a₀={a0:.3f} Å)", fontsize=13)
        fig.tight_layout()
        png = outdir / "qsd.png"
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
        description="Quasi-static drag test for Ge MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--a0", type=float, default=5.779,
                        help="Lattice constant Å (default: 5.779)")
    parser.add_argument("--n-steps", type=int, default=80,
                        help="Number of drag steps per direction (default: 80)")
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="Displacement per step in Å (default: 0.05)")
    parser.add_argument("--max-disp", type=float, default=2.0,
                        help="Max displacement per direction in Å (default: 2.0)")
    parser.add_argument("--outdir", default="results/tests/quasi_static_drag")
    parser.add_argument("--mlp", default=MLP_DEFAULT, help="mlp binary path")
    args = parser.parse_args()

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        mlp=args.mlp,
        a0=args.a0,
        n_steps=args.n_steps,
        step_size=args.step_size,
        max_disp=args.max_disp,
    )


if __name__ == "__main__":
    main()
