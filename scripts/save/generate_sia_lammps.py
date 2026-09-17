"""
Generate LAMMPS data files for each SIA type and pre-strained bulk cells.

These files are used as lammps_structure seeds for active-learning MD runs
(via --structure flag in active_learning.py), replacing manual DFT enumeration.

SIA structures (64+1 atoms, 2×2×2 conventional supercell):
  data/sia/tet.lammps   — tetrahedral interstitial
  data/sia/hex.lammps   — hexagonal interstitial
  data/sia/split.lammps — <110> dumbbell
  data/sia/bond.lammps  — bond-centre interstitial

Strained bulk cells (64 atoms, 2×2×2 conventional supercell):
  data/strained/vol_pm2pct.lammps  — ±2% isotropic volume strain

Usage (from repo root):
    python scripts/save/generate_sia_lammps.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    from ase.build import bulk
    from ase.io import write as ase_write
except ImportError as e:
    print(f"ERROR: {e}. Activate the mtp4ge environment.", file=sys.stderr)
    sys.exit(1)

GE_A0 = 5.7620   # Å — PBE equilibrium lattice constant


# ---------------------------------------------------------------------------
# SIA builder (same geometry logic as generate_sia_configs.py)
# ---------------------------------------------------------------------------

def build_sia_structures(n: int = 2) -> dict[str, object]:
    prim = bulk("Ge", crystalstructure="diamond", a=GE_A0, cubic=True)
    sup = prim.repeat(n)
    cell = sup.get_cell()
    positions = sup.get_positions()

    structures = {}

    # Tetrahedral (T) — void at (1/2, 1/2, 1/2) of the conventional cell
    t_pos = cell @ np.array([1 / (2 * n), 1 / (2 * n), 1 / (2 * n)])
    t = sup.copy(); t.append("Ge"); t.positions[-1] = t_pos
    structures["tet"] = t

    # Hexagonal (H) — channel at (5/8, 5/8, 5/8) of the conventional cell
    h_pos = cell @ np.array([5 / (8 * n), 5 / (8 * n), 5 / (8 * n)])
    h = sup.copy(); h.append("Ge"); h.positions[-1] = h_pos
    structures["hex"] = h

    # <110> dumbbell (split/X) — pair displaced along [110] from site 0
    d_110 = 1.2   # Å half-separation
    origin = positions[0].copy()
    dir110 = np.array([1, 1, 0], dtype=float) / np.sqrt(2)
    x = sup.copy()
    x.positions[0] = origin + d_110 * dir110
    x.append("Ge"); x.positions[-1] = origin - d_110 * dir110
    structures["split"] = x

    # Bond-centre (B) — midpoint of nearest-neighbour bond from site 0
    dists = np.linalg.norm(positions[1:] - positions[0], axis=1)
    nn_idx = np.argmin(dists) + 1
    bond_mid = 0.5 * (positions[0] + positions[nn_idx])
    b = sup.copy(); b.append("Ge"); b.positions[-1] = bond_mid
    structures["bond"] = b

    return structures


# ---------------------------------------------------------------------------
# Strained bulk builder
# ---------------------------------------------------------------------------

def build_strained_bulk(n: int = 2, strains: list[float] = None) -> dict[str, object]:
    """Isotropic volume strains for AL exploration of compressed/expanded states."""
    if strains is None:
        strains = [-0.02, 0.02]
    prim = bulk("Ge", crystalstructure="diamond", a=GE_A0, cubic=True)
    sup = prim.repeat(n)
    structures = {}
    for e in strains:
        s = (1 + e) ** (1 / 3)
        strained = sup.copy()
        strained.set_cell(strained.get_cell() * s, scale_atoms=True)
        label = f"vol_{'p' if e > 0 else 'm'}{abs(int(e * 100)):02d}pct"
        structures[label] = strained
    return structures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sia_dir = Path("data/sia")
    sia_dir.mkdir(parents=True, exist_ok=True)
    strained_dir = Path("data/strained")
    strained_dir.mkdir(parents=True, exist_ok=True)

    print("=== SIA LAMMPS structures ===")
    for label, atoms in build_sia_structures(n=2).items():
        out = sia_dir / f"{label}.lammps"
        ase_write(str(out), atoms, format="lammps-data", atom_style="atomic")
        print(f"  {label}: {len(atoms)} atoms → {out}")

    print("\n=== Strained bulk LAMMPS structures ===")
    for label, atoms in build_strained_bulk(n=2).items():
        out = strained_dir / f"{label}.lammps"
        ase_write(str(out), atoms, format="lammps-data", atom_style="atomic")
        print(f"  {label}: {len(atoms)} atoms → {out}")

    print("\nAll structures written. Use with AL via:")
    print("  bash scripts/save/03_active_learning_sia.sh results/potentials/pot_al.almtp")


if __name__ == "__main__":
    main()
