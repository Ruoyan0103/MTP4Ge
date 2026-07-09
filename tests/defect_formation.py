"""
Point defect formation energy test for MTP potentials.

Computes formation energies of:
  V  – monovacancy (remove 1 atom, 215-atom cell)
  T  – tetrahedral interstitial  (add at void centre (a/2,a/2,a/2))
  H  – hexagonal interstitial    (add in ⟨111⟩ channel (5a/8,5a/8,5a/8))
  X  – ⟨110⟩ split dumbbell      (replace 1 atom with 2 displaced ±δ along [110])
  B  – bond-centre interstitial  (add at midpoint of nearest-neighbour bond)

All defects use a 3×3×3 conventional supercell (216 ± 1 atoms, a₀ = MTP equilibrium).
Structures are relaxed at fixed cell with BFGS using mlp relax.

Formation energy (elemental reference):
  E_f = E_defect − N_defect × μ_Ge,   μ_Ge = E_bulk / N_bulk

Reference (GAP-Ge paper, DFT-GGA, Table 3 / Fig. 4):
  Vacancy: 2.24 eV (DFT)  |  2.06 eV (GAP)

Usage:
    python -m tests.defect_formation --pot results/potentials/pot_al.almtp
    python -m tests.defect_formation --pot results/potentials/pot_al.almtp \\
        --a0 5.779 --fmax 0.0001 --max-steps 500
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import _mlp_env, calc_efs, write_bare_cfg

DEFAULT_TRAIN_CONFIG = "config/training.yaml"

REF_DIR = Path(__file__).parent / "reference"


def _load_ref_dict(filename: str) -> dict[str, float]:
    """Load a key-value reference file (name value per line, # comments)."""
    path = REF_DIR / filename
    if not path.exists():
        return {}
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = float(parts[1])
    return d

DFT_REF = _load_ref_dict("defect_formation_energy_dft.dat")
GAP_REF = _load_ref_dict("defect_formation_energy_gap.dat")



# ---------------------------------------------------------------------------
# Structure builders  (all use 3×3×3 = 216-atom conventional supercell)
# ---------------------------------------------------------------------------

def _sc(a0: float, repeat=(3, 3, 3)):
    """Return 3×3×3 ASE Atoms object."""
    from ase.build import bulk
    return bulk("Ge", crystalstructure="diamond", a=a0, cubic=True).repeat(repeat)


def build_bulk_ref(a0: float):
    """8-atom conventional cell for μ_Ge reference."""
    from ase.build import bulk
    atoms = bulk("Ge", crystalstructure="diamond", a=a0, cubic=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_vacancy(a0: float, repeat=(3, 3, 3)):
    """Remove the atom nearest the origin."""
    atoms = _sc(a0, repeat)
    dists = np.linalg.norm(atoms.get_positions(), axis=1)
    del atoms[int(np.argmin(dists))]
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_T_interstitial(a0: float, repeat=(3, 3, 3)):
    """
    Tetrahedral void: (1/2,1/2,1/2) fractional of conventional cell.
    Surrounded by 4 B-sublattice atoms at distance a√3/4 (= bond length).
    """
    from ase import Atoms
    atoms = _sc(a0, repeat)
    t_pos = np.array([a0 / 2, a0 / 2, a0 / 2])
    atoms += Atoms("Ge", positions=[t_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_H_interstitial(a0: float, repeat=(3, 3, 3)):
    """
    Hexagonal site: (0.625, 0.625, 0.625) fractional of conventional cell. 
    Surrounded by 6 atoms in a hexagonal ring.
    """
    from ase import Atoms
    atoms = _sc(a0, repeat)
    h_pos = np.array([0.625*a0, 0.625*a0, 0.625*a0])
    atoms += Atoms("Ge", positions=[h_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_X_dumbbell(a0: float, repeat=(3, 3, 3), half_disp: float = 1.2):
    """
    ⟨110⟩ split dumbbell: remove atom at origin, add two Ge displaced
    ±half_disp Å along [1,1,0]/√2.  Default half_disp ≈ bond_length/2 ≈ 1.2 Å.
    """
    from ase import Atoms
    atoms = _sc(a0, repeat)
    dists = np.linalg.norm(atoms.get_positions(), axis=1)
    idx = int(np.argmin(dists))
    center = atoms.get_positions()[idx].copy()
    del atoms[idx]
    disp = np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * half_disp
    atoms += Atoms("Ge2",
                   positions=[center + disp, center - disp],
                   cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_B_bond_center(a0: float, repeat=(3, 3, 3)):
    """
    Bond-centre site: midpoint of the bond between the atom at (0,0,0) and its
    nearest neighbour at (a/4, a/4, a/4).  Position = (a/8, a/8, a/8).
    """
    from ase import Atoms
    atoms = _sc(a0, repeat)
    b_pos = np.array([a0 / 8, a0 / 8, a0 / 8])
    atoms += Atoms("Ge", positions=[b_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


# ---------------------------------------------------------------------------
# EFS parser — energy + forces from a single-config EFS CFG
# ---------------------------------------------------------------------------

def _parse_efs(path: Path) -> tuple[float, np.ndarray]:
    """Return (energy [eV], forces [N×3 eV/Å]) from the first CFG block."""
    text = path.read_text()
    m = re.search(r"BEGIN_CFG(.*?)END_CFG", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"No CFG block found in {path}")
    block = m.group(1)

    em = re.search(r"\bEnergy\b\s*\n\s*([-\d.eE+]+)", block)
    if not em:
        raise RuntimeError(f"No Energy field in {path}")
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
            if len(parts) >= 8:   # id type x y z fx fy fz
                forces.append([float(parts[5]), float(parts[6]), float(parts[7])])

    return energy, np.array(forces)


# ---------------------------------------------------------------------------
# Fixed-cell relaxation via mlp relax (BFGS)
# ---------------------------------------------------------------------------

def relax(
    mlp: str,
    pot: str,
    cell: np.ndarray,
    pos: np.ndarray,
    types: list,
    workdir: Path,
    fmax: float = 0.0001,
    max_steps: int = 500,
) -> float:
    """
    Relax atomic positions at fixed cell using mlp relax (BFGS).
    Returns final energy [eV].
    """
    workdir.mkdir(parents=True, exist_ok=True)

    ini = workdir / "relax.ini"
    log = workdir / "relaxation.log"
    ini.write_text(
        f"relax:force_tolerance      {fmax}\n"
        f"relax:stress_tolerance     0.001\n"
        f"relax:iteration_limit      {max_steps}\n"
        f"relax:max_step             0.5\n"
        f"relax:min_step             1.0e-8\n"
        f"relax:mindist              1.0\n"
        f"relax:use_gd_method        FALSE\n"
        f"relax:freeze_small_grads   FALSE\n"
        f"relax:correct_cell         FALSE\n"
        f"relax:pressure             0.0\n"
        f"relax:log                  {log}\n"
        f"relax:init_mindist         1.0\n"
        f"relax:bfgs_wolfe_c1        1.0e-3\n"
        f"relax:bfgs_wolfe_c2        0.7\n"
        f"relax:forces_elasticity_scale    10\n"
        f"relax:stress_elasticity_scale    50\n"
    )

    in_cfg  = workdir / "in.cfg"
    out_cfg = workdir / "relaxed.cfg"
    write_bare_cfg([(cell, pos, types)], in_cfg)

    out_cfg.unlink(missing_ok=True)
    cmd = [mlp, "relax", str(pot), str(in_cfg), str(out_cfg),
           f"--relaxation_settings={ini}"]
    result = subprocess.run(cmd, env=_mlp_env(), capture_output=True, text=True)

    # mlp relax may append '.0' to the output filename
    if not out_cfg.exists():
        candidate = Path(str(out_cfg) + ".0")
        if candidate.exists():
            candidate.rename(out_cfg)

    if not out_cfg.exists():
        raise RuntimeError(
            f"mlp relax produced no output at {out_cfg}\n"
            f"stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
        )

    energy, forces = _parse_efs(out_cfg)
    max_f = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    status = "Converged" if max_f < fmax else "Max steps reached"
    print(f"    {status} | E = {energy:.6f} eV | Fmax = {max_f:.4f} eV/Å")
    return energy


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

DEFECTS = {
    "vacancy":        (build_vacancy,        -1),
    "T_interstitial": (build_T_interstitial, +1),
    "H_interstitial": (build_H_interstitial, +1),
    "X_dumbbell":     (build_X_dumbbell,     +1),
    "B_bond_center":  (build_B_bond_center,  +1),
}

DISPLAY = {
    "vacancy":        "V  (vacancy)",
    "T_interstitial": "T  (tetrahedral)",
    "H_interstitial": "H  (hexagonal)",
    "X_dumbbell":     "X  (⟨110⟩ dumbbell)",
    "B_bond_center":  "B  (bond-centre)",
}


def run(pot: str, outdir: Path, mlp: str, a0: float,
        fmax: float, max_steps: int, no_relax: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Bulk reference ──
    print("\n  [Bulk reference — 8-atom cell]")
    b_cell, b_pos, b_types = build_bulk_ref(a0)
    b_out = outdir / "bulk"
    b_out.mkdir(parents=True, exist_ok=True)
    if no_relax:
        write_bare_cfg([(b_cell, b_pos, b_types)], b_out / "bulk.cfg")
        calc_efs(mlp, pot, b_out / "bulk.cfg", b_out / "bulk_efs.cfg")
        e_bulk, _ = _parse_efs(b_out / "bulk_efs.cfg")
    else:
        e_bulk = relax(mlp, pot, b_cell, b_pos, b_types, b_out,
                       fmax=fmax, max_steps=max_steps)
    n_bulk = len(b_pos)
    mu = e_bulk / n_bulk
    print(f"  μ_Ge = {mu:.6f} eV/atom")

    # ── Defects ──
    results = {}
    for name, (builder, _) in DEFECTS.items():
        print(f"\n  [{DISPLAY[name]}]")
        d_cell, d_pos, d_types = builder(a0)
        n_def = len(d_pos)
        print(f"    {n_def} atoms | cell {tuple(round(x,3) for x in d_cell.diagonal())} Å")
        d_out = outdir / name
        d_out.mkdir(parents=True, exist_ok=True)
        try:
            if no_relax:
                write_bare_cfg([(d_cell, d_pos, d_types)], d_out / "defect.cfg")
                calc_efs(mlp, pot, d_out / "defect.cfg", d_out / "defect_efs.cfg")
                e_def, _ = _parse_efs(d_out / "defect_efs.cfg")
            else:
                e_def = relax(mlp, pot, d_cell, d_pos, d_types, d_out,
                              fmax=fmax, max_steps=max_steps)
            e_f = e_def - n_def * mu
            results[name] = {"n": n_def, "E_total": e_def, "E_f": e_f}
            dft_str = f"  (DFT ref: {DFT_REF[name]:.2f} eV)" if name in DFT_REF else ""
            print(f"    E_f = {e_f:.3f} eV{dft_str}")
        except RuntimeError as exc:
            print(f"    FAILED: {exc}")
            results[name] = {"n": n_def, "E_total": None, "E_f": None}

    # ── Write output file ──
    out_txt = outdir / "defect_formation_energies.txt"
    relax_note = "no relaxation" if no_relax else f"BFGS (mlp relax) fmax={fmax} eV/Å"
    with open(out_txt, "w") as f:
        f.write(f"Point defect formation energies — MTP: {pot}\n")
        f.write(f"Lattice constant a₀ = {a0:.4f} Å | Relaxation: {relax_note}\n")
        f.write(f"μ_Ge = {mu:.6f} eV/atom  (from {n_bulk}-atom bulk cell)\n\n")

        f.write(f"{'Defect':<22} {'N_atoms':>8} {'E_f (eV)':>10} "
                f"{'DFT (eV)':>10} {'GAP-ref (eV)':>13}\n")
        f.write("-" * 66 + "\n")
        for name, r in results.items():
            dft = f"{DFT_REF[name]:.2f}" if name in DFT_REF else "—"
            gap = f"{GAP_REF[name]:.2f}" if name in GAP_REF else "—"
            ef_str = f"{r['E_f']:>10.3f}" if r['E_f'] is not None else "     FAILED"
            f.write(f"{DISPLAY[name]:<22} {r['n']:>8} {ef_str} "
                    f"{dft:>10} {gap:>13}\n")

        f.write("\n--- SIA relative stability (w.r.t. X dumbbell) ---\n")
        if "X_dumbbell" in results and results["X_dumbbell"]["E_f"] is not None:
            e_x = results["X_dumbbell"]["E_f"]
            for name in ["T_interstitial", "H_interstitial", "B_bond_center"]:
                if name in results and results[name]["E_f"] is not None:
                    delta = results[name]["E_f"] - e_x
                    f.write(f"  ΔE({DISPLAY[name]} − X) = {delta:+.3f} eV\n")
        f.write("\nExpected SIA ordering (DFT/GAP): X < H ≈ T < B\n")

    # ── Console summary ──
    print(f"\n  {'─'*60}")
    print(f"  {'Defect':<22} {'E_f (eV)':>10}  {'DFT ref':>9}  {'GAP ref':>9}")
    print(f"  {'─'*60}")
    for name, r in results.items():
        dft = f"{DFT_REF[name]:.2f}" if name in DFT_REF else "  —  "
        gap = f"{GAP_REF[name]:.2f}" if name in GAP_REF else "  —  "
        ef_str = f"{r['E_f']:>10.3f}" if r['E_f'] is not None else "     FAILED"
        print(f"  {DISPLAY[name]:<22} {ef_str}  {dft:>9}  {gap:>9}")
    print(f"  {'─'*60}")

    if "X_dumbbell" in results and results["X_dumbbell"]["E_f"] is not None:
        print("\n  SIA relative formation energies (ΔE w.r.t. X dumbbell):")
        e_x = results["X_dumbbell"]["E_f"]
        for name in ["T_interstitial", "H_interstitial", "B_bond_center"]:
            if name in results and results[name]["E_f"] is not None:
                print(f"    ΔE({DISPLAY[name]} − X) = {results[name]['E_f'] - e_x:+.3f} eV")
        print("  Expected ordering: X < H ≈ T < B")

    print(f"\n  Results written to {out_txt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point defect formation energy test for MTP potential"
    )
    parser.add_argument("--pot",        required=True,
                        help="Path to potential (.almtp)")
    parser.add_argument("--a0",         type=float, default=5.779,
                        help="Lattice constant Å — use MTP equilibrium from E-V test (default: 5.779)")
    parser.add_argument("--fmax",       type=float, default=0.0001,
                        help="Force convergence threshold eV/Å (default: 0.0001)")
    parser.add_argument("--max-steps",  type=int,   default=500,
                        help="Max relaxation steps per defect (default: 500)")
    parser.add_argument("--no-relax",   action="store_true",
                        help="Skip relaxation — compute energy at as-built geometry")
    parser.add_argument("--outdir",     default="results/tests/defect_formation",
                        help="Output directory")
    parser.add_argument("--config",     default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML (provides mlp_binary)")
    parser.add_argument("--mlp",        default=None,
                        help="Override mlp binary path")
    args = parser.parse_args()

    mlp = args.mlp
    if mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        mlp=mlp,
        a0=args.a0,
        fmax=args.fmax,
        max_steps=args.max_steps,
        no_relax=args.no_relax,
    )


if __name__ == "__main__":
    main()
