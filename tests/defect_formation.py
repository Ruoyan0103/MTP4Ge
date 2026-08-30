"""
Point defect formation energy test for MTP potentials.

Computes formation energies of:
  V  – monovacancy (remove 1 atom, 215-atom cell)
  T  – tetrahedral interstitial  (add at void centre (a/2,a/2,a/2))
  H  – hexagonal interstitial    (add in ⟨111⟩ channel (5a/8,5a/8,5a/8))
  X  – ⟨110⟩ split dumbbell      (replace 1 atom with 2 displaced ±δ along [110])
  B  – bond-centre interstitial  (add at midpoint of nearest-neighbour bond)

All defects use the same conventional supercell (repeated a₀ = MTP equilibrium cell),
sized via --supercell / SUPERCELL (default 2×2×2, 64 ± 1 atoms).

Relaxation follows the same two-stage protocol used for the DFT reference
(see MTP4Ge/../00-subsets/05-defects/Defect_formation):
  1. Bulk reference: BOTH ions and cell relax together (ISIF=3 analog),
     giving the MTP's own equilibrium lattice + μ_Ge.
  2. Each defect is built ON TOP OF that relaxed bulk cell/positions (not
     rebuilt from the ideal input a₀), then relaxed with the cell HELD
     FIXED — only ions move (ISIF=2 analog). This keeps E_bulk and E_defect
     on the same cell/volume, which is what makes E_f comparable to a
     fixed-cell DFT formation energy.
Both stages use `mlp relax`; cell relaxation is controlled by the sign of
`stress_tolerance` in the relaxation settings file (mlip-3's own relax
driver disables cell relaxation whenever stress_tolerance <= 0 — see
src/drivers/relaxation.cpp: relax_cell_flag = (tol_stress > 0)).

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
from utils import _mlp_env, calc_efs, write_bare_cfg, _read_cfg_geometry

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from cfg_to_xyz import _parse_all_cfgs

DEFAULT_TRAIN_CONFIG = "config/training.yaml"

# Conventional-cell repeat count applied to every defect + bulk-reference structure.
# Override via --supercell nx ny nz.
SUPERCELL = (3, 3, 3)

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
# Structure builders  (all use the SUPERCELL conventional-cell repeat count)
# ---------------------------------------------------------------------------

def _sc(a0: float, repeat=SUPERCELL):
    """Return an ASE Atoms object: conventional cell repeated `repeat` times."""
    from ase.build import bulk
    return bulk("Ge", crystalstructure="diamond", a=a0, cubic=True).repeat(repeat)


def build_bulk_ref(a0: float, repeat=SUPERCELL):
    """Conventional supercell for μ_Ge reference."""
    atoms = _sc(a0, repeat)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_isolated_atom(box_size: float = 25.0):
    """Single Ge atom centred in a cubic box ≫ MTP cutoff (no periodic neighbours)."""
    cell = np.eye(3) * box_size
    pos = np.array([[box_size / 2, box_size / 2, box_size / 2]])
    return cell, pos, [0]


def _nearest_origin_idx(pos: np.ndarray, cell: np.ndarray) -> int:
    """Index of the atom nearest the origin, minimum-image convention."""
    frac = pos @ np.linalg.inv(cell)
    frac -= np.round(frac)
    cart = frac @ cell
    return int(np.argmin(np.linalg.norm(cart, axis=1)))


# ---------------------------------------------------------------------------
# Defect builders — operate on an ALREADY-RELAXED bulk (cell, pos, types)
# rather than rebuilding an idealized a0 lattice from scratch. This mirrors
# the DFT-side protocol: defects are cut from the ISIF=3-relaxed pristine
# CONTCAR, not from the ideal unrelaxed lattice, so the defect cell always
# matches the bulk reference's own true equilibrium cell exactly.
# ---------------------------------------------------------------------------

def build_vacancy_from(cell, pos, types, repeat=None, **kwargs):
    """Remove the atom nearest the origin."""
    idx = _nearest_origin_idx(pos, cell)
    new_pos = np.delete(pos, idx, axis=0)
    new_types = types[:idx] + types[idx + 1:]
    return cell.copy(), new_pos, new_types


def build_T_interstitial_from(cell, pos, types, repeat=SUPERCELL, **kwargs):
    """
    Tetrahedral void: (1/2,1/2,1/2) fractional of one conventional cell,
    surrounded by 4 B-sublattice atoms at distance a√3/4 (= bond length).
    Placed in the middle conventional-cell block (repeat // 2) using the
    actual relaxed per-axis conventional-cell length, not the ideal a0.
    """
    a0_eff = np.diag(cell) / np.array(repeat)
    block = np.array(repeat) // 2
    t_cart = (block + 0.5) * a0_eff
    new_pos = np.vstack([pos, t_cart])
    return cell.copy(), new_pos, types + [0]


def build_H_interstitial_from(cell, pos, types, repeat=SUPERCELL, **kwargs):
    """
    Hexagonal site: (0.625, 0.625, 0.625) fractional of one conventional
    cell, surrounded by 6 atoms in a hexagonal ring. Same middle-block /
    actual-a0_eff placement as the T site.
    """
    a0_eff = np.diag(cell) / np.array(repeat)
    block = np.array(repeat) // 2
    h_cart = (block + 0.625) * a0_eff
    new_pos = np.vstack([pos, h_cart])
    return cell.copy(), new_pos, types + [0]


def build_X_dumbbell_from(cell, pos, types, half_disp: float = 1.2, repeat=None, **kwargs):
    """
    ⟨110⟩ split dumbbell: remove atom nearest the origin, add two Ge
    displaced ±half_disp Å along [1,1,0]/√2 from its original site.
    Default half_disp ≈ bond_length/2 ≈ 1.2 Å.
    """
    idx = _nearest_origin_idx(pos, cell)
    center = pos[idx].copy()
    new_pos = np.delete(pos, idx, axis=0)
    new_types = types[:idx] + types[idx + 1:]
    disp = np.array([1.0, 1.0, 0.0]) / np.sqrt(2) * half_disp
    new_pos = np.vstack([new_pos, center + disp, center - disp])
    return cell.copy(), new_pos, new_types + [0, 0]


def build_B_bond_center_from(cell, pos, types, repeat=None, **kwargs):
    """
    Bond-centre site: midpoint of the bond between the atom nearest the
    origin and its true nearest neighbour (found directly in the relaxed
    structure, minimum-image convention), not the ideal a0/8 formula.
    """
    idx = _nearest_origin_idx(pos, cell)
    diff = pos - pos[idx]
    frac_diff = diff @ np.linalg.inv(cell)
    frac_diff -= np.round(frac_diff)
    cart_diff = frac_diff @ cell
    dist = np.linalg.norm(cart_diff, axis=1)
    dist[idx] = np.inf
    nbr = int(np.argmin(dist))
    b_cart = pos[idx] + 0.5 * cart_diff[nbr]
    new_pos = np.vstack([pos, b_cart])
    return cell.copy(), new_pos, types + [0]


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


def cfg_to_xyz(cfg_path: Path, xyz_path: Path | None = None) -> Path:
    """Convert a CFG file to extended XYZ for viewing in OVITO/VESTA."""
    from ase.io import write as ase_write
    if xyz_path is None:
        xyz_path = cfg_path.with_suffix(".xyz")
    atoms_list = _parse_all_cfgs(cfg_path)
    ase_write(str(xyz_path), atoms_list if len(atoms_list) > 1 else atoms_list[0],
              format="extxyz")
    return xyz_path


def cfg_to_poscar(cfg_path: Path, poscar_path: Path | None = None) -> Path:
    """Convert a (single-frame) CFG file to a VASP POSCAR for viewing in OVITO/VESTA."""
    from ase.io import write as ase_write
    if poscar_path is None:
        poscar_path = cfg_path.with_suffix(".POSCAR")
    atoms_list = _parse_all_cfgs(cfg_path)
    ase_write(str(poscar_path), atoms_list[0], format="vasp")
    return poscar_path


# ---------------------------------------------------------------------------
# Relaxation via mlp relax (BFGS) — fixed cell or cell+ions, per fix_cell
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
    fix_cell: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Relax a configuration with mlp relax (BFGS).

    fix_cell=True  -> ISIF=2 analog: only ionic positions relax, lattice
                       vectors held fixed. Implemented via
                       stress_tolerance <= 0, which mlip-3's own relax
                       driver treats as "disable cell relaxation"
                       (src/drivers/relaxation.cpp: relax_cell_flag =
                       (tol_stress > 0)).
    fix_cell=False -> ISIF=3 analog: ions AND cell relax together.

    Returns (final energy [eV], relaxed cell [3x3 Å], relaxed positions [Nx3 Å]).
    """
    workdir.mkdir(parents=True, exist_ok=True)

    stress_tolerance = 0.0 if fix_cell else 0.001
    ini = workdir / "relax.ini"
    log = workdir / "relaxation.log"
    ini.write_text(
        f"relax:force_tolerance      {fmax}\n"
        f"relax:stress_tolerance     {stress_tolerance}\n"
        f"relax:iteration_limit      {max_steps}\n"
        f"relax:max_step             0.05\n"
        f"relax:min_step             1.0e-8\n"
        f"relax:mindist              0.5\n"
        f"relax:use_gd_method        FALSE\n"
        f"relax:freeze_small_grads   FALSE\n"
        f"relax:correct_cell         FALSE\n"
        f"relax:pressure             0.0\n"
        f"relax:log                  {log}\n"
        f"relax:init_mindist         1.0\n"
        f"relax:bfgs_wolfe_c1        0.1\n"
        f"relax:bfgs_wolfe_c2        0.8\n"
        f"relax:forces_elasticity_scale    10\n"
        f"relax:stress_elasticity_scale    50\n"
    )

    in_cfg  = workdir / "in.cfg"
    out_cfg = workdir / "relaxed.cfg"
    write_bare_cfg([(cell, pos, types)], in_cfg)
    cfg_to_xyz(in_cfg, workdir / "unrelaxed.xyz")
    cfg_to_poscar(in_cfg, workdir / "unrelaxed.POSCAR")

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

    cfg_to_xyz(out_cfg)
    cfg_to_poscar(out_cfg)

    energy, forces = _parse_efs(out_cfg)
    rlx_cell, rlx_pos, _ = _read_cfg_geometry(out_cfg)
    max_f = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    status = "Converged" if max_f < fmax else "Max steps reached"
    mode = "fixed cell" if fix_cell else "cell + ions"
    print(f"    [{mode}] {status} | E = {energy:.6f} eV | Fmax = {max_f:.4f} eV/Å")
    return energy, rlx_cell, rlx_pos


def compute_cohesive_energy(mlp: str, pot: str, mu: float, workdir: Path,
                             box_size: float = 25.0) -> float:
    """
    Cohesive energy per atom: E_coh = E_atom(isolated) − μ_Ge.

    μ_Ge (bulk energy per atom) is passed in rather than recomputed, since
    run() already relaxes the bulk reference cell for the defect μ_Ge term.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    cell, pos, types = build_isolated_atom(box_size)

    in_cfg  = workdir / "isolated_atom.cfg"
    out_cfg = workdir / "isolated_atom_efs.cfg"
    write_bare_cfg([(cell, pos, types)], in_cfg)
    calc_efs(mlp, pot, in_cfg, out_cfg)
    e_atom, _ = _parse_efs(out_cfg)

    e_coh = e_atom - mu
    print(f"    E_atom(isolated) = {e_atom:.6f} eV | μ_Ge = {mu:.6f} eV/atom")
    print(f"    E_cohesive = {e_coh:.6f} eV/atom")
    return e_coh


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

DEFECTS = {
    "vacancy":        (build_vacancy_from,        -1),
    "T_interstitial": (build_T_interstitial_from, +1),
    "H_interstitial": (build_H_interstitial_from, +1),
    "X_dumbbell":     (build_X_dumbbell_from,      +1),
    "B_bond_center":  (build_B_bond_center_from,   +1),
}

DISPLAY = {
    "vacancy":        "V  (vacancy)",
    "T_interstitial": "T  (tetrahedral)",
    "H_interstitial": "H  (hexagonal)",
    "X_dumbbell":     "X  (⟨110⟩ dumbbell)",
    "B_bond_center":  "B  (bond-centre)",
}


def run(pot: str, outdir: Path, mlp: str, a0: float,
        fmax: float, max_steps: int, no_relax: bool,
        supercell: tuple[int, int, int] = SUPERCELL) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Bulk reference: ISIF=3 analog — ions + cell relax together ──
    print("\n  [Bulk reference: cell + ion relaxation]")
    b_cell, b_pos, b_types = build_bulk_ref(a0, repeat=supercell)
    b_out = outdir / "bulk"
    b_out.mkdir(parents=True, exist_ok=True)
    if no_relax:
        write_bare_cfg([(b_cell, b_pos, b_types)], b_out / "bulk.cfg")
        calc_efs(mlp, pot, b_out / "bulk.cfg", b_out / "bulk_efs.cfg")
        e_bulk, _ = _parse_efs(b_out / "bulk_efs.cfg")
    else:
        e_bulk, b_cell, b_pos = relax(mlp, pot, b_cell, b_pos, b_types, b_out,
                                       fmax=fmax, max_steps=max_steps, fix_cell=False)
    n_bulk = len(b_pos)
    mu = e_bulk / n_bulk
    print(f"  μ_Ge = {mu:.6f} eV/atom  |  relaxed cell diag = "
          f"{tuple(round(x, 4) for x in np.diag(b_cell))} Å")

    # ── Cohesive energy ──
    print("\n  [Cohesive energy]")
    e_coh = compute_cohesive_energy(mlp, pot, mu, outdir / "cohesive")

    # ── Defects: ISIF=2 analog — cut from the relaxed bulk cell, fixed-cell relax ──
    results = {}
    for name, (builder, _) in DEFECTS.items():
        print(f"\n  [{DISPLAY[name]}]")
        d_cell, d_pos, d_types = builder(b_cell, b_pos, b_types, repeat=supercell)
        n_def = len(d_pos)
        print(f"    {n_def} atoms | cell {tuple(round(x,3) for x in np.diag(d_cell))} Å")
        d_out = outdir / name
        d_out.mkdir(parents=True, exist_ok=True)
        try:
            if no_relax:
                write_bare_cfg([(d_cell, d_pos, d_types)], d_out / "defect.cfg")
                calc_efs(mlp, pot, d_out / "defect.cfg", d_out / "defect_efs.cfg")
                e_def, _ = _parse_efs(d_out / "defect_efs.cfg")
            else:
                e_def, _, _ = relax(mlp, pot, d_cell, d_pos, d_types, d_out,
                                     fmax=fmax, max_steps=max_steps, fix_cell=True)
            e_f = e_def - n_def * mu
            results[name] = {"n": n_def, "E_total": e_def, "E_f": e_f}
            dft_str = f"  (DFT ref: {DFT_REF[name]:.2f} eV)" if name in DFT_REF else ""
            print(f"    E_f = {e_f:.3f} eV{dft_str}")
        except RuntimeError as exc:
            print(f"    FAILED: {exc}")
            results[name] = {"n": n_def, "E_total": None, "E_f": None}

    # ── Write output file ──
    out_txt = outdir / "defect_formation_energies.txt"
    relax_note = "no relaxation" if no_relax else (
        f"bulk: BFGS cell+ions (ISIF=3 analog); "
        f"defects: BFGS fixed-cell ions-only (ISIF=2 analog), fmax={fmax} eV/Å"
    )
    with open(out_txt, "w") as f:
        f.write(f"Point defect formation energies — MTP: {pot}\n")
        f.write(f"Lattice constant a₀ = {a0:.4f} Å | Relaxation: {relax_note}\n")
        f.write(f"μ_Ge = {mu:.6f} eV/atom  (from {n_bulk}-atom bulk cell)\n")
        f.write(f"E_cohesive = {e_coh:.6f} eV/atom\n\n")

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
        print("  Expected ordering: X < T < H < B")
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
    parser.add_argument("--a0",         type=float, default=5.76,
                        help="Lattice constant Å — use MTP equilibrium from E-V test (default: 5.76)")
    parser.add_argument("--fmax",       type=float, default=0.0001,
                        help="Force convergence threshold eV/Å (default: 0.0001)")
    parser.add_argument("--max-steps",  type=int,   default=500,
                        help="Max relaxation steps per defect (default: 500)")
    parser.add_argument("--no-relax",   action="store_true",
                        help="Skip relaxation — compute energy at as-built geometry")
    parser.add_argument("--supercell",  type=int, nargs=3, default=list(SUPERCELL),
                        metavar=("NX", "NY", "NZ"),
                        help=f"Conventional-cell repeat count (default: {list(SUPERCELL)})")
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
        supercell=tuple(args.supercell),
    )


if __name__ == "__main__":
    main()
