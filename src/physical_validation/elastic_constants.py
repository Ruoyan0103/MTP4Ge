"""
Elastic constants quality test for MTP potentials.

Computes the full 6×6 stiffness tensor via finite differences of the stress
tensor (central differences, ±strain per Voigt component).  For a cubic
crystal (Ge diamond) this yields C11, C12, and C44 and derived moduli.

Voigt convention used throughout:
  indices 0–5  →  xx yy zz yz xz xy
  shear strains are engineering shear (γ = 2ε_tensor)

Reference values for Ge (experimental):
  C11 ≈ 126 GPa,  C12 ≈ 48 GPa,  C44 ≈ 67 GPa,  B ≈ 74 GPa

Usage:
    python src/physical_validation/elastic_constants.py --pot results/potentials/pot.almtp
    python src/physical_validation/elastic_constants.py --pot results/potentials/pot.almtp \\
        --struct data/relaxed.cfg --strain 0.01
"""

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import calc_efs, load_structure, write_bare_cfg

DEFAULT_TRAIN_CONFIG = "config/training.yaml"
DEFAULT_AL_CONFIG = "config/active_learning.yaml"
EV_A3_TO_GPA = 160.21766   # 1 eV/Å³ = 160.21766 GPa

REF_DIR = Path(__file__).parent / "reference"
LAMMPS_ELASTIC_DIR = Path(__file__).parent / "elastic_constant"


def _load_elastic_ref(filename: str) -> dict[str, float] | None:
    """Load reference elastic constants (C11 C12 C44 B, in GPa) from a file."""
    path = REF_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    return {"C11": float(parts[0]), "C12": float(parts[1]),
                            "C44": float(parts[2]), "B": float(parts[3])}
    return None


EXP_ELASTIC = _load_elastic_ref("elastic_constants/elastic_constants_EXP.dat") or \
    {"C11": 126, "C12": 48, "C44": 67, "B": 74}
DFT_ELASTIC = _load_elastic_ref("elastic_constants/elastic_constants_DFT.dat") or \
    {"C11": 105, "C12": 37, "C44": 54, "B": 59}

VOIGT_NAMES = ["xx", "yy", "zz", "yz", "xz", "xy"]


# ---------------------------------------------------------------------------
# Strain application
# ---------------------------------------------------------------------------

def _apply_voigt_strain(cell: np.ndarray, voigt_idx: int, strain_mag: float) -> np.ndarray:
    """
    Apply a Voigt strain component to cell (row-vector convention).

    For normal strains (0–2): strain_mag = ε_ii (direct).
    For shear strains (3–5): strain_mag = γ (engineering shear = 2×tensor shear).
    The deformation gradient is symmetric: F = I + ε_tensor.
    """
    eps = np.zeros((3, 3))
    if voigt_idx == 0:
        eps[0, 0] = strain_mag
    elif voigt_idx == 1:
        eps[1, 1] = strain_mag
    elif voigt_idx == 2:
        eps[2, 2] = strain_mag
    elif voigt_idx == 3:          # yz: tensor ε_yz = γ/2
        eps[1, 2] = eps[2, 1] = strain_mag / 2
    elif voigt_idx == 4:          # xz
        eps[0, 2] = eps[2, 0] = strain_mag / 2
    elif voigt_idx == 5:          # xy
        eps[0, 1] = eps[1, 0] = strain_mag / 2
    F = np.eye(3) + eps
    return (F @ cell.T).T          # new_cell rows = F @ old_cell rows


def _scale_positions(pos: np.ndarray, cell_old: np.ndarray, cell_new: np.ndarray) -> np.ndarray:
    """Map Cartesian positions from old cell to new cell via fractional coordinates."""
    frac = pos @ np.linalg.inv(cell_old)
    return frac @ cell_new


# ---------------------------------------------------------------------------
# CFG stress parsing
# ---------------------------------------------------------------------------

def _parse_stresses(path: Path) -> list[np.ndarray]:
    """
    Parse PlusStress blocks from an EFS CFG file.
    Returns a list of arrays [xx, yy, zz, yz, xz, xy] in eV.
    PlusStress = σ × V_cell, so stress = PlusStress / V.
    """
    stresses = []
    for block in re.split(r"(?=BEGIN_CFG\b)", path.read_text()):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        m = re.search(
            r"PlusStress:.*?\n\s*"
            r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+"
            r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)",
            block,
        )
        if m:
            stresses.append(np.array([float(m.group(i)) for i in range(1, 7)]))
    return stresses


# ---------------------------------------------------------------------------
# Elastic constants via finite differences
# ---------------------------------------------------------------------------

def compute_elastic_constants(
    mlp: str,
    pot: str,
    cell0: np.ndarray,
    pos0: np.ndarray,
    types: list,
    outdir: Path,
    strain_mag: float = 0.01,
) -> dict:
    """
    Finite-difference computation of the 6×6 stiffness tensor.

    For each Voigt index j, apply ±strain_mag and use central differences:
      C[i, j] = (σ_i(+ε) − σ_i(−ε)) / (2 ε)
    where σ_i = PlusStress_i / V₀  [eV/Å³].

    For cubic crystals averages C11, C12, C44 are reported.
    """
    V0 = abs(np.linalg.det(cell0))

    configs, labels = [], []
    for vi in range(6):
        for sign in (+1, -1):
            s = sign * strain_mag
            new_cell = _apply_voigt_strain(cell0, vi, s)
            new_pos  = _scale_positions(pos0, cell0, new_cell)
            configs.append((new_cell, new_pos, types))
            labels.append((vi, sign))

    in_cfg  = outdir / "strained.cfg"
    out_cfg = outdir / "strained_efs.cfg"
    write_bare_cfg(configs, in_cfg)

    rc = calc_efs(mlp, pot, in_cfg, out_cfg)
    if rc != 0:
        print(f"  Warning: mlp calculate_efs returned {rc}")
    if not out_cfg.exists():
        raise RuntimeError("mlp calculate_efs produced no output file")

    stresses = _parse_stresses(out_cfg)
    if len(stresses) != len(configs):
        raise RuntimeError(
            f"Expected {len(configs)} stress blocks, got {len(stresses)}"
        )

    # Build 6×6 stiffness matrix in eV/Å³, then convert to GPa
    # MLIP PlusStress = −V × σ_Cauchy  →  σ_Cauchy = −PlusStress / V
    # C[i,j] = dσ_i/dε_j = −(PS(+ε) − PS(−ε)) / (2 ε V0)
    C = np.zeros((6, 6))
    for j in range(6):
        pos_stress = stresses[j * 2]       # sign = +1
        neg_stress = stresses[j * 2 + 1]  # sign = −1
        C[:, j] = -(pos_stress - neg_stress) / (2 * strain_mag * V0)

    # Symmetrize to remove numerical noise
    C = 0.5 * (C + C.T)
    C_GPa = C * EV_A3_TO_GPA

    return _moduli_from_matrix(C_GPa)


def _moduli_from_matrix(C_GPa: np.ndarray) -> dict:
    """Derive cubic-average elastic constants and moduli from a full 6×6 matrix [GPa]."""
    # Cubic averages (average symmetry-equivalent components)
    C11 = float(np.mean([C_GPa[0, 0], C_GPa[1, 1], C_GPa[2, 2]]))
    C12 = float(np.mean([C_GPa[0, 1], C_GPa[0, 2], C_GPa[1, 2]]))
    C44 = float(np.mean([C_GPa[3, 3], C_GPa[4, 4], C_GPa[5, 5]]))

    # Derived moduli (Voigt averages, valid for polycrystalline / cubic)
    B  = (C11 + 2 * C12) / 3
    G  = (C11 - C12 + 3 * C44) / 5
    E  = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))

    return dict(
        C11=C11, C22=float(C_GPa[1,1]), C33=float(C_GPa[2,2]),
        C12=C12, C13=float(C_GPa[0,2]), C23=float(C_GPa[1,2]),
        C44=C44, C55=float(C_GPa[4,4]), C66=float(C_GPa[5,5]),
        B=B, G=G, E=E, nu=nu,
        C_GPa=C_GPa,
    )


# ---------------------------------------------------------------------------
# Elastic constants via LAMMPS (relaxed-ion: atoms minimize at fixed strain)
# ---------------------------------------------------------------------------

def compute_elastic_constants_lammps(lammps_cmd: str, pot: str, outdir: Path) -> dict:
    """
    Run the LAMMPS ELASTIC example (src/physical_validation/elastic_constant/) to compute the
    full 6×6 stiffness tensor. Unlike compute_elastic_constants, this relaxes
    atomic positions (min_style cg) at each fixed strain before reading the
    stress, so it reports the relaxed-ion (not clamped-ion) elastic response.

    Only potential.mod is templated (${POT_PATH}, ${GRADE_THRESHOLD},
    ${GRADE_BREAK}); init.mod, displace.mod, in.elastic are used as-is.
    """
    run_dir = outdir / "lammps"
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in ("init.mod", "displace.mod", "in.elastic"):
        shutil.copy(LAMMPS_ELASTIC_DIR / name, run_dir / name)

    potential_text = (
        (LAMMPS_ELASTIC_DIR / "potential.mod").read_text()
        .replace("${POT_PATH}", str(Path(pot).resolve()))
    )
    (run_dir / "potential.mod").write_text(potential_text)

    cmd = shlex.split(lammps_cmd) + ["-in", "in.elastic", "-log", "log.lammps", "-screen", "none"]
    print("  LAMMPS:", " ".join(cmd), f"(cwd={run_dir})")
    result = subprocess.run(cmd, cwd=str(run_dir))
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS exited {result.returncode}; see {run_dir / 'log.lammps'}")

    log_text = (run_dir / "log.lammps").read_text()
    C_GPa = np.zeros((6, 6))
    for p, q in [(1,1),(2,2),(3,3),(1,2),(1,3),(2,3),(4,4),(5,5),(6,6),
                 (1,4),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)]:
        m = re.search(rf"Elastic Constant C{p}{q}all\s*=\s*([-\d.eE+]+)", log_text)
        if not m:
            raise RuntimeError(f"Could not find C{p}{q}all in {run_dir / 'log.lammps'}")
        val = float(m.group(1))
        C_GPa[p - 1, q - 1] = val
        C_GPa[q - 1, p - 1] = val

    return _moduli_from_matrix(C_GPa)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    struct: str | None,
    outdir: Path,
    strain_mag: float,
    mlp: str,
    lattice_const: float = 5.76,
    method: str = "mtp",
    lammps_cmd: str | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if method == "lammps":
        print(f"  Method: LAMMPS relaxed-ion elastic constants (src/physical_validation/elastic_constant/)")
        res = compute_elastic_constants_lammps(lammps_cmd, pot, outdir)
        method_line = "LAMMPS (relaxed-ion, src/physical_validation/elastic_constant/in.elastic)"
    else:
        if struct is None:
            # Use 8-atom conventional cubic cell so Cartesian x/y/z align with [100]/[010]/[001]
            from ase.build import bulk
            atoms = bulk("Ge", crystalstructure="diamond", a=lattice_const, cubic=True)
            cell0 = atoms.get_cell().array.copy()
            pos0  = atoms.get_positions()
            types = [0] * len(atoms)
        else:
            cell0, pos0, types = load_structure(struct)

        V0 = abs(np.linalg.det(cell0))
        n_atoms = len(pos0)
        a_eff = V0 ** (1.0 / 3.0) if struct is None else (V0 / n_atoms * 8) ** (1.0 / 3.0)
        print(f"  Structure: {n_atoms} atoms  |  a = {a_eff:.4f} Å  |  V₀ = {V0:.4f} Å³  |  strain = {strain_mag}")

        res = compute_elastic_constants(mlp, pot, cell0, pos0, types, outdir, strain_mag)
        method_line = f"MTP calculate_efs (clamped-ion, finite-difference strain magnitude: {strain_mag})"

    out_txt = outdir / "elastic_constants.txt"
    with open(out_txt, "w") as f:
        f.write(f"Elastic constants — MTP potential: {pot}\n")
        f.write(f"Method: {method_line}\n\n")

        f.write("--- Cubic averages (symmetry-equivalent components averaged) ---\n")
        f.write(f"C11  = {res['C11']:8.2f} GPa    (exp. Ge: ~{EXP_ELASTIC['C11']:.0f} GPa) (DFT. Ge: ~{DFT_ELASTIC['C11']:.0f} GPa)\n")
        f.write(f"C12  = {res['C12']:8.2f} GPa    (exp. Ge:  ~{EXP_ELASTIC['C12']:.0f} GPa) (DFT. Ge: ~{DFT_ELASTIC['C12']:.0f} GPa)\n")
        f.write(f"C44  = {res['C44']:8.2f} GPa    (exp. Ge:  ~{EXP_ELASTIC['C44']:.0f} GPa) (DFT. Ge: ~{DFT_ELASTIC['C44']:.0f} GPa)\n\n")

        f.write("--- Individual components (verify cubic symmetry) ---\n")
        f.write(
            f"C11, C22, C33 = {res['C11']:.2f}, {res['C22']:.2f}, {res['C33']:.2f} GPa\n"
        )
        f.write(
            f"C12, C13, C23 = {res['C12']:.2f}, {res['C13']:.2f}, {res['C23']:.2f} GPa\n"
        )
        f.write(
            f"C44, C55, C66 = {res['C44']:.2f}, {res['C55']:.2f}, {res['C66']:.2f} GPa\n\n"
        )

        f.write("--- Derived moduli (Voigt averages) ---\n")
        f.write(f"Bulk modulus     B  = {res['B']:.2f} GPa (exp. Ge: ~{EXP_ELASTIC['B']:.0f} GPa) (DFT. Ge: ~{DFT_ELASTIC['B']:.0f} GPa) \n")
        f.write(f"Shear modulus    G  = {res['G']:.2f} GPa (exp. Ge: ~54 GPa) (DFT. Ge: ~45 GPa) \n")
        f.write(f"Young's modulus  E  = {res['E']:.2f} GPa (exp. Ge: ~131 GPa) (DFT. Ge: ~108 GPa) \n")
        f.write(f"Poisson ratio    ν  = {res['nu']:.4f}\n\n")

        f.write("--- Full 6×6 stiffness matrix C [GPa] ---\n")
        f.write("     " + "  ".join(f"{'ε_'+n:>8s}" for n in VOIGT_NAMES) + "\n")
        for i, row in enumerate(res["C_GPa"]):
            f.write(
                f"σ_{VOIGT_NAMES[i]:2s} "
                + "  ".join(f"{x:8.2f}" for x in row)
                + "\n"
            )

    print(f"\n  Elastic constants (cubic averages):")
    print(f"    C11 = {res['C11']:.2f} GPa")
    print(f"    C12 = {res['C12']:.2f} GPa")
    print(f"    C44 = {res['C44']:.2f} GPa")
    print(f"    B   = {res['B']:.2f} GPa  |  G = {res['G']:.2f} GPa  |  E = {res['E']:.2f} GPa")
    print(f"  Results written to {out_txt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elastic constants test for MTP potential"
    )
    parser.add_argument("--pot",    required=True,
                        help="Path to potential (.almtp)")
    parser.add_argument("--struct", default=None,
                        help="Reference structure (CFG / XYZ / LAMMPS data). "
                             "Default: 8-atom cubic ASE diamond Ge (a=5.76 Å)")
    parser.add_argument("--strain", type=float, default=0.01,
                        help="Engineering strain magnitude (default: 0.01)")
    parser.add_argument("--lattice-constant", type=float, default=5.76,
                        help="Lattice constant in Å for default Ge diamond structure "
                             "(default: 5.76 exp.; use MTP equilibrium a0 from E-V test)")
    parser.add_argument("--outdir", default="results/tests/elastic_constants",
                        help="Output directory")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML config (provides mlp_binary)")
    parser.add_argument("--mlp",    default=None,
                        help="Override mlp binary path")
    parser.add_argument("--method", choices=["mtp", "lammps"], default="lammps",
                        help="'mtp' (default): clamped-ion, mlp calculate_efs finite differences. "
                             "'lammps': relaxed-ion, runs src/physical_validation/elastic_constant/in.elastic "
                             "(atoms minimized at each fixed strain).")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG,
                        help="Active-learning YAML config (provides lammps_binary; --method lammps only)")
    parser.add_argument("--lammps", default=None,
                        help="Override LAMMPS command (--method lammps only), e.g. 'srun /path/to/lmp_mpi'")
    args = parser.parse_args()

    mlp = args.mlp
    if mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    lammps_cmd = args.lammps
    if args.method == "lammps" and lammps_cmd is None:
        with open(args.al_config) as f:
            lammps_cmd = yaml.safe_load(f)["lammps_binary"]

    run(
        pot=args.pot,
        struct=args.struct,
        outdir=Path(args.outdir),
        strain_mag=args.strain,
        mlp=mlp,
        lattice_const=args.lattice_constant,
        method=args.method,
        lammps_cmd=lammps_cmd,
    )


if __name__ == "__main__":
    main()
