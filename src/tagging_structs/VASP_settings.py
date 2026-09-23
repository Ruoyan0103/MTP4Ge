"""Shared VASP single-point calculator settings.

Single source of truth for VASP DFT physics parameters, used by both the
structure-generation scripts and the active-learning DFT tagging step
(src/active_learning_structs/active_learning.py, tagging_engine: vasp).
Cluster/job settings (command, ncore, kpar, pp path/version, output
directory) vary per caller and per cluster, so they stay overridable via
keyword arguments rather than being baked in here.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from ase.calculators.vasp import Vasp
from ase.io import read


def set_cal(**overrides):
    """Static single-point VASP calculator shared by all VASP tagging callers.

    Physics settings (ENCUT, EDIFF, smearing, functional, precision, k-point
    spacing, ...) are fixed here so every VASP tagging run uses identical DFT
    parameters. Pass keyword arguments to override or add settings (e.g.
    directory=..., ncore=..., kpar=..., pp_version=...).
    """
    kwargs = dict(
        command='srun vasp_std',
        setups={'Ge': '_d'},

        # Initialisation:
        istart=0,          # (Default = 0; from scratch)

        # Ionic:
        ibrion=-1,         # (Static calculation, default for nsw=0)
        nsw=0,             # (Max ionic steps)
        isif=2,            # (Stress/relaxation flag; forces only)

        # Electronic:
        ediff=1E-07,       # (SCF energy convergence; eV)
        ismear=0,          # (Electronic temperature, Gaussian smearing)
        sigma=0.05,        # (Smearing value in eV)
        nelm=100,          # (Max SCF steps)
        gga='PE',          # (Exchange-correlation functional)

        # Plane wave basis set:
        encut=500,         # (Default = largest ENMAX in the POTCAR file)
        prec='Accurate',   # (Accurate forces are required)
        lasph=True,        # (Non-spherical elements included)

        # Reciprocal space
        kspacing=0.15,     # (Smallest spacing between k points in 1/A)
        kgamma=True,       # (Default = Gamma centered)

        # Parallelisation:
        ncore=4,
        kpar=1,

        # Output features:
        lwave=False,       # (WAVECAR is NOT written out)
        lcharg=False,      # (CHGCAR is NOT written out)
    )
    kwargs.update(overrides)
    return Vasp(**kwargs)


def default_potcar_path(pp_path=None, pp_version="64"):
    """POTCAR location implied by set_cal: gga='PE' -> potpaw_PBE, setups
    {'Ge': '_d'} -> Ge_d, i.e. $VASP_PP_PATH/potpaw_PBE.<pp_version>/Ge_d/POTCAR
    (no version suffix if pp_version is empty)."""
    pp_path = pp_path or os.environ.get("VASP_PP_PATH", "")
    pp_dir = f"potpaw_PBE.{pp_version}" if pp_version else "potpaw_PBE"
    return Path(pp_path) / pp_dir / "Ge_d" / "POTCAR"


def write_vasp_dirs(cfg_path, outdir, potcar=None, **overrides):
    """Build one VASP single-point folder per structure in a CFG file.

    Each CFG block becomes <outdir>/struct_N/ (N = 1, 2, ...) containing
    POSCAR (via convert_format.cfg_to_poscars), INCAR (from set_cal, with
    **overrides applied) and a copy of *potcar* (default:
    default_potcar_path()). No KPOINTS is written: set_cal uses KSPACING.
    Returns the list of created folders.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    from convert_format import cfg_to_poscars

    potcar = Path(potcar) if potcar else default_potcar_path()
    if not potcar.is_file():
        raise FileNotFoundError(f"POTCAR not found: {potcar}")

    calc = set_cal(**overrides)
    struct_dirs = cfg_to_poscars(Path(cfg_path), Path(outdir))
    for struct_dir in struct_dirs:
        atoms = read(str(struct_dir / "POSCAR"), format="vasp")
        # Attributes write_incar reads that calc.initialize() would set; set
        # them directly since initialize() also looks up POTCARs under
        # $VASP_PP_PATH, which an explicit --potcar is meant to bypass.
        calc.spinpol = atoms.get_initial_magnetic_moments().any()
        calc.sort = list(range(len(atoms)))
        calc.write_incar(atoms, directory=str(struct_dir))
        shutil.copy(potcar, struct_dir / "POTCAR")
    print(f"Wrote {len(struct_dirs)} VASP folders → {outdir}")
    return struct_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write struct_N/{POSCAR,INCAR,POTCAR} folders from a CFG file.")
    parser.add_argument("--input", required=True, help="CFG file, e.g. iter_1/new_added.cfg")
    parser.add_argument("--outdir", required=True, help="Folder to hold struct_* subfolders")
    parser.add_argument("--potcar", default="config/DFT/POTCAR",
                        help="Ge_d PBE POTCAR to copy (default: config/DFT/POTCAR)")
    parser.add_argument("--kpar", type=int, default=None, help="Override set_cal's KPAR")
    parser.add_argument("--ncore", type=int, default=None, help="Override set_cal's NCORE")
    args = parser.parse_args()
    overrides = {k: v for k, v in (("kpar", args.kpar), ("ncore", args.ncore)) if v is not None}
    write_vasp_dirs(args.input, args.outdir, potcar=args.potcar, **overrides)
