"""ASE Calculators wrapping MTP evaluation for NEB and geometry optimization.

Two calculators, same role (one subprocess per energy/force evaluation —
intended for small-cell single-point calculations such as NEB images or
relaxation steps, where accuracy matters more than throughput):

  MTPCalculator       - wraps `mlp calculate_efs`.
  MTPLammpsCalculator - wraps LAMMPS (pair_style hybrid/overlay mtp nlh, via
                         config/lammps/phonon_dispersion_mtp.in). Needed for
                         potentials trained with a radial basis type that
                         `mlp calculate_efs` cannot load ("Wrong radial basis
                         type") — see src/physical_validation/elastic_constant/potential.mod
                         for the same workaround used for elastic constants.
"""

import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

sys.path.insert(0, str(Path(__file__).parent))
from utils import _mlp_env, write_bare_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAMMPS_TEMPLATE = REPO_ROOT / "config" / "lammps" / "phonon_dispersion_mtp.in"
GE_MASS = 72.630


class MTPCalculator(Calculator):
    """ASE Calculator that calls `mlp calculate_efs` via subprocess."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, mlp: str, pot: str, label: str = "mtp_calc", **kwargs):
        Calculator.__init__(self, label=label, **kwargs)
        self.mlp = mlp
        self.pot = pot

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        cell = np.array(atoms.get_cell())
        pos = atoms.get_positions()
        types = [0] * len(atoms)  # single-species Ge (type index 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Path(tmpdir) / "calc.cfg"
            write_bare_cfg([(cell, pos, types)], cfg)

            cmd = [self.mlp, "calculate_efs", self.pot, str(cfg)]
            result = subprocess.run(
                cmd, env=_mlp_env(), capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mlp calculate_efs failed:\n{result.stderr[-800:]}"
                )

            energy, forces = _parse_efs_cfg(cfg)

        self.results["energy"] = energy
        self.results["forces"] = forces


def _parse_efs_cfg(path: Path) -> tuple[float, np.ndarray]:
    """Parse energy and forces from the first CFG block in an EFS output file."""
    text = path.read_text()
    m = re.search(r"BEGIN_CFG(.*?)END_CFG", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"No CFG block in {path}")
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
            if len(parts) >= 8:  # id type x y z fx fy fz
                forces.append([float(parts[5]), float(parts[6]), float(parts[7])])

    return energy, np.array(forces)


class MTPLammpsCalculator(Calculator):
    """ASE Calculator that runs one LAMMPS single-point evaluation per call
    (pair_style hybrid/overlay mtp nlh). Same tradeoff as MTPCalculator.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, lammps_cmd: str, pot: str, label: str = "mtp_lammps_calc", **kwargs):
        Calculator.__init__(self, label=label, **kwargs)
        self.lammps_cmd = lammps_cmd
        self.pot = str(Path(pot).resolve())

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        from ase.io import write as ase_write

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            ase_write(str(workdir / "structure.lammps"), atoms,
                      format="lammps-data", atom_style="atomic")

            text = (
                LAMMPS_TEMPLATE.read_text()
                .replace("${STRUCT_FILE}", "structure.lammps")
                .replace("${POT_PATH}", self.pot)
                .replace("${MASS}", str(GE_MASS))
            )
            (workdir / "in.lammps").write_text(text)

            cmd = shlex.split(self.lammps_cmd) + [
                "-in", "in.lammps", "-log", "log.lammps", "-screen", "none",
            ]
            result = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"LAMMPS exited {result.returncode}:\n{result.stderr[-800:]}"
                )

            energy = float((workdir / "energy.txt").read_text().split()[0])
            forces = _parse_forces_from_dump(workdir / "force.dump", len(atoms))

        self.results["energy"] = energy
        self.results["forces"] = forces


def _parse_forces_from_dump(path: Path, n_atoms: int) -> np.ndarray:
    """Parse a `dump custom ... id type x y z fx fy fz` snapshot → (n_atoms, 3)."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS")) + 1
    forces = np.zeros((n_atoms, 3))
    for line in lines[start:start + n_atoms]:
        parts = line.split()
        atom_id = int(parts[0])
        forces[atom_id - 1] = [float(parts[5]), float(parts[6]), float(parts[7])]
    return forces
