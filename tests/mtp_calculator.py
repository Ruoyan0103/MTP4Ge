"""ASE Calculator wrapping mlp calculate_efs for NEB and geometry optimization."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import _mlp_env, write_bare_cfg


class MTPCalculator(Calculator):
    """ASE Calculator that calls `mlp calculate_efs` via subprocess.

    Spawns a subprocess per energy/force evaluation — intended for small-cell
    single-point calculations (NEB images, relaxation steps) where accuracy
    matters more than throughput.
    """

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
    import re
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
