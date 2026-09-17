"""Smoke test for MTPCalculator — energy is a float, forces ≈ 0 on perfect diamond."""

import os
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk

sys.path.insert(0, str(Path(__file__).parent))
from mtp_calculator import MTPCalculator
from utils import MLP_DEFAULT as MLP

POT = os.environ.get("MTP_POT", "/scratch/project_2012355/Paper_3/MTP4Ge/results/potentials/pot.almtp")


def test_mtp_calculator_energy_forces():
    atoms = bulk("Ge", "diamond", a=5.658, cubic=True)
    calc = MTPCalculator(mlp=MLP, pot=POT)
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert isinstance(e, float), f"Energy is not a float: {type(e)}"
    assert f.shape == (8, 3), f"Forces shape unexpected: {f.shape}"
    assert np.allclose(f, 0.0, atol=0.05), f"Forces not near zero on perfect diamond:\n{f}"
    print(f"  Energy = {e:.6f} eV  ({e/8:.6f} eV/atom)")
    print(f"  Max |F| = {np.abs(f).max():.4e} eV/Å  [pass]")


if __name__ == "__main__":
    test_mtp_calculator_energy_forces()
    print("All tests passed.")
