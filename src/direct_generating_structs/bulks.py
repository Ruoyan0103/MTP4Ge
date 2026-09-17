"""
Generate distorted bulk Ge structures (8 phases: diamond, fcc, bcc, hcp, hd,
betaTin, bc8, st12) and run VASP single-point calculations on each, writing
extended XYZ databases suitable for MTP training data.

Combined from https://github.com/Ruoyan0103/ilearn/tree/main/dft-files/structures/bulk
(diamond.py, fcc.py, bcc.py, hcp.py, hd.py, betaTin.py, bc8.py, st12.py).
"""

from pathlib import Path
from ase import Atoms
from ase.io import write, read
import numpy as np
from ase.build import bulk
import os, shutil
import sys
import argparse
import copy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vasp_settings import set_cal


def _run_and_write(Ge_bulk, n, outfile_dir, config_type, bulk_list):
    """Single-point VASP calc + virial/config_type bookkeeping + per-step xyz dump."""
    v = Ge_bulk.get_volume()
    natoms = len(Ge_bulk)
    os.makedirs(f'database/{config_type}', exist_ok=True)
    os.makedirs(f'mydatabase/{config_type}', exist_ok=True)
    with open(f'database/{config_type}/volume', 'a') as vfile:
        vfile.write(str(n + 1) + ' , ' + str(v / natoms) + '\n')

    Ge_bulk.set_calculator(set_cal())
    Ge_bulk.get_potential_energy(force_consistent=True)  # TOTEN

    stress = Ge_bulk.get_stress(voigt=False)
    Ge_bulk.info['virial'] = v * -stress
    del Ge_bulk.calc.results['dipole']
    del Ge_bulk.calc.results['magmom']
    del Ge_bulk.calc.results['magmoms']
    Ge_bulk.info["config_type"] = config_type

    Ge_add = copy.deepcopy(Ge_bulk)
    bulk_list.append(Ge_add)
    write(f"mydatabase/{config_type}/random_{n}.xyz", Ge_add, format='extxyz')


def _create_cell_distorted(Ge_bulk, nstruc, outfile, config_type, dist_max=0.1):
    """Cell-matrix distortion loop shared by the cubic phases (diamond/fcc/bcc/hcp/bc8)."""
    bulk_list = []
    perfect_cell = Ge_bulk.get_cell()
    maxVal = np.max(perfect_cell)

    for n in range(int(nstruc)):
        rands = np.random.uniform(-dist_max, dist_max, 9).reshape(3, 3)
        distortions = maxVal * rands
        Ge_bulk.set_cell(perfect_cell + distortions, scale_atoms=True)
        _run_and_write(Ge_bulk, n, outfile, config_type, bulk_list)

    write(outfile, bulk_list, format='extxyz')


def create_diamond(nstruc, outfile, a=5.7620):
    Ge_bulk = bulk('Ge', 'diamond', a=a)
    _create_cell_distorted(Ge_bulk, nstruc, outfile, "distorted_bulk")


def create_fcc(nstruc, outfile, a=4.2749):
    Ge_bulk = bulk('Ge', 'fcc', a=a)
    _create_cell_distorted(Ge_bulk, nstruc, outfile, "fcc_bulk")


def create_bcc(nstruc, outfile, a=3.3858):
    Ge_bulk = bulk('Ge', 'bcc', a=a)
    _create_cell_distorted(Ge_bulk, nstruc, outfile, "bcc_bulk")


def create_hcp(nstruc, outfile, a=3.0118):
    Ge_bulk = bulk('Ge', 'hcp', a=a)
    _create_cell_distorted(Ge_bulk, nstruc, outfile, "hcp_bulk")


def create_bc8(nstruc, outfile, a=7.053):
    lattice = np.array([[-0.5 * a, 0.5 * a, 0.5 * a],
                         [0.5 * a, -0.5 * a, 0.5 * a],
                         [0.5 * a, 0.5 * a, -0.5 * a]])
    x = 0.1013
    basis = np.array([
        [x * a, x * a, x * a], [-x * a, -x * a, -x * a],
        [x * a, -x * a, (0.5 - x) * a], [-x * a, x * a, -(0.5 - x) * a],
        [(0.5 - x) * a, x * a, -x * a], [-(0.5 - x) * a, -x * a, x * a],
        [-x * a, (0.5 - x) * a, x * a], [x * a, -(0.5 - x) * a, -x * a],
    ])
    Ge_bulk = Atoms('Ge8', positions=basis, cell=lattice, pbc=[1, 1, 1])
    _create_cell_distorted(Ge_bulk, nstruc, outfile, "bc8_bulk")


def create_hd(nstruc, outfile, a=4.0561, c_a_ratio=1.6496):
    """Hexagonal diamond (lonsdaleite)."""
    bulk_list = []
    a_per = a
    z = 0.06281
    for n in range(int(nstruc)):
        rand = np.random.uniform(1 - 0.1, 1 + 0.1, 1)
        a = a_per * rand[0]
        c = a * c_a_ratio
        lattice = np.array([[0.5 * a, -0.5 * np.sqrt(3) * a, 0],
                             [0.5 * a, 0.5 * np.sqrt(3) * a, 0],
                             [0, 0, c]])
        basis = np.array([
            [0.5 * a, a / np.sqrt(12), z * c],
            [0.5 * a, -a / np.sqrt(12), (0.5 + z) * c],
            [0.5 * a, a / np.sqrt(12), (0.5 - z) * c],
            [0.5 * a, -a / np.sqrt(12), (-z) * c],
        ])
        Ge_bulk = Atoms('Ge4', positions=basis, cell=lattice, pbc=[1, 1, 1])
        _run_and_write(Ge_bulk, n, outfile, "hd", bulk_list)

    write(outfile, bulk_list, format='extxyz')


def create_betaTin(nstruc, outfile, a=5.1856, c_a_ratio=0.5527):
    bulk_list = []
    a_per = a
    for n in range(int(nstruc)):
        rand = np.random.uniform(1 - 0.1, 1 + 0.1, 1)
        a = a_per * rand[0]
        c = a * c_a_ratio
        lattice = np.array([[a, 0, 0], [0, a, 0], [0.5 * a, 0.5 * a, 0.5 * c]])
        basis = np.array([[0, -0.25 * a, 0.125 * c], [0, 0.25 * a, -0.125 * c]])
        Ge_bulk = Atoms('Ge2', positions=basis, cell=lattice, pbc=[1, 1, 1])
        _run_and_write(Ge_bulk, n, outfile, "betaTin", bulk_list)

    write(outfile, bulk_list, format='extxyz')


def create_st12(nstruc, outfile, a=6.0177, c_a_ratio=1.182):
    bulk_list = []
    a_per = a
    x1, x2, y2, z2 = 0.0874, 0.1709, 0.3704, 0.2525
    for n in range(int(nstruc)):
        rand = np.random.uniform(1 - 0.1, 1 + 0.1, 1)
        a = a_per * rand[0]
        c = a * c_a_ratio
        lattice = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
        basis = np.array([
            [x1 * a, x1 * a, 0], [-x1 * a, -x1 * a, 0.5 * c],
            [(0.5 - x1) * a, (0.5 + x1) * a, 0.75 * c],
            [(0.5 + x1) * a, (0.5 - x1) * a, 0.25 * c],
            [x2 * a, y2 * a, z2 * c], [-x2 * a, -y2 * a, (0.5 + z2) * c],
            [(0.5 - y2) * a, (0.5 + x2) * a, (0.75 + z2) * c],
            [(0.5 + y2) * a, (0.5 - x2) * a, (0.25 + z2) * c],
            [y2 * a, x2 * a, -z2 * c], [-y2 * a, -x2 * a, (0.5 - z2) * c],
            [(0.5 - x2) * a, (0.5 + y2) * a, (0.75 - z2) * c],
            [(0.5 + x2) * a, (0.5 - y2) * a, (0.25 - z2) * c],
        ])
        Ge_bulk = Atoms('Ge12', positions=basis, cell=lattice, pbc=[1, 1, 1])
        _run_and_write(Ge_bulk, n, outfile, "st12", bulk_list)

    write(outfile, bulk_list, format='extxyz')


PHASES = {
    "diamond": create_diamond,
    "fcc": create_fcc,
    "bcc": create_bcc,
    "hcp": create_hcp,
    "bc8": create_bc8,
    "hd": create_hd,
    "betaTin": create_betaTin,
    "st12": create_st12,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", dest="phase", choices=PHASES.keys(), required=True)
    parser.add_argument("-n", "--nstruc", dest="nstruc", required=True)
    parser.add_argument("-f", "--outfile", dest="outfile", required=True)
    parser.add_argument("-l", "--lattice", dest="lattice", type=float, default=None)
    parser.add_argument("--c-a-ratio", dest="c_a_ratio", type=float, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.lattice is not None:
        kwargs["a"] = args.lattice
    if args.c_a_ratio is not None:
        kwargs["c_a_ratio"] = args.c_a_ratio

    PHASES[args.phase](int(args.nstruc), str(args.outfile), **kwargs)
