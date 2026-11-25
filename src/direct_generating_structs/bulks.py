"""
Generate perfect or distorted bulk Ge structures (8 phases: diamond, fcc, bcc,
hcp, hd, betaTin, bc8, st12).

--format extxyz (default) runs a VASP single-point calculation on each
structure and writes an extended XYZ database suitable for MTP training data.
--format lammps-data writes geometry-only LAMMPS data file(s) (atom_style
atomic) and runs no VASP; with several structures, files are numbered
<stem>_NNN<suffix>.

--mode distorted (default) generates --nstruc randomly distorted cells;
--mode perfect generates a single undistorted cell at the given lattice
parameters (--nstruc is ignored).

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


def _run_and_write(Ge_bulk, n, outfile_dir, config_type, bulk_list):
    """Single-point VASP calc + virial/config_type bookkeeping + per-step xyz dump."""
    from vasp_settings import set_cal  # imported here so --format lammps-data needs no VASP setup

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


def _n_iter(nstruc, distort):
    """Number of structures to generate: nstruc if distorting, else one perfect cell."""
    return int(nstruc) if distort else 1


def _lattice_scale(distort, dist_max=0.1):
    """Random isotropic lattice-constant scale factor (1.0 for a perfect cell)."""
    return np.random.uniform(1 - dist_max, 1 + dist_max) if distort else 1.0


def _write_extxyz(structs, outfile, config_type):
    """Run VASP on every structure and write the labelled extended XYZ database."""
    bulk_list = []
    for n, Ge_bulk in enumerate(structs):
        _run_and_write(Ge_bulk, n, outfile, config_type, bulk_list)
    write(outfile, bulk_list, format='extxyz')


def _write_lammps_data(structs, outfile):
    """Write geometry-only LAMMPS data file(s) (atom_style atomic); no VASP is run.

    One structure goes to `outfile` as given; several go to <stem>_NNN<suffix>.
    """
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    for n, Ge_bulk in enumerate(structs):
        path = out if len(structs) == 1 else out.with_name(f"{out.stem}_{n:03d}{out.suffix}")
        atoms = Ge_bulk.copy()
        atoms.wrap()
        write(str(path), atoms, format='lammps-data', atom_style='atomic')


def _cell_distorted(Ge_bulk, nstruc, distort=True, dist_max=0.1):
    """Cell-matrix distortion shared by the cubic phases (diamond/fcc/bcc/hcp/bc8)."""
    perfect_cell = Ge_bulk.get_cell()
    maxVal = np.max(perfect_cell)
    structs = []
    for n in range(_n_iter(nstruc, distort)):
        atoms = Ge_bulk.copy()
        if distort:
            rands = np.random.uniform(-dist_max, dist_max, 9).reshape(3, 3)
            distortions = maxVal * rands
            atoms.set_cell(perfect_cell + distortions, scale_atoms=True)
        structs.append(atoms)
    return structs


# Each build_<phase>() returns (list of Atoms, config_type).

def build_diamond(nstruc, a=5.7620, distort=True):
    Ge_bulk = bulk('Ge', 'diamond', a=a)
    config_type = "distorted_bulk" if distort else "bulk"
    return _cell_distorted(Ge_bulk, nstruc, distort), config_type


def build_fcc(nstruc, a=4.2749, distort=True):
    return _cell_distorted(bulk('Ge', 'fcc', a=a), nstruc, distort), "fcc_bulk"


def build_bcc(nstruc, a=3.3858, distort=True):
    return _cell_distorted(bulk('Ge', 'bcc', a=a), nstruc, distort), "bcc_bulk"


def build_hcp(nstruc, a=3.0118, distort=True):
    return _cell_distorted(bulk('Ge', 'hcp', a=a), nstruc, distort), "hcp_bulk"


def build_bc8(nstruc, a=7.053, distort=True):
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
    return _cell_distorted(Ge_bulk, nstruc, distort), "bc8_bulk"


def build_hd(nstruc, a=4.0561, c_a_ratio=1.6496, distort=True):
    """Hexagonal diamond (lonsdaleite)."""
    structs = []
    a_per = a
    z = 0.06281
    for n in range(_n_iter(nstruc, distort)):
        a = a_per * _lattice_scale(distort)
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
        structs.append(Atoms('Ge4', positions=basis, cell=lattice, pbc=[1, 1, 1]))
    return structs, "hd"


def build_betaTin(nstruc, a=5.1856, c_a_ratio=0.5527, distort=True):
    structs = []
    a_per = a
    for n in range(_n_iter(nstruc, distort)):
        a = a_per * _lattice_scale(distort)
        c = a * c_a_ratio
        lattice = np.array([[a, 0, 0], [0, a, 0], [0.5 * a, 0.5 * a, 0.5 * c]])
        basis = np.array([[0, -0.25 * a, 0.125 * c], [0, 0.25 * a, -0.125 * c]])
        structs.append(Atoms('Ge2', positions=basis, cell=lattice, pbc=[1, 1, 1]))
    return structs, "betaTin"


def build_st12(nstruc, a=6.0177, c_a_ratio=1.182, distort=True):
    structs = []
    a_per = a
    x1, x2, y2, z2 = 0.0874, 0.1709, 0.3704, 0.2525
    for n in range(_n_iter(nstruc, distort)):
        a = a_per * _lattice_scale(distort)
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
        structs.append(Atoms('Ge12', positions=basis, cell=lattice, pbc=[1, 1, 1]))
    return structs, "st12"


PHASES = {
    "diamond": build_diamond,
    "fcc": build_fcc,
    "bcc": build_bcc,
    "hcp": build_hcp,
    "bc8": build_bc8,
    "hd": build_hd,
    "betaTin": build_betaTin,
    "st12": build_st12,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--phase", dest="phase", choices=PHASES.keys(), required=True)
    parser.add_argument("-m", "--mode", dest="mode", choices=["distorted", "perfect"],
                        default="distorted",
                        help="distorted: nstruc random cells; perfect: one undistorted cell")
    parser.add_argument("-n", "--nstruc", dest="nstruc", type=int, default=None,
                        help="number of distorted structures (required for --mode distorted)")
    parser.add_argument("-f", "--outfile", dest="outfile", required=True)
    parser.add_argument("--format", dest="fmt", choices=["extxyz", "lammps-data"],
                        default="extxyz",
                        help="extxyz: VASP-labelled database; lammps-data: geometry only, no VASP")
    parser.add_argument("-l", "--lattice", dest="lattice", type=float, default=None)
    parser.add_argument("--c-a-ratio", dest="c_a_ratio", type=float, default=None)
    args = parser.parse_args()

    distort = args.mode == "distorted"
    if distort and args.nstruc is None:
        parser.error("--nstruc is required for --mode distorted")

    kwargs = {"distort": distort}
    if args.lattice is not None:
        kwargs["a"] = args.lattice
    if args.c_a_ratio is not None:
        kwargs["c_a_ratio"] = args.c_a_ratio

    structs, config_type = PHASES[args.phase](args.nstruc if distort else 1, **kwargs)
    if args.fmt == "lammps-data":
        _write_lammps_data(structs, args.outfile)
    else:
        _write_extxyz(structs, args.outfile, config_type)
