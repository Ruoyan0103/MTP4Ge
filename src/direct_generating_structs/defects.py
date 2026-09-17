"""
Generate point-defect Ge structures (vacancy, 1NN/2NN divacancy, and T/H/X/B
interstitials) in a diamond-cubic supercell, writing VASP POSCAR + LAMMPS
data files for external DFT single-point runs, and collecting the resulting
vasprun.xml outputs into an MLIP-3 CFG dataset.

Ported from /scratch/project_2012355/Paper_3/00-subsets/05-defects/data/gen_data.py
to match the structure of bulks.py.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.geometry import get_distances

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vasp_settings import set_cal


def _sc(a0: float, repeat=(1, 1, 1)):
    """Return diamond-cubic conventional-cell supercell."""
    return bulk("Ge", crystalstructure="diamond", a=a0, cubic=True).repeat(repeat)


def build_bulk_ref(a0: float, repeat=(2, 2, 2)):
    """Defect-free supercell, for the mu_Ge reference energy."""
    atoms = _sc(a0, repeat)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_vacancy(a0: float, repeat=(2, 2, 2)):
    """Remove the atom nearest the origin."""
    atoms = _sc(a0, repeat)
    dists = np.linalg.norm(atoms.get_positions(), axis=1)
    del atoms[int(np.argmin(dists))]
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def _neighbor_shell_index(atoms, center, shell: int, tol: float = 1e-2):
    """Index of an atom in the `shell`-th nearest-neighbor shell of `center` (1 = 1NN)."""
    _, d = get_distances([center], atoms.get_positions(), cell=atoms.get_cell(), pbc=True)
    d = d[0]
    order = np.argsort(d)
    shells = []
    for val in d[order]:
        if not shells or val - shells[-1] > tol:
            shells.append(val)
        if len(shells) >= shell:
            break
    target = shells[shell - 1]
    return int(order[np.argmin(np.abs(d[order] - target))])


def build_1NNdivacancy(a0: float, repeat=(2, 2, 2)):
    """Remove an atom and one of its 1NN (nearest-neighbor) atoms."""
    atoms = _sc(a0, repeat)
    dists = np.linalg.norm(atoms.get_positions(), axis=1)
    idx0 = int(np.argmin(dists))
    pos0 = atoms.get_positions()[idx0].copy()
    del atoms[idx0]
    del atoms[_neighbor_shell_index(atoms, pos0, shell=1)]
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_2NNdivacancy(a0: float, repeat=(2, 2, 2)):
    """Remove an atom and one of its 2NN (second-nearest-neighbor) atoms."""
    atoms = _sc(a0, repeat)
    dists = np.linalg.norm(atoms.get_positions(), axis=1)
    idx0 = int(np.argmin(dists))
    pos0 = atoms.get_positions()[idx0].copy()
    del atoms[idx0]
    del atoms[_neighbor_shell_index(atoms, pos0, shell=2)]
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_T_interstitial(a0: float, repeat=(2, 2, 2)):
    """
    Tetrahedral void: (1/2,1/2,1/2) fractional of conventional cell.
    Surrounded by 4 B-sublattice atoms at distance a*sqrt(3)/4 (= bond length).
    """
    atoms = _sc(a0, repeat)
    t_pos = np.array([a0 / 2, a0 / 2, a0 / 2])
    atoms += Atoms("Ge", positions=[t_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_H_interstitial(a0: float, repeat=(2, 2, 2)):
    """
    Hexagonal site: (0.625, 0.625, 0.625) fractional of conventional cell.
    Surrounded by 6 atoms in a hexagonal ring.
    """
    atoms = _sc(a0, repeat)
    h_pos = np.array([0.625 * a0, 0.625 * a0, 0.625 * a0])
    atoms += Atoms("Ge", positions=[h_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


def build_X_dumbbell(a0: float, repeat=(2, 2, 2), half_disp: float = 1.2):
    """
    <110> split dumbbell: remove atom at origin, add two Ge displaced
    +/-half_disp Angstrom along [1,1,0]/sqrt(2). Default half_disp ~ bond_length/2 ~ 1.2 A.
    """
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


def build_B_bond_center(a0: float, repeat=(2, 2, 2)):
    """
    Bond-centre site: midpoint of the bond between the atom at (0,0,0) and its
    nearest neighbour at (a/4, a/4, a/4). Position = (a/8, a/8, a/8).
    """
    atoms = _sc(a0, repeat)
    b_pos = np.array([a0 / 8, a0 / 8, a0 / 8])
    atoms += Atoms("Ge", positions=[b_pos], cell=atoms.get_cell(), pbc=True)
    return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)


DEFECTS = {
    "bulk_ref": build_bulk_ref,
    "vac": build_vacancy,
    "1NN_vac": build_1NNdivacancy,
    "2NN_vac": build_2NNdivacancy,
    "T_int": build_T_interstitial,
    "H_int": build_H_interstitial,
    "X_int": build_X_dumbbell,
    "B_int": build_B_bond_center,
}


def _write_defect_inputs(name, builder, a0, repeat, out_dir):
    """Write POSCAR + LAMMPS data (and a relaxed LAMMPS data, if a CONTCAR is already present) for one defect."""
    cell, positions, _species = builder(a0, repeat=repeat)
    atoms = Atoms(symbols=["Ge"] * len(positions), positions=positions, cell=cell, pbc=True)

    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    write(os.path.join(folder, "POSCAR-333"), atoms, format="vasp")
    write(
        os.path.join(folder, "unrelaxed_lammps-333.data"),
        atoms,
        format="lammps-data",
        atom_style="atomic",
    )

    contcar = os.path.join(folder, "CONTCAR")
    if os.path.exists(contcar):
        relaxed = read(contcar, format="vasp")
        write(
            os.path.join(folder, "relaxed_lammps.data"),
            relaxed,
            format="lammps-data",
            atom_style="atomic",
        )

    calc = set_cal()
    calc.directory = folder
    calc.write_input(atoms)

    return folder


SPECIES_MAP = {'Ge': 0}


def collect_dft_to_cfg(job_list='job_list.txt', cfg_out='dataset.cfg', data_dir=None):
    """
    Read vasprun.xml from each single-point DFT run listed in job_list.txt
    (as produced by _write_defect_inputs) and write the collected
    energies/forces/stresses to a single MLIP-3 CFG file.
    """
    data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(data_dir, job_list)) as f:
        job_dirs = [line.strip() for line in f if line.strip()]

    species_map = SPECIES_MAP.copy()
    next_idx = max(species_map.values(), default=-1) + 1

    frames = []
    for orig_idx, job_dir in enumerate(job_dirs):
        vasprun = os.path.join(job_dir, 'vasprun.xml')
        if not os.path.exists(vasprun):
            print(f'  WARNING: missing {vasprun}, skipping')
            continue
        try:
            atoms = read(vasprun, format='vasp-xml', index=-1)
        except Exception as e:
            print(f'  WARNING: failed to read {vasprun} ({e}), skipping')
            continue
        atoms.info['config_type'] = os.path.relpath(job_dir, data_dir)
        frames.append((orig_idx, atoms))

    n_written = 0
    with open(os.path.join(data_dir, cfg_out), 'w') as f:
        for orig_idx, atoms in frames:
            calc_results = atoms.calc.results if atoms.calc else {}
            energy = calc_results.get('free_energy')
            if energy is None:
                continue

            cell = atoms.get_cell()
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()

            try:
                forces = atoms.get_forces()
            except Exception:
                forces = np.zeros_like(positions)

            try:
                vol = atoms.get_volume()
                stress_voigt = atoms.get_stress(voigt=True)  # eV/A^3
                virial = -stress_voigt * vol                  # eV
            except Exception:
                virial = None

            types = []
            for sym in symbols:
                if sym not in species_map:
                    species_map[sym] = next_idx
                    next_idx += 1
                types.append(species_map[sym])

            f.write('BEGIN_CFG\n')
            f.write(' Size\n')
            f.write(f'    {len(atoms)}\n')
            f.write(' Supercell\n')
            for row in cell:
                f.write(f'    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n')
            f.write(
                ' AtomData:  id type       cartes_x      cartes_y'
                '      cartes_z           fx          fy          fz\n'
            )
            for k, (t, pos, frc) in enumerate(zip(types, positions, forces), start=1):
                f.write(
                    f'    {k:8d}  {t:3d}  '
                    f'  {pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}'
                    f'  {frc[0]:12.6f}  {frc[1]:12.6f}  {frc[2]:12.6f}\n'
                )
            f.write(' Energy\n')
            f.write(f'    {energy:.12f}\n')
            if virial is not None:
                f.write(' PlusStress:  xx          yy          zz'
                        '          yz          xz          xy\n')
                f.write(
                    '    '
                    + '  '.join(f'{v:12.5f}' for v in virial)
                    + '\n'
                )
            f.write(f' Feature   orig_index\t {orig_idx}\n')
            f.write(f' Feature   type\t {atoms.info["config_type"]}\n')
            f.write('END_CFG\n\n')
            n_written += 1

    print(f'  Wrote {n_written} configs -> {cfg_out}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--defects", dest="defects", nargs="+", choices=DEFECTS.keys(),
                         default=list(DEFECTS.keys()))
    parser.add_argument("-l", "--lattice", dest="lattice", type=float, default=5.762)
    parser.add_argument("-r", "--repeat", dest="repeat", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("-o", "--outdir", dest="outdir",
                         default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "supercell-222"))
    parser.add_argument("--collect", dest="collect", action="store_true",
                         help="Instead of generating inputs, collect vasprun.xml results into a CFG dataset")
    parser.add_argument("--job-list", dest="job_list", default="job_list.txt")
    parser.add_argument("--cfg-out", dest="cfg_out", default="dataset.cfg")
    args = parser.parse_args()

    if args.collect:
        collect_dft_to_cfg(job_list=args.job_list, cfg_out=args.cfg_out)
    else:
        os.makedirs(args.outdir, exist_ok=True)
        job_dirs = [
            _write_defect_inputs(name, DEFECTS[name], args.lattice, tuple(args.repeat), args.outdir)
            for name in args.defects
        ]

        job_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.job_list)
        with open(job_list_path, 'w') as f:
            f.write('\n'.join(job_dirs) + '\n')
