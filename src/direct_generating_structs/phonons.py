"""
Generate phonon/thermal training-data structures for diamond-cubic Ge: 10
distinct supercell shapes (chosen so their commensurate q-points sample the
Brillouin zone differently), each rattled at a range of volumetric strains,
for external VASP single-point runs. Writes VASP input directories (via
vasp_settings.set_cal()) and collects the resulting vasprun.xml outputs into
an MLIP-3 CFG dataset.

Ported from /scratch/project_2012355/Paper_3/00-subsets/02-thermal/data/gen_data.py
to match the structure of bulks.py / defects.py.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk, make_supercell
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vasp_settings import set_cal


DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phonon-supercells")

R_VALUES = [-0.15, -0.12, -0.10, -0.07, -0.05, -0.03, -0.02,
            0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]


def _supercell_matrices():
    """10 integer transformation matrices for the 2-atom diamond-Ge primitive cell."""
    m1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    m2 = [[3, -1, -1], [0, 1, 0], [0, 0, 1]]

    # interchange rows, org: [1,2,-1  1,-1,0  0,0,1]
    m5_1 = [[1, 2, -1], [0, 0, 1], [1, -1, 0]]
    m5_2 = [[0, 0, 1], [1, -1, 0], [1, 2, -1]]
    m5_3 = [[1, -1, 0], [1, 2, -1], [0, 0, 1]]
    # multiplying three rows by -1
    m5_7 = [[-1, -2, 1], [-1, 1, 0], [0, 0, -1]]

    # interchange rows, org: [1,1,-1  1,-2,0  0,0,1]
    m7_1 = [[1, 1, -1], [0, 0, 1], [1, -2, 0]]
    m7_2 = [[0, 0, 1], [1, -2, 0], [1, 1, -1]]
    m7_3 = [[1, -2, 0], [1, 1, -1], [0, 0, 1]]
    # multiplying three rows by -1
    m7_7 = [[-1, -1, 1], [-1, 2, 0], [0, 0, -1]]

    return [m1, m2, m5_1, m5_2, m5_3, m5_7, m7_1, m7_2, m7_3, m7_7]


def crt_supercell(matrix, folder, a0=5.762):
    Ge_prim = bulk('Ge', 'diamond', a=a0)
    supercell = make_supercell(Ge_prim, matrix)
    os.makedirs(folder, exist_ok=True)
    write(os.path.join(folder, 'POSCAR'), supercell, format='vasp')
    return supercell


def crt_supercells(a0=5.762, outdir=DEFAULT_OUTDIR):
    """Builds the 10 diamond-Ge supercells and writes each one's POSCAR into outdir/sup-{1..10}/."""
    matrices = _supercell_matrices()
    for i, matrix in enumerate(matrices):
        crt_supercell(matrix, os.path.join(outdir, f'sup-{i + 1}'), a0=a0)


def displace(in_poscar, r=0.0, atom_disp=0.02):
    """
    r: target isotropic volumetric strain, e.g. -0.05 for -5% volume, 0.0 for equilibrium
    atom_disp (Å): stdev of rattle
    """
    s = (1 + r) ** (1 / 3)
    Ge_bulk = read(in_poscar, format='vasp')
    new_cell = Ge_bulk.get_cell() * s
    Ge_bulk.set_cell(new_cell, scale_atoms=True)
    seed = random.randint(0, 2**32 - 1)
    Ge_bulk.rattle(stdev=atom_disp, seed=seed)
    return Ge_bulk


def crt_rattled_copies(n, r_values, outdir=DEFAULT_OUTDIR):
    """
    n: number of rattled copies per (supercell, volume)
    r_values: list of target volumetric strains

    Each rattled copy gets its own VASP input directory (INCAR/POTCAR/KPOINTS/
    POSCAR from vasp_settings.set_cal()). All directories are collected into
    job_list.txt for use with a SLURM array job.

    WARNING: destructive -- wipes and rewrites every vol_{idx} folder. Only
    use for the very first batch. To add more samples to an existing batch
    without touching already-completed DFT runs, use add_rattled_copies().
    """
    job_dirs = []
    for i in range(1, 11):
        directory_path = os.path.join(outdir, f'sup-{i}')
        in_poscar = os.path.join(directory_path, 'POSCAR')
        for idx, r in enumerate(r_values):
            vol_path = os.path.join(directory_path, '01-rattled', f'vol_{idx}')
            if os.path.exists(vol_path):
                shutil.rmtree(vol_path)
            for j in range(n):
                rattled_path = os.path.join(vol_path, f'rattle_{j}')
                os.makedirs(rattled_path, exist_ok=True)
                atoms = displace(in_poscar, r=r)

                calc = set_cal()
                calc.directory = rattled_path
                calc.write_input(atoms)

                job_dirs.append(os.path.abspath(rattled_path))

    job_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_list.txt')
    with open(job_list_path, 'w') as f:
        f.write('\n'.join(job_dirs) + '\n')
    print(f'  Wrote {len(job_dirs)} job dirs -> {job_list_path}')


def add_rattled_copies(n_start, n_new, r_values, outdir=DEFAULT_OUTDIR, job_list_out='job_list_new.txt'):
    """
    Appends n_new additional rattled copies (indices n_start .. n_start+n_new-1)
    per (supercell, volume) WITHOUT touching any existing rattle_* folder --
    safe to run after a previous batch's DFT has already completed.

    n_start: first new rattle index (e.g. 1 if rattle_0 already exists from the
             original n=1 batch)
    n_new: number of additional copies to add per (supercell, volume)
    r_values: MUST start with the same values in the same order as the
              original call (new volumes may be appended at the end, but
              existing ones must keep their original index so vol_{idx}
              folders line up with already-completed runs)
    job_list_out: filename for the list of ONLY the newly created job dirs,
                  so you can submit just these to VASP without resubmitting
                  already-finished work
    """
    job_dirs = []
    for i in range(1, 11):
        directory_path = os.path.join(outdir, f'sup-{i}')
        in_poscar = os.path.join(directory_path, 'POSCAR')
        for idx, r in enumerate(r_values):
            vol_path = os.path.join(directory_path, '01-rattled', f'vol_{idx}')
            for j in range(n_start, n_start + n_new):
                rattled_path = os.path.join(vol_path, f'rattle_{j}')
                if os.path.exists(rattled_path):
                    print(f'  Skipping existing {rattled_path}')
                    continue
                os.makedirs(rattled_path, exist_ok=True)
                atoms = displace(in_poscar, r=r)

                calc = set_cal()
                calc.directory = rattled_path
                calc.write_input(atoms)

                job_dirs.append(os.path.abspath(rattled_path))

    job_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), job_list_out)
    with open(job_list_path, 'w') as f:
        f.write('\n'.join(job_dirs) + '\n')
    print(f'  Wrote {len(job_dirs)} new job dirs -> {job_list_path}')


SPECIES_MAP = {'Ge': 0}


def collect_dft_to_cfg(job_list='job_list.txt', cfg_out='dataset.cfg', data_dir=None):
    """
    Reads vasprun.xml from each single-point DFT run listed in job_list.txt
    (as produced by crt_rattled_copies / add_rattled_copies) and writes the
    collected energies/forces/stresses to a single MLIP-3 CFG file.
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
    sub = parser.add_subparsers(dest="stage", required=True)

    p_super = sub.add_parser("supercells", help="Build the 10 diamond-Ge supercells (POSCARs only)")
    p_super.add_argument("-l", "--lattice", type=float, default=5.762)
    p_super.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR)

    p_rattle = sub.add_parser("rattle", help="Rattle each supercell at each volume and write VASP inputs")
    p_rattle.add_argument("-n", type=int, default=1, help="rattled copies per (supercell, volume)")
    p_rattle.add_argument("-r", "--r-values", type=float, nargs="+", default=R_VALUES)
    p_rattle.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR)

    p_more = sub.add_parser("rattle-more", help="Append rattled copies without touching completed runs")
    p_more.add_argument("--n-start", type=int, required=True)
    p_more.add_argument("--n-new", type=int, required=True)
    p_more.add_argument("-r", "--r-values", type=float, nargs="+", default=R_VALUES)
    p_more.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR)
    p_more.add_argument("--job-list-out", default="job_list_new.txt")

    p_collect = sub.add_parser("collect", help="Collect vasprun.xml results into an MLIP-3 CFG dataset")
    p_collect.add_argument("--job-list", default="job_list.txt")
    p_collect.add_argument("--cfg-out", default="dataset.cfg")

    args = parser.parse_args()

    if args.stage == "supercells":
        crt_supercells(a0=args.lattice, outdir=args.outdir)
    elif args.stage == "rattle":
        crt_rattled_copies(args.n, args.r_values, outdir=args.outdir)
    elif args.stage == "rattle-more":
        add_rattled_copies(args.n_start, args.n_new, args.r_values,
                            outdir=args.outdir, job_list_out=args.job_list_out)
    elif args.stage == "collect":
        collect_dft_to_cfg(job_list=args.job_list, cfg_out=args.cfg_out)
