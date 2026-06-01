"""Shared utilities for MTP quality tests."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

MLP_DEFAULT = "/scratch/project_2012355/Paper_3/mlip-3-prune/bin/mlp"
_OPENBLAS_LIB = "/appl/spack/v023/install-tree/gcc-14.2.0/openblas-0.3.28-r66ni7/lib"


def _mlp_env():
    """Return environment with libopenblas on LD_LIBRARY_PATH."""
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{_OPENBLAS_LIB}:{existing}" if existing else _OPENBLAS_LIB
    return env


def load_structure(struct_path: str | None):
    """Load a reference structure.

    Accepts CFG (MLIP), XYZ, LAMMPS data, or any ASE-readable format.
    Returns None → creates a 2-atom Ge diamond primitive cell via ASE.
    Returns (cell [3×3 Å], positions [N×3 Å], types [int list]).
    """
    if struct_path is None:
        from ase.build import bulk
        atoms = bulk("Ge", crystalstructure="diamond", a=5.658)
        return atoms.get_cell().array.copy(), atoms.get_positions(), [0] * len(atoms)

    path = Path(struct_path)
    if path.suffix == ".cfg":
        return _read_cfg_geometry(path)

    from ase.io import read as ase_read
    atoms = ase_read(str(path))
    cell = atoms.get_cell().array.copy()
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    unique = sorted(set(syms))
    sym_idx = {s: i for i, s in enumerate(unique)}
    return cell, pos, [sym_idx[s] for s in syms]


def _read_cfg_geometry(path: Path):
    """Parse the first CFG block → (cell, positions, types)."""
    text = path.read_text()
    m = re.search(r"BEGIN_CFG(.*?)END_CFG", text, re.DOTALL)
    if not m:
        raise ValueError(f"No CFG block found in {path}")
    lines = m.group(1).splitlines()
    n_atoms, cell, positions, types, mode = 0, [], [], [], None

    for line in lines:
        s = line.strip()
        if s == "Size":
            mode = "size"
        elif mode == "size":
            n_atoms = int(s); mode = None
        elif s == "Supercell":
            mode = "supercell"
        elif mode == "supercell":
            cell.append([float(x) for x in s.split()])
            if len(cell) == 3:
                mode = None
        elif s.startswith("AtomData:"):
            mode = "atoms"
        elif mode == "atoms" and s and not s.startswith(
            ("Feature", "Energy", "PlusStress", "BEGIN", "END")
        ):
            parts = s.split()
            if len(parts) >= 5:
                types.append(int(parts[1]))
                positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                if len(positions) == n_atoms:
                    mode = None

    return np.array(cell), np.array(positions), types


def write_bare_cfg(configs: list, path: Path) -> None:
    """Write geometry-only CFG blocks (no energy/forces) for mlp calculate_efs."""
    with open(path, "w") as f:
        for cell, positions, types in configs:
            n = len(positions)
            f.write("BEGIN_CFG\n")
            f.write(" Size\n")
            f.write(f"    {n}\n")
            f.write(" Supercell\n")
            for row in cell:
                f.write(f"    {row[0]:16.6f}  {row[1]:16.6f}  {row[2]:16.6f}\n")
            f.write(
                " AtomData:  id type       cartes_x      cartes_y      cartes_z\n"
            )
            for i, (t, pos) in enumerate(zip(types, positions), start=1):
                f.write(
                    f"    {i:8d}  {t:3d}    "
                    f"{pos[0]:14.6f}  {pos[1]:14.6f}  {pos[2]:14.6f}\n"
                )
            f.write("END_CFG\n\n")


def calc_efs(mlp: str, pot: str, in_cfg: Path, out_cfg: Path, quiet: bool = False) -> int:
    """Call `mlp calculate_efs pot cfg` (modifies in-place); copy result to out_cfg."""
    shutil.copy(in_cfg, out_cfg)
    cmd = [mlp, "calculate_efs", pot, str(out_cfg)]
    if not quiet:
        print("  Running:", " ".join(cmd))
    devnull = subprocess.DEVNULL if quiet else None
    return subprocess.run(cmd, env=_mlp_env(), stdout=devnull, stderr=devnull).returncode


# ---------------------------------------------------------------------------
# RDF utilities — shared by liquid_rdf.py and amorphous_rdf.py
# ---------------------------------------------------------------------------

def compute_rdf(positions: np.ndarray, cell: np.ndarray,
                rmax: float = 8.0, nbins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Compute g(r) for a single frame using vectorised minimum-image distances."""
    N = len(positions)
    dr = rmax / nbins
    cell_inv = np.linalg.inv(cell)

    diff = positions[:, None, :] - positions[None, :, :]  # (N,N,3)
    frac = diff @ cell_inv
    frac -= np.round(frac)
    diff_mic = frac @ cell
    dists = np.linalg.norm(diff_mic, axis=-1)  # (N,N)
    np.fill_diagonal(dists, rmax + 1.0)

    flat = dists.flatten()
    flat = flat[flat < rmax]
    hist, _ = np.histogram(flat, bins=nbins, range=(0.0, rmax))

    r = np.linspace(dr / 2.0, rmax - dr / 2.0, nbins)
    volume = abs(np.linalg.det(cell))
    rho = N / volume
    shell_vol = (4.0 / 3.0) * np.pi * ((r + dr / 2.0) ** 3 - (r - dr / 2.0) ** 3)
    g_r = hist / (N * rho * shell_vol)
    return r, g_r


def read_lammps_dump(path: Path, n_atoms: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Parse LAMMPS custom dump → list of (cell, positions).

    Reads ITEM: BOX BOUNDS for cell and ITEM: ATOMS id type x y z for positions.
    Returns frames where both cell and full atom count are available.
    """
    frames = []
    text = path.read_text()
    blocks = text.split("ITEM: TIMESTEP")
    for block in blocks[1:]:
        lines = block.splitlines()
        # Box bounds
        try:
            box_line = next(i for i, l in enumerate(lines) if "ITEM: BOX BOUNDS" in l)
            lx = float(lines[box_line + 1].split()[1]) - float(lines[box_line + 1].split()[0])
            ly = float(lines[box_line + 2].split()[1]) - float(lines[box_line + 2].split()[0])
            lz = float(lines[box_line + 3].split()[1]) - float(lines[box_line + 3].split()[0])
            cell = np.diag([lx, ly, lz])
        except (StopIteration, IndexError, ValueError):
            cell = None

        try:
            atom_line = next(i for i, l in enumerate(lines) if "ITEM: ATOMS" in l)
        except StopIteration:
            continue

        pos = []
        for line in lines[atom_line + 1 : atom_line + 1 + n_atoms]:
            parts = line.split()
            if len(parts) >= 5:
                pos.append([float(parts[2]), float(parts[3]), float(parts[4])])
        if len(pos) == n_atoms and cell is not None:
            frames.append((cell, np.array(pos)))
    return frames


def coordination_number(r: np.ndarray, g_r: np.ndarray,
                        rho: float, r_cutoff: float = 3.5) -> float:
    """Integrate g(r) to get coordination number up to r_cutoff."""
    mask = r <= r_cutoff
    dr = r[1] - r[0]
    return float(4.0 * np.pi * rho * np.trapz(g_r[mask] * r[mask] ** 2, r[mask]))


def load_ref_rdf(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load a two-column reference RDF file with a 2-line #-prefixed header."""
    if path is None or not Path(path).exists():
        return None, None
    data = np.loadtxt(path, skiprows=2)
    return data[:, 0], data[:, 1]
