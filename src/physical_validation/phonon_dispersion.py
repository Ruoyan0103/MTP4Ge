"""
Phonon dispersion quality test for MTP potentials.

Generates displaced supercells via phonopy and evaluates forces with one of
three engines (--method):
  mtp          - `mlp calculate_efs` (default)
  lammps-mlip  - LAMMPS pair_style mlip (config/lammps/phonon_dispersion.in)
  lammps-nlh   - LAMMPS pair_style hybrid/overlay mtp nlh
                 (config/lammps/phonon_dispersion_mtp.in) — needed for
                 potentials trained with a radial basis type that neither
                 `mlp calculate_efs` nor `pair_style mlip` can load (fails
                 with "Wrong radial basis type"); see
                 src/physical_validation/elastic_constant/potential.mod for the same LAMMPS
                 pair_style workaround used for elastic constants.
Produces a side-by-side phonon band structure (Γ→X→K→Γ→L, FCC BZ, diamond
cubic Ge) and total DOS plot, overlaid with DFT/experiment references.

Usage:
    python src/physical_validation/phonon_dispersion.py --pot results/potentials/pot.almtp
    python src/physical_validation/phonon_dispersion.py --pot results/potentials/pot.almtp \\
        --alat 5.658 --supercell 2 --displacement 0.01 --npoints 51
    python src/physical_validation/phonon_dispersion.py --pot results/potentials/pot.almtp \\
        --method lammps-mlip --lammps "srun /path/to/lmp_mpi"
    python src/physical_validation/phonon_dispersion.py --pot results/potentials/20.mtp \\
        --method lammps-nlh --lammps "srun /path/to/lmp_mpi"
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# Must be imported before `phonopy` anywhere in this process: on this cluster
# a system (TYKKY) scipy install shadows the venv's scipy if phonopy resolves
# scipy first, causing an ABI mismatch (CXXABI_1.3.15 not found) deep inside
# ase.io -> ase.dft.kpoints -> scipy.optimize the first time ase.io is used.
# Importing ase.io up front caches the correct scipy in sys.modules first.
import ase.io  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import calc_efs, write_bare_cfg

DEFAULT_TRAIN_CONFIG = "config/training.yaml"
DEFAULT_AL_CONFIG = "config/active_learning.yaml"
LAMMPS_TEMPLATE = Path("config/lammps/phonon_dispersion.in")
LAMMPS_TEMPLATE_NLH = Path("config/lammps/phonon_dispersion_mtp.in")
GE_MASS = 72.630
DEFAULT_LAMMPS_NLH = "/projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"

# High-symmetry q-path in FCC primitive reciprocal coordinates: Γ→X→K | U→Γ→L
#
# Two disconnected path groups with a zone-boundary break:
#   Group 1: Γ→X→K  with K = (3/8, 3/8, 3/4) — approached from the X side (Z line)
#   Group 2: U→Γ→L  with U = (5/8, 1/4, 5/8) — approached from the Γ side (Σ line)
#
# K ≠ U numerically, so get_band_qpoints_and_path_connections produces
# path_connections = [True, False, True] — a path break at the zone boundary.


# Matches src/physical_validation/reference/phonon_dipsersion_DFT/band.conf's BAND grouping exactly:
#   Γ→X | (1/2,1/2,1)→K→Γ→L — the break sits right after X, not after
#   (1/2,1/2,1). (1/2,1/2,1) is periodicity+point-group equivalent to X
#   (see project notes), so this traces the (1/2,1/2,1)→K segment, not the
#   X→(1/2,1/2,1) segment the grouping above would produce. Kept deliberately
#   to match the digitized experimental reference's path convention.
_QPATH = [
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
    [[0.5, 0.5, 1.0], [0.375, 0.375, 0.75], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
]


_LABELS = [r"$\Gamma$", "X", "K", r"$\Gamma$", "L"]


# ---------------------------------------------------------------------------
# Structure conversion helpers
# ---------------------------------------------------------------------------

def _phonopy_to_tuple(ph_atoms):
    """Convert PhonopyAtoms → (cell, positions, types) for write_bare_cfg.

    All symbols map to integer type indices in sorted order (Ge only → type 0).
    """
    symbols = ph_atoms.symbols
    unique = sorted(set(symbols))
    sym_idx = {s: i for i, s in enumerate(unique)}
    types = [sym_idx[s] for s in symbols]
    return ph_atoms.cell.copy(), ph_atoms.positions.copy(), types


def _phonopy_to_ase(ph_atoms):
    """Convert PhonopyAtoms → ASE Atoms."""
    from ase import Atoms
    return Atoms(
        symbols=list(ph_atoms.symbols),
        cell=ph_atoms.cell.copy(),
        positions=ph_atoms.positions.copy(),
        pbc=True,
    )


# ---------------------------------------------------------------------------
# LAMMPS force evaluation (pair_style mlip)
# ---------------------------------------------------------------------------

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


def _run_lammps_forces(
    lammps_cmd: str, pot: str, supercells: list, outdir: Path,
) -> list[np.ndarray]:
    """Run one LAMMPS single-point evaluation (pair_style mlip) per displaced
    supercell and return the per-atom forces, in supercell order.
    """
    from ase.io import write as ase_write

    template = LAMMPS_TEMPLATE.read_text()
    pot_abs = str(Path(pot).resolve())
    lammps_dir = outdir / "lammps"

    forces_list = []
    for i, sc in enumerate(supercells):
        disp_dir = lammps_dir / f"disp_{i:03d}"
        disp_dir.mkdir(parents=True, exist_ok=True)

        atoms = _phonopy_to_ase(sc)
        ase_write(str(disp_dir / "structure.lammps"), atoms,
                  format="lammps-data", atom_style="atomic")

        text = (
            template
            .replace("${STRUCT_FILE}", "structure.lammps")
            .replace("${POT_PATH}", pot_abs)
            .replace("${MASS}", str(GE_MASS))
        )
        (disp_dir / "in.phonon").write_text(text)

        cmd = shlex.split(lammps_cmd) + ["-in", "in.phonon", "-log", "log.lammps", "-screen", "none"]
        print(f"  LAMMPS [{i + 1}/{len(supercells)}]:", " ".join(cmd), f"(cwd={disp_dir})")
        result = subprocess.run(cmd, cwd=str(disp_dir))
        if result.returncode != 0:
            raise RuntimeError(f"LAMMPS exited {result.returncode}; see {disp_dir / 'log.lammps'}")

        forces_list.append(_parse_forces_from_dump(disp_dir / "force.dump", len(atoms)))

    return forces_list


def _run_lammps_forces_nlh(
    lammps_cmd: str, pot: str, supercells: list, outdir: Path,
) -> list[np.ndarray]:
    """Same as _run_lammps_forces but with pair_style hybrid/overlay mtp nlh
    (config/lammps/phonon_dispersion_mtp.in) instead of pair_style mlip —
    for potentials whose radial basis type pair_style mlip can't load.
    """
    from ase.io import write as ase_write

    template = LAMMPS_TEMPLATE_NLH.read_text()
    pot_abs = str(Path(pot).resolve())
    lammps_dir = outdir / "lammps"

    forces_list = []
    for i, sc in enumerate(supercells):
        disp_dir = lammps_dir / f"disp_{i:03d}"
        disp_dir.mkdir(parents=True, exist_ok=True)

        atoms = _phonopy_to_ase(sc)
        ase_write(str(disp_dir / "structure.lammps"), atoms,
                  format="lammps-data", atom_style="atomic")

        text = (
            template
            .replace("${STRUCT_FILE}", "structure.lammps")
            .replace("${POT_PATH}", pot_abs)
            .replace("${MASS}", str(GE_MASS))
        )
        (disp_dir / "in.phonon").write_text(text)

        cmd = shlex.split(lammps_cmd) + ["-in", "in.phonon", "-log", "log.lammps", "-screen", "none"]
        print(f"  LAMMPS [{i + 1}/{len(supercells)}]:", " ".join(cmd), f"(cwd={disp_dir})")
        result = subprocess.run(cmd, cwd=str(disp_dir))
        if result.returncode != 0:
            raise RuntimeError(f"LAMMPS exited {result.returncode}; see {disp_dir / 'log.lammps'}")

        forces_list.append(_parse_forces_from_dump(disp_dir / "force.dump", len(atoms)))

    return forces_list


# ---------------------------------------------------------------------------
# CFG force parser
# ---------------------------------------------------------------------------

def _parse_forces_from_cfg(path: Path) -> list[np.ndarray]:
    """Extract per-atom forces from an EFS CFG output file.

    Returns a list of arrays shaped (n_atoms, 3) — one per CFG block.
    Column indices for fx/fy/fz are resolved from the AtomData header line
    so the parser is robust to column-order changes in mlp output.
    """
    forces_list = []
    text = path.read_text()
    for block in re.split(r"(?=BEGIN_CFG\b)", text):
        if not block.strip().startswith("BEGIN_CFG"):
            continue
        header_m = re.search(r"AtomData:\s*(.*)", block)
        if not header_m:
            continue
        cols = header_m.group(1).split()
        try:
            ix = cols.index("fx")
            iy = cols.index("fy")
            iz = cols.index("fz")
        except ValueError:
            continue
        atom_forces = []
        in_atoms = False
        for line in block.splitlines():
            s = line.strip()
            if "AtomData:" in s:
                in_atoms = True
                continue
            if in_atoms:
                if not s or re.match(
                    r"(Feature|Energy|PlusStress|BEGIN_CFG|END_CFG)", s
                ):
                    break
                parts = s.split()
                if len(parts) > max(ix, iy, iz):
                    atom_forces.append(
                        [float(parts[ix]), float(parts[iy]), float(parts[iz])]
                    )
        if atom_forces:
            forces_list.append(np.array(atom_forces))
    return forces_list


# ---------------------------------------------------------------------------
# Reference data (DFT / EXP) loading and x-axis rescaling
# ---------------------------------------------------------------------------

def _load_reference_data(path: Path):
    """Parse a DFT or EXP phonon dispersion data file.

    File format:
      - Optional header comment ``# End points of segments: x0 x1 x2 ...``
      - Comma-separated ``x, y`` data lines (EXP) or space-separated (DFT)
      - Single blank line  → piece break within a branch
      - Double blank line  → branch break

    Returns:
        (endpoints, branches)
        endpoints : list[float] or None — segment boundary x-values from header
        branches  : list[list[tuple[np.ndarray, np.ndarray]]]
                    outer list = branches; inner list = pieces; tuple = (x, y)
    """
    lines = path.read_text().splitlines()

    endpoints = None
    data_lines = []
    grab_endpoints_next = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if grab_endpoints_next:
                nums = re.findall(r"[\d.]+(?:\.\d*)?", stripped.lstrip("#").strip())
                endpoints = [float(n) for n in nums]
                grab_endpoints_next = False
            elif "end points" in stripped.lower():
                after_colon = stripped.split(":")[-1].strip().lstrip("#").strip()
                nums = re.findall(r"[\d.]+", after_colon)
                if nums:
                    endpoints = [float(n) for n in nums]
                else:
                    grab_endpoints_next = True
        else:
            grab_endpoints_next = False
            data_lines.append(stripped)

    branches: list = []
    current_branch: list = []
    px: list = []
    py: list = []
    prev_blank = False

    def _flush_piece():
        if px:
            current_branch.append((np.array(px), np.array(py)))
            px.clear()
            py.clear()

    def _flush_branch():
        _flush_piece()
        if current_branch:
            branches.append(list(current_branch))
            current_branch.clear()

    for line in data_lines:
        if not line:
            if prev_blank:
                _flush_branch()
                prev_blank = False
            else:
                _flush_piece()
                prev_blank = True
        else:
            prev_blank = False
            # auto-detect comma-separated (EXP) or space-separated (DFT)
            if "," in line:
                parts = line.split(",")
            else:
                parts = line.split()
            if len(parts) >= 2:
                try:
                    px.append(float(parts[0]))
                    py.append(float(parts[1]))
                except ValueError:
                    pass

    _flush_branch()
    return endpoints, branches


def _load_dos_data(path: Path):
    """Parse a total DOS file with space-separated columns: frequency[THz]  DOS.

    Returns (freq_arr, dos_arr) filtered to freq >= 0.
    """
    freqs, dos = [], []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                freqs.append(float(parts[0]))
                dos.append(float(parts[1]))
            except ValueError:
                pass
    freq_arr = np.array(freqs)
    dos_arr = np.array(dos)
    mask = freq_arr >= 0
    return freq_arr[mask], dos_arr[mask]


def _rescale_x(
    x_arr: np.ndarray,
    src_ends: list[float],
    dst_ends: list[float],
) -> np.ndarray:
    """Piecewise-linear x rescaling from src segment endpoints to dst endpoints.

    Points outside [src_ends[0], src_ends[-1]] are clamped to the nearest segment.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    x_clamped = np.clip(x_arr, src_ends[0], src_ends[-1])
    out = np.full_like(x_clamped, np.nan)
    for i in range(len(src_ends) - 1):
        x0, x1 = src_ends[i], src_ends[i + 1]
        d0, d1 = dst_ends[i], dst_ends[i + 1]
        mask = (x_clamped >= x0 - 1e-9) & (x_clamped <= x1 + 1e-9)
        if not mask.any():
            continue
        t = np.clip((x_clamped[mask] - x0) / (x1 - x0), 0.0, 1.0)
        out[mask] = d0 + t * (d1 - d0)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dispersion(
    band_dict: dict,
    labels: list[str],
    outdir: Path,
    dft_data=None,
    exp_data=None,
    mtp_dos=None,
    dft_dos_data=None,
) -> None:
    """Save phonon_dispersion.png: band structure (left) + total DOS (right).

    dft_data / exp_data : return value of _load_reference_data (endpoints, branches).
    mtp_dos             : (freq_arr, dos_arr) from phonopy get_total_dos_dict().
    dft_dos_data        : (freq_arr, dos_arr) from _load_dos_data().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    # phonopy returns distances as CUMULATIVE values across the whole path.
    distances   = band_dict["distances"]    # list of 1-D arrays (cumulative)
    frequencies = band_dict["frequencies"]  # list of (npts, nbands) arrays

    # MTP segment endpoints (cumulative x values at each high-symmetry point)
    mtp_ends = [float(np.array(distances[0])[0])]
    for seg_d in distances:
        mtp_ends.append(float(np.array(seg_d)[-1]))

    # DFT endpoint list used for rescaling both DFT and EXP data
    ref_ends = dft_data[0] if (dft_data is not None and dft_data[0] is not None) else None

    # y-axis upper limit covering all datasets
    all_mtp_freqs = np.concatenate([np.array(f).ravel() for f in frequencies])
    ymax = max(all_mtp_freqs.max() * 1.05, 10.0)
    if dft_dos_data is not None:
        ymax = max(ymax, float(dft_dos_data[0].max()) * 1.05)

    # 1×2 figure: dispersion (wider) + DOS (narrower), shared y-axis
    fig, (ax, ax_dos) = plt.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=(8, 5),
        layout="constrained",
    )

    # --- DFT overlay on dispersion (red dashed) ---
    dft_handle = None
    if dft_data is not None and ref_ends is not None:
        _, dft_branches = dft_data
        for branch in dft_branches:
            for x_arr, y_arr in branch:
                xr = _rescale_x(x_arr, ref_ends, mtp_ends)
                valid = ~np.isnan(xr)
                if not valid.any():
                    continue
                line, = ax.plot(
                    xr[valid], y_arr[valid],
                    color="tab:red", lw=0.9, ls="--", alpha=0.85, zorder=2,
                )
                if dft_handle is None:
                    dft_handle = line

    # --- EXP overlay on dispersion (black scatter dots) ---
    exp_handle = None
    if exp_data is not None and ref_ends is not None:
        _, exp_branches = exp_data
        for branch in exp_branches:
            for x_arr, y_arr in branch:
                xr = _rescale_x(x_arr, ref_ends, mtp_ends)
                valid = ~np.isnan(xr)
                if not valid.any():
                    continue
                sc = ax.scatter(
                    xr[valid], y_arr[valid],
                    color="k", s=6, zorder=4, linewidths=0,
                )
                if exp_handle is None:
                    exp_handle = sc

    # --- MTP bands on dispersion (blue solid) ---
    mtp_handle = None
    for seg_d, seg_f in zip(distances, frequencies):
        d = np.array(seg_d)
        f = np.array(seg_f)
        for branch in f.T:
            line, = ax.plot(d, branch, color="tab:blue", lw=1.1, zorder=3)
            if mtp_handle is None:
                mtp_handle = line

    # Vertical lines at high-symmetry points
    xtick_pos = [mtp_ends[0]]
    for seg_d in distances:
        x_end = float(np.array(seg_d)[-1])
        ax.axvline(x_end, color="k", lw=0.8, ls="-")
        xtick_pos.append(x_end)

    # Legend on dispersion panel
    legend_handles, legend_labels = [], []
    if mtp_handle is not None:
        legend_handles.append(mtp_handle)
        legend_labels.append("MTP")
    if dft_handle is not None:
        legend_handles.append(dft_handle)
        legend_labels.append("DFT")
    if exp_handle is not None:
        legend_handles.append(exp_handle)
        legend_labels.append("Exp.")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=9, loc="upper right",
                  framealpha=0.7)

    ax.set_xlim(xtick_pos[0], xtick_pos[-1])
    ax.set_ylim(0, ymax)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Frequency (THz)", fontsize=12)
    ax.set_title("Ge Phonon Dispersion", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="x", length=0)

    # --- DOS panel (x = DOS, y = frequency — shared with left panel) ---
    if dft_dos_data is not None:
        dft_freq_dos, dft_dos_vals = dft_dos_data
        ax_dos.plot(dft_dos_vals, dft_freq_dos,
                    color="tab:red", lw=0.9, ls="--", alpha=0.85, label="DFT")

    if mtp_dos is not None:
        mtp_freq_dos, mtp_dos_vals = mtp_dos
        mask = mtp_freq_dos >= 0
        ax_dos.plot(mtp_dos_vals[mask], mtp_freq_dos[mask],
                    color="tab:blue", lw=1.1, label="MTP")

    ax_dos.set_xlim(left=0)
    ax_dos.set_xlabel("DOS\n(states/THz)", fontsize=10)
    ax_dos.set_title("Total DOS", fontsize=13)
    ax_dos.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax_dos.tick_params(axis="y", length=0)
    ax_dos.yaxis.set_tick_params(labelleft=False)

    out_path = outdir / "phonon_dispersion.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(
    pot: str,
    outdir: Path,
    alat: float,
    supercell_size: int,
    displacement: float,
    npoints: int,
    mlp: str,
    dos_mesh: int,
    method: str = "mtp",
    lammps_cmd: str | None = None,
) -> None:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    outdir.mkdir(parents=True, exist_ok=True)

    # Conventional 8-atom cubic diamond cell + primitive_matrix reduction to the
    # 2-atom FCC primitive cell (ilearn convention). Supercells built this way
    # are orthogonal (LAMMPS-friendly); the FCC primitive cell's own ~60°
    # lattice vectors would otherwise produce triclinic tilt factors beyond
    # LAMMPS's default skew limit. The primitive reciprocal lattice — and
    # therefore _QPATH — is identical either way.
    unitcell = PhonopyAtoms(
        symbols=["Ge"] * 8,
        cell=np.eye(3) * alat,
        scaled_positions=[
            [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ],
    )
    S = supercell_size
    print(f"  Unit cell: 8 atoms (conventional cubic)  |  a = {alat:.4f} Å")
    print(f"  Supercell: {S}×{S}×{S} conventional = {S**3 * 8} atoms per displaced config")

    phonon = Phonopy(
        unitcell,
        supercell_matrix=[[S, 0, 0], [0, S, 0], [0, 0, S]],
        primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements
    print(f"  Generated {len(supercells)} displaced supercell(s)")

    if method in ("lammps-mlip", "lammps-nlh"):
        run_forces = _run_lammps_forces if method == "lammps-mlip" else _run_lammps_forces_nlh
        pair_style = "pair_style mlip" if method == "lammps-mlip" else "pair_style mtp+nlh"
        print(f"  Method: LAMMPS force evaluation ({pair_style}, {lammps_cmd})")
        cache = outdir / "lammps_forces.npz"
        if cache.exists():
            print(f"  Found existing {cache.name} — skipping LAMMPS calculation")
            data = np.load(cache)
            forces_list = [data[f"arr_{i}"] for i in range(len(supercells))]
        else:
            forces_list = run_forces(lammps_cmd, pot, supercells, outdir)
            np.savez(cache, *forces_list)
    else:
        configs = [_phonopy_to_tuple(sc) for sc in supercells]

        in_cfg  = outdir / "displaced.cfg"
        out_cfg = outdir / "displaced_efs.cfg"

        if out_cfg.exists():
            print(f"  Found existing {out_cfg.name} — skipping mlp calculation")
        else:
            write_bare_cfg(configs, in_cfg)
            rc = calc_efs(mlp, pot, in_cfg, out_cfg)
            if rc != 0:
                print(f"  Warning: mlp calculate_efs returned {rc}")
            if not out_cfg.exists():
                print("  Error: no EFS output produced.", file=sys.stderr)
                sys.exit(1)

        forces_list = _parse_forces_from_cfg(out_cfg)

    if len(forces_list) != len(supercells):
        print(
            f"  Error: expected {len(supercells)} force blocks, got {len(forces_list)}",
            file=sys.stderr,
        )
        sys.exit(1)

    phonon.forces = forces_list
    phonon.produce_force_constants()
    print("  Force constants computed")

    # --- Band structure ---
    qpoints, connections = get_band_qpoints_and_path_connections(
        _QPATH, npoints=npoints
    )
    phonon.run_band_structure(
        qpoints, path_connections=connections, labels=_LABELS
    )

    band_yaml = outdir / "band.yaml"
    phonon.write_yaml_band_structure(filename=str(band_yaml))
    print(f"  Band structure written to {band_yaml}")

    band_dict = phonon.get_band_structure_dict()

    # --- Total DOS ---
    print(f"  Computing total DOS on {dos_mesh}×{dos_mesh}×{dos_mesh} mesh ...")
    phonon.run_mesh([dos_mesh, dos_mesh, dos_mesh], is_gamma_center=True)
    phonon.run_total_dos()
    dos_dict = phonon.get_total_dos_dict()
    mtp_dos = (
        np.array(dos_dict["frequency_points"]),
        np.array(dos_dict["total_dos"]),
    )
    print(f"  Total DOS computed ({len(mtp_dos[0])} frequency points)")

    # --- Load reference overlays ---
    dft_path = Path("src/physical_validation/reference/phonon_dipsersion_DFT/raw-data.txt")
    exp_path = Path("src/physical_validation/reference/phonon_dipsersion_DFT/phonon_dispersion_EXP.dat")
    dft_dos_path = Path("src/physical_validation/reference/phonon_dipsersion_DFT/phonon_dos.dat")

    dft_data = _load_reference_data(dft_path) if dft_path.exists() else None
    exp_data = _load_reference_data(exp_path) if exp_path.exists() else None
    dft_dos_data = _load_dos_data(dft_dos_path) if dft_dos_path.exists() else None

    if dft_data is not None:
        print(f"  DFT dispersion loaded ({sum(len(b) for b in dft_data[1])} pieces)")
    else:
        print(f"  DFT dispersion not found at {dft_path}")
    if exp_data is not None:
        print(f"  EXP dispersion loaded ({sum(len(b) for b in exp_data[1])} pieces)")
    else:
        print(f"  EXP dispersion not found at {exp_path}")
    if dft_dos_data is not None:
        print(f"  DFT DOS loaded ({len(dft_dos_data[0])} points)")
    else:
        print(f"  DFT DOS not found at {dft_dos_path}")

    _plot_dispersion(
        band_dict, _LABELS, outdir,
        dft_data=dft_data, exp_data=exp_data,
        mtp_dos=mtp_dos, dft_dos_data=dft_dos_data,
    )

    # Write frequencies.txt for programmatic inspection
    freq_txt = outdir / "frequencies.txt"
    with open(freq_txt, "w") as f:
        f.write("# seg  pt  distance[1/Å]  " +
                "  ".join(f"freq_{i+1}[THz]" for i in range(
                    band_dict["frequencies"][0].shape[1]
                )) + "\n")
        for seg_idx, (seg_d, seg_f) in enumerate(
            zip(band_dict["distances"], band_dict["frequencies"])
        ):
            for pt_idx, (d, freqs) in enumerate(
                zip(seg_d, seg_f)
            ):
                f.write(
                    f"  {seg_idx}  {pt_idx:4d}  {d:12.6f}  "
                    + "  ".join(f"{v:12.6f}" for v in freqs)
                    + "\n"
                )
    print(f"  Frequencies written to {freq_txt}")

    all_freqs = np.concatenate([np.array(f).ravel() for f in band_dict["frequencies"]])
    print(f"\n  Frequency range: {all_freqs.min():.3f} – {all_freqs.max():.3f} THz")
    print(f"  (Ge reference: acoustic max ≈ 9 THz, optical ≈ 9–10 THz)")
    print(f"  Results in {outdir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phonon dispersion + total DOS test for MTP potential (Ge diamond)"
    )
    parser.add_argument("--pot", required=True,
                        help="Path to potential (.almtp)")
    parser.add_argument("--alat", type=float, default=5.6524,
                        help="Lattice constant in Å (default: 5.6524 Å — natural Ge at 80 K)")
    parser.add_argument("--supercell", type=int, default=6,
                        help="Primitive supercell size S; builds S×S×S supercell of the "
                             "FCC primitive cell (default: 6 → 432 atoms)")
    parser.add_argument("--displacement", type=float, default=0.01,
                        help="Phonopy displacement distance in Å (default: 0.01)")
    parser.add_argument("--npoints", type=int, default=11,
                        help="q-points per path segment (default: 11)")
    parser.add_argument("--dos-mesh", type=int, default=30,
                        help="Uniform q-mesh size for DOS (default: 30 → 30×30×30)")
    parser.add_argument("--outdir", default="results/tests/phonon_dispersion",
                        help="Output directory")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG,
                        help="Training YAML config (provides mlp_binary)")
    parser.add_argument("--mlp", default=None,
                        help="Override mlp binary path")
    parser.add_argument("--method", choices=["mtp", "lammps-mlip", "lammps-nlh"], default="mtp",
                        help="'mtp' (default): mlp calculate_efs on displaced supercells. "
                             "'lammps-mlip': run LAMMPS (pair_style mlip) per displaced "
                             "supercell, as in the ilearn workflow. "
                             "'lammps-nlh': run LAMMPS (pair_style hybrid/overlay mtp nlh) "
                             "per displaced supercell — for potentials whose radial basis "
                             "type neither of the other two methods can load.")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG,
                        help="Active-learning YAML config (provides lammps_binary; "
                             "--method lammps-mlip only)")
    parser.add_argument("--lammps", default=None,
                        help="Override LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             "(--method lammps-mlip / lammps-nlh only; lammps-nlh default: "
                             f"{DEFAULT_LAMMPS_NLH})")
    args = parser.parse_args()

    mlp = args.mlp
    if args.method == "mtp" and mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    lammps_cmd = args.lammps
    if lammps_cmd is None:
        if args.method == "lammps-mlip":
            with open(args.al_config) as f:
                lammps_cmd = yaml.safe_load(f)["lammps_binary"]
        elif args.method == "lammps-nlh":
            lammps_cmd = DEFAULT_LAMMPS_NLH

    run(
        pot=args.pot,
        outdir=Path(args.outdir),
        alat=args.alat,
        supercell_size=args.supercell,
        displacement=args.displacement,
        npoints=args.npoints,
        mlp=mlp,
        dos_mesh=args.dos_mesh,
        method=args.method,
        lammps_cmd=lammps_cmd,
    )


if __name__ == "__main__":
    main()
