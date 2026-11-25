"""Melt-quench amorphous Ge RDF via LAMMPS.

Protocol:
  1.  NVT at 300 K                 (5 ps   — equilibrate diamond)
  2.  NVT ramp 300 K → T_melt      (5 ps), then NVT hold at T_melt (20 ps — melt)
  2b. NPT at T_melt, 0 bar         (100 ps — relax box to the liquid volume)
  3.  NPT quench T_melt → 300 K at rate --quench-rate K/s, 0 bar
  4.  NPT at 300 K, 0 bar          (5 ps   — equilibrate amorphous; RDF sampled here)

Melting is checked from the mean-squared displacement accumulated over
stages 2–2b (msd.dat): a solid only vibrates (MSD ≲ 1 Å²), a liquid diffuses
(MSD ≫ 10 Å² over 125 ps). A warning is printed if the MSD stays low.

Reference a-Ge RDF: 1st peak ≈ 2.5 Å, no long-range order.

0 K energy report (after stage 4, skip with --no-energy):
  The final amorphous structure (lammps_run/amorphous.lammps) and a perfect
  diamond supercell are each relaxed at 0 K with the same potential
  (CG on positions, then `fix box/relax iso 0.0`, then positions again), and
    E_a - E_c          (eV/atom, relaxed a-Ge minus relaxed c-Ge),
    relaxed density    (g/cm^3) of the a-Ge and of c-Ge,
    mean coordination  (LAMMPS coord/atom, r < --cn-cutoff, default 3.2 Å)
                       and the fraction of 4-fold atoms
  are printed, added as `#` header lines to amorphous_rdf.txt and written to
  amorphous_energy.txt (relaxed structure: lammps_run/energy/amorphous_relaxed.lammps;
  with --from-run: <outdir>/energy_0K/).
  --from-run DIR recomputes only this report for an existing run directory
  (DIR = the amorphous_rdf output dir or its lammps_run/), without rerunning MD.
  It reads lammps_run/amorphous.lammps, or, if that is missing (run killed
  before write_data), the last frame of lammps_run/dump_amor.lammpstrj.
  Reference E_a - E_c and a-Ge density: TODO (calorimetric heat of
  crystallization of a-Ge is of order 0.1 eV/atom; exact value/source TODO).

Three pair styles (--backend), the first two as in liquid_rdf.py:
  lammps-mlip - `pair_style mlip load_from=...` (default, unchanged behaviour)
  lammps-nlh  - `pair_style hybrid/overlay mtp nlh` (mtp+nlh .mtp potentials,
                LAMMPS-MTP build; default --lammps becomes that binary)
  lammps-sw   - SW + NLH table (--pot = its directory, src/utils/lammps_pair.py);
                ~4 GB per MPI rank for the table, so use few ranks

Usage:
    # Quick test (10× faster quench, interactive):
    python src/physical_validation/amorphous_rdf.py --pot results/potentials/pot_al.almtp --quick

    # Full production run (consider submitting via SLURM for long quenches):
    python src/physical_validation/amorphous_rdf.py --pot results/potentials/pot_al.almtp \\
        --T-melt 1800 --quench-rate 1e12

    # mtp+nlh potential:
    python src/physical_validation/amorphous_rdf.py --pot results/potentials/pot_1019/bal.mtp \
        --backend lammps-nlh --quick

    # 0 K energy report only, from an existing run (no MD):
    python src/physical_validation/amorphous_rdf.py --pot results/potentials/pot_660277/pot.almtp \
        --from-run results/tests/pot_660277/amorphous_rdf \
        --lammps /projappl/project_2012355/CODE/lammps-stable_23Jun2022_update4-EPH/src/lmp_mpi
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils import compute_rdf, coordination_number, load_ref_rdf, read_lammps_dump
from energy_volume import DEFAULT_LAMMPS as DEFAULT_LAMMPS_NLH
from liquid_rdf import _PAIR_STYLE_BLOCKS
import lammps_pair

LAMMPS_DEFAULT = "lmp_mpi"
_GE_MASS_AMU = 72.630
_AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660   # 1 amu/Å^3 in g/cm^3


def _pair_block(backend: str, pot_path: str) -> str:
    """LAMMPS pair_style/pair_coeff lines for --backend."""
    if backend == "lammps-mlip":
        # Kept exactly as the original amorphous_rdf.py input (select=FALSE).
        return f"pair_style  mlip load_from={pot_path} select=FALSE\npair_coeff  * *"
    if backend == "lammps-sw":
        return lammps_pair.sw_block(pot_path).rstrip("\n")
    return _PAIR_STYLE_BLOCKS[backend](pot_path)

# Fixed stage lengths (steps, 1 fs timestep); the quench length is derived
# from --quench-rate. Shared by write_amorphous_input() and run()'s summary.
STEPS_EQ_DIAMOND = 5_000
STEPS_HEAT       = 5_000
STEPS_MELT       = 20_000
STEPS_NPT_EQ     = 100_000
STEPS_EQ_AMOR    = 5_000

# Below this MSD (Å²) at the end of the melt, the system has most likely not
# melted: a crystal at T_melt only vibrates (~0.1–1 Å²).
_MELT_MSD_MIN = 10.0


# ---------------------------------------------------------------------------
# LAMMPS input writer
# ---------------------------------------------------------------------------


def _quench_steps(T_high: float, T_low: float, rate_K_per_s: float,
                  timestep_ps: float = 0.001) -> int:
    """Compute number of MD steps for a given temperature ramp."""
    delta_T = T_high - T_low
    time_ps = delta_T / rate_K_per_s * 1e12  # seconds → picoseconds
    return max(1000, int(time_ps / timestep_ps))


def write_amorphous_input(
    workdir: Path,
    pot_path: str,
    n_cells: int = 3,
    T_melt: float = 1800.0,
    quench_rate: float = 1e12,
    steps_eq_diamond: int = STEPS_EQ_DIAMOND,
    steps_heat: int = STEPS_HEAT,
    steps_melt: int = STEPS_MELT,
    steps_npt_eq: int = STEPS_NPT_EQ,
    steps_eq_amor: int = STEPS_EQ_AMOR,
    seed: int = 99999,
    dump_every: int = 200,
    backend: str = "lammps-mlip",
) -> None:
    """Write LAMMPS input for melt-quench amorphous Ge."""
    a0 = 5.7567
    n_quench = _quench_steps(T_melt, 300.0, quench_rate)

    lammps_in = f"""# Melt-quench amorphous Ge — generated by amorphous_rdf.py
units       metal
boundary    p p p
atom_style  atomic

lattice     diamond {a0}
region      box block 0 {n_cells} 0 {n_cells} 0 {n_cells} units lattice
create_box  1 box
create_atoms 1 box
mass        1 {_GE_MASS_AMU}

{_pair_block(backend, pot_path)}

neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes
timestep    0.001
thermo      1000
thermo_style custom step time temp press vol etotal

# ── Stage 1: equilibrate diamond at 300 K ──
velocity    all create 300 {seed} mom yes rot yes
fix         nvt1 all nvt temp 300 300 0.1
run         {steps_eq_diamond}
unfix       nvt1

# ── Melt check: MSD from here to the end of stage 2b, written to msd.dat ──
compute     msd all msd
fix         msdout all ave/time 1000 1 1000 c_msd[4] file msd.dat

# ── Stage 2: heat to {T_melt:.0f} K and melt ──
fix         nvt2 all nvt temp 300 {T_melt} 0.5
run         {steps_heat}
unfix       nvt2
fix         nvt3 all nvt temp {T_melt} {T_melt} 0.1
run         {steps_melt}
unfix       nvt3

# ── Stage 2b: NPT equilibrate melt at {T_melt:.0f} K (relax box to liquid volume) ──
# Use 10 ps barostat damping (10× longer than default) to avoid runaway expansion
# when starting from high NVT pressure (~24 kbar after melt).
fix         npt_melt all npt temp {T_melt} {T_melt} 0.1 iso 0 0 10.0
run         {steps_npt_eq}
unfix       npt_melt
unfix       msdout
uncompute   msd

# ── Stage 3: NPT quench {T_melt:.0f} K → 300 K ──
fix         npt1 all npt temp {T_melt} 300 0.1 iso 0 0 10.0
run         {n_quench}
unfix       npt1

# ── Stage 4: equilibrate amorphous at 300 K, 0 bar ──
# NPT, not NVT: an NVT box would be frozen at whatever instantaneous volume the
# barostat had on the last quench step. 1 ps damping — no melt-pressure
# transient left to guard against here.
fix         npt4 all npt temp 300 300 0.1 iso 0 0 1.0
dump        dmp all custom {dump_every} dump_amor.lammpstrj id type x y z
dump_modify dmp sort id
run         {steps_eq_amor}
unfix       npt4

write_data  amorphous.lammps
"""
    (workdir / "in.amorphous").write_text(lammps_in)


# ---------------------------------------------------------------------------
# 0 K energy report: relaxed a-Ge vs relaxed c-Ge
# ---------------------------------------------------------------------------

_MINIMIZE = """min_style   cg
minimize    0 1.0e-10 20000 200000
fix         br all box/relax iso 0.0 vmax 0.001
minimize    0 1.0e-10 20000 200000
unfix       br
minimize    0 1.0e-10 20000 200000"""


def _relax_block(tag: str, cn_cutoff: float, data_out: str) -> str:
    """Relax positions + box (iso, 0 bar), then print per-atom observables.

    Lines starting with 'RESULT' are parsed by _parse_results().
    """
    return f"""
compute     cn all coord/atom cutoff {cn_cutoff}
variable    is4 atom "c_cn == 4"
compute     cnavg all reduce ave c_cn
compute     f4 all reduce ave v_is4
thermo      100
thermo_style custom step pe vol press c_cnavg c_f4
{_MINIMIZE}
run         0
variable    e_{tag} equal pe/atoms
variable    v_{tag} equal vol/atoms
variable    p_{tag} equal press
variable    cn_{tag} equal c_cnavg
variable    f4_{tag} equal c_f4
print       "RESULT {tag}_E_per_atom ${{e_{tag}}}"
print       "RESULT {tag}_V_per_atom ${{v_{tag}}}"
print       "RESULT {tag}_press_bar ${{p_{tag}}}"
print       "RESULT {tag}_CN ${{cn_{tag}}}"
print       "RESULT {tag}_frac4 ${{f4_{tag}}}"
print       "RESULT {tag}_natoms $(atoms)"
write_data  {data_out}
"""


def write_energy_input(workdir: Path, data_file: Path, pot_path: str, backend: str,
                       a0_diamond: float, n_cells_diamond: int, cn_cutoff: float) -> None:
    """LAMMPS input relaxing the amorphous structure and a perfect diamond cell."""
    pair = _pair_block(backend, pot_path)
    header = """units       metal
boundary    p p p
atom_style  atomic
"""
    neigh = """neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes"""
    text = f"""# 0 K relaxation of a-Ge and c-Ge — generated by amorphous_rdf.py
{header}
read_data   {data_file}
mass        1 {_GE_MASS_AMU}
velocity    all set 0.0 0.0 0.0
{pair}
{neigh}
{_relax_block("amor", cn_cutoff, "amorphous_relaxed.lammps")}
clear
{header}
lattice     diamond {a0_diamond}
region      box block 0 {n_cells_diamond} 0 {n_cells_diamond} 0 {n_cells_diamond} units lattice
create_box  1 box
create_atoms 1 box
mass        1 {_GE_MASS_AMU}
{pair}
{neigh}
{_relax_block("diam", cn_cutoff, "diamond_relaxed.lammps")}
"""
    (workdir / "in.energy").write_text(text)


def _parse_results(log_path: Path) -> dict[str, float]:
    res = {}
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "RESULT":
            res[parts[1]] = float(parts[2])
    return res


def _last_dump_frame_to_data(dump: Path, data_out: Path) -> None:
    """Write the last frame of a LAMMPS dump as a LAMMPS data file (atomic style)."""
    from ase.io import read, write
    atoms = read(str(dump), index=-1, format="lammps-dump-text")
    atoms.set_chemical_symbols(["Ge"] * len(atoms))
    atoms.wrap()
    write(str(data_out), atoms, format="lammps-data", specorder=["Ge"], atom_style="atomic")


def amorphous_energy(
    run_dir: Path,
    pot: str,
    lammps: str,
    backend: str,
    np_cores: int = 1,
    a0_diamond: float = 5.7567,
    n_cells_diamond: int = 3,
    cn_cutoff: float = 3.2,
    workdir: Path | None = None,
) -> dict[str, float]:
    """Relax a-Ge (from run_dir = lammps_run/) and c-Ge at 0 K; return the report dict.

    LAMMPS files go to workdir (default run_dir/energy); run_dir is only read.

    Keys: E_a_minus_E_c (eV/atom), amor_/diam_ E_per_atom, V_per_atom, press_bar, CN,
    frac4, natoms, density_g_cm3, plus 'source' (the structure file used).
    """
    data_file = run_dir / "amorphous.lammps"
    workdir = workdir if workdir is not None else run_dir / "energy"
    workdir.mkdir(parents=True, exist_ok=True)
    if not data_file.exists():
        dump = run_dir / "dump_amor.lammpstrj"
        if not dump.exists():
            raise FileNotFoundError(f"neither {data_file} nor {dump} exists")
        print(f"  {data_file.name} missing — using the last frame of {dump.name}")
        data_file = workdir / "amorphous_from_dump.lammps"
        _last_dump_frame_to_data(dump, data_file)

    write_energy_input(workdir, data_file.resolve(), str(Path(pot).resolve()), backend,
                       a0_diamond, n_cells_diamond, cn_cutoff)
    lmp_cmd = shlex.split(lammps) + ["-in", "in.energy", "-log", "energy.log"]
    if np_cores > 1:
        lmp_cmd = ["mpirun", "-np", str(np_cores)] + lmp_cmd
    result = subprocess.run(lmp_cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        print("  LAMMPS stderr (last 500 chars):")
        print(result.stderr[-500:])
        raise RuntimeError(f"LAMMPS 0 K relaxation failed (see {workdir / 'energy.log'})")

    res = _parse_results(workdir / "energy.log")
    needed = [f"{t}_{k}" for t in ("amor", "diam")
              for k in ("E_per_atom", "V_per_atom", "CN", "frac4", "natoms")]
    missing = [k for k in needed if k not in res]
    if missing:
        raise RuntimeError(f"0 K relaxation: missing {missing} in {workdir / 'energy.log'}")
    res["E_a_minus_E_c"] = res["amor_E_per_atom"] - res["diam_E_per_atom"]
    for t in ("amor", "diam"):
        res[f"{t}_density_g_cm3"] = _GE_MASS_AMU / res[f"{t}_V_per_atom"] * _AMU_PER_A3_TO_G_PER_CM3
    res["source"] = str(data_file)
    res["cn_cutoff"] = cn_cutoff
    return res


def energy_report_lines(res: dict) -> list[str]:
    """Human-readable report lines (no comment prefix)."""
    return [
        f"0 K relaxed (CG + box/relax iso 0 bar), structure: {res['source']}",
        f"E_a - E_c = {res['E_a_minus_E_c']:.4f} eV/atom  "
        f"(E_a = {res['amor_E_per_atom']:.5f}, E_c = {res['diam_E_per_atom']:.5f} eV/atom)  "
        f"[reference: TODO]",
        f"a-Ge density = {res['amor_density_g_cm3']:.4f} g/cm^3  "
        f"({1.0 / res['amor_V_per_atom']:.5f} atoms/A^3)   "
        f"c-Ge density = {res['diam_density_g_cm3']:.4f} g/cm^3  "
        f"(a0 = {(8.0 * res['diam_V_per_atom']) ** (1 / 3):.4f} A)  [a-Ge reference: TODO]",
        f"a-Ge mean coordination (r < {res['cn_cutoff']:.2f} A) = {res['amor_CN']:.3f}   "
        f"4-fold fraction = {res['amor_frac4']:.3f}   "
        f"(c-Ge: {res['diam_CN']:.3f}, {res['diam_frac4']:.3f})",
        f"N_atoms a-Ge = {int(res['amor_natoms'])}, c-Ge = {int(res['diam_natoms'])}; "
        f"residual pressure a-Ge = {res.get('amor_press_bar', float('nan')):.1f} bar",
    ]


def write_energy_report(path: Path, res: dict, pot: str) -> None:
    with open(path, "w") as f:
        f.write(f"# Amorphous Ge 0 K energy  |  pot: {pot}\n")
        for line in energy_report_lines(res):
            f.write(f"# {line}\n")
        f.write("# key  value\n")
        for k, v in res.items():
            if k == "source":
                continue
            f.write(f"{k}  {v:.8g}\n")


def run_energy_only(pot: str, from_run: Path, outdir: Path | None, lammps: str, backend: str,
                    np_cores: int, a0_diamond: float, n_cells_diamond: int,
                    cn_cutoff: float) -> dict:
    """--from-run: 0 K energy report for an existing run, no MD."""
    run_dir = from_run / "lammps_run" if (from_run / "lammps_run").is_dir() else from_run
    outdir = outdir if outdir is not None else run_dir.parent
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  0 K energy report from existing run {run_dir}")
    res = amorphous_energy(run_dir, pot, lammps, backend, np_cores=np_cores,
                           a0_diamond=a0_diamond, n_cells_diamond=n_cells_diamond,
                           cn_cutoff=cn_cutoff, workdir=outdir / "energy_0K")
    for line in energy_report_lines(res):
        print(f"  {line}")
    txt = outdir / "amorphous_energy.txt"
    write_energy_report(txt, res, pot)
    print(f"  Energy report written to {txt}")
    return res


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run(
    pot: str,
    outdir: Path,
    lammps: str,
    np_cores: int,
    n_cells: int,
    T_melt: float,
    quench_rate: float,
    rmax: float,
    nbins: int,
    ref_highbound: str | None = None,
    ref_lowbound: str | None = None,
    backend: str = "lammps-mlip",
    energy: bool = True,
    a0_diamond: float = 5.7567,
    n_cells_diamond: int = 3,
    cn_cutoff: float = 3.2,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    n_atoms = 8 * n_cells ** 3
    n_quench = _quench_steps(T_melt, 300.0, quench_rate)
    total_steps = (STEPS_EQ_DIAMOND + STEPS_HEAT + STEPS_MELT + STEPS_NPT_EQ
                   + n_quench + STEPS_EQ_AMOR)

    print(f"  Amorphous Ge  |  {n_atoms} atoms  |  T_melt = {T_melt:.0f} K")
    print(f"  Quench rate = {quench_rate:.0e} K/s  →  {n_quench:,} quench steps")
    print(f"  Total steps ≈ {total_steps:,}  (~{total_steps * 0.001 / 1000:.1f} ns)")

    if n_quench > 500_000 and sys.stdout.isatty():
        print("\n  *** NOTE: This is a long run (>500 k steps). ***")
        print("  Consider submitting via SLURM.")
        print("  Or use --quick for a 10× faster quench (10^13 K/s).\n")

    workdir = outdir / "lammps_run"
    workdir.mkdir(exist_ok=True)

    pot_abs = str(Path(pot).resolve())
    write_amorphous_input(
        workdir, pot_abs,
        n_cells=n_cells,
        T_melt=T_melt,
        quench_rate=quench_rate,
        backend=backend,
    )

    print("  Running LAMMPS ...")
    # shlex.split: --lammps may be a launcher + binary, e.g. "srun /path/lmp_mpi".
    lmp_cmd = shlex.split(lammps) + ["-in", "in.amorphous", "-log", "amorphous.log"]
    if np_cores > 1:
        lmp_cmd = ["mpirun", "-np", str(np_cores)] + lmp_cmd
    result = subprocess.run(lmp_cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        print("  LAMMPS stderr (last 500 chars):")
        print(result.stderr[-500:])
        raise RuntimeError("LAMMPS run failed")

    # Melt check: last MSD value written over stages 2–2b.
    msd_file = workdir / "msd.dat"
    msd_melt = None
    if msd_file.exists():
        rows = np.loadtxt(msd_file, comments="#", ndmin=2)
        if rows.size:
            msd_melt = float(rows[-1, 1])
    if msd_melt is None:
        print("  WARNING: msd.dat missing or empty — could not verify that the system melted.")
    elif msd_melt < _MELT_MSD_MIN:
        print(f"  WARNING: MSD at end of melt = {msd_melt:.2f} Å² (< {_MELT_MSD_MIN:.0f} Å²) — "
              f"the system most likely did NOT melt at T_melt = {T_melt:.0f} K. "
              f"The 'amorphous' structure is probably still crystalline; raise --T-melt.")
    else:
        print(f"  Melt check: MSD at end of melt = {msd_melt:.1f} Å² (melted)")

    dump = workdir / "dump_amor.lammpstrj"
    if not dump.exists():
        raise RuntimeError(f"LAMMPS dump not found at {dump}")

    print("  Reading dump and computing RDF ...")
    frames = read_lammps_dump(dump, n_atoms)
    if not frames:
        raise RuntimeError("No frames read from dump file")

    skip_frames = 10
    frames_to_use = frames[skip_frames:] if len(frames) > skip_frames else frames
    print(f"  {len(frames)} frames available; using {len(frames_to_use)} for RDF averaging "
          f"(skipping first {skip_frames})")

    g_total = np.zeros(nbins)
    r = None
    for cell_frame, pos in frames_to_use:
        r, g = compute_rdf(pos, cell_frame, rmax=rmax, nbins=nbins)
        g_total += g
    g_r = g_total / len(frames_to_use)

    i_peak = int(np.argmax(g_r[r < 3.0]))
    r_peak = r[i_peak]

    # Box fluctuates under NPT in stage 4: use the mean volume over the frames.
    mean_vol = np.mean([abs(np.linalg.det(c)) for c, _ in frames_to_use])
    rho = n_atoms / mean_vol
    cn = coordination_number(r, g_r, rho, r_cutoff=3.2)

    print(f"  1st peak at r = {r_peak:.3f} Å  [expected ≈ 2.5 Å for a-Ge]")
    print(f"  Coordination number (r < 3.2 Å): {cn:.2f}  [expected ≈ 4 for a-Ge]")

    energy_lines: list[str] = []
    if energy:
        print("  0 K relaxation of a-Ge and c-Ge ...")
        try:
            res = amorphous_energy(workdir, pot, lammps, backend, np_cores=np_cores,
                                   a0_diamond=a0_diamond, n_cells_diamond=n_cells_diamond,
                                   cn_cutoff=cn_cutoff)
            energy_lines = energy_report_lines(res)
            for line in energy_lines:
                print(f"  {line}")
            write_energy_report(outdir / "amorphous_energy.txt", res, pot)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  WARNING: 0 K energy report failed: {e}")

    txt = outdir / "amorphous_rdf.txt"
    with open(txt, "w") as f:
        f.write(f"# Amorphous Ge RDF  |  pot: {pot}\n")
        f.write(f"# T_melt = {T_melt:.0f} K  quench_rate = {quench_rate:.0e} K/s\n")
        f.write(f"# 1st peak at r = {r_peak:.3f} Å   CN = {cn:.2f}\n")
        for line in energy_lines:
            f.write(f"# {line}\n")
        f.write("# r[A]  g(r)\n")
        for ri, gi in zip(r, g_r):
            f.write(f"  {ri:.4f}  {gi:.6f}\n")
    print(f"  Data written to {txt}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        # potential's result
        ax.plot(r, g_r, color="tab:purple", lw=1.5, label="SW" if backend == "lammps-sw" else "MTP")
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")

        # Reference data overlays
        ref_sets = [
            ("Exp. high", "tab:orange", 1.0, ref_highbound),
            ("Exp. low",  "tab:green",  1.0, ref_lowbound),
        ]
        for label, color, alpha, fpath in ref_sets:
            rr, gg = load_ref_rdf(fpath)
            if rr is not None:
                ax.plot(rr, gg, color=color, lw=1.0, ls="--", alpha=alpha, label=label)

        ax.axvline(r_peak, color="tab:red", lw=0.8, ls=":", label=f"1st peak: {r_peak:.2f} Å")
        ax.set_xlabel("r (Å)", fontsize=12)
        ax.set_ylabel("g(r)", fontsize=12)
        ax.set_title(
            f"Amorphous Ge g(r)  [quench {quench_rate:.0e} K/s]", fontsize=13
        )
        ax.legend(fontsize=9)
        ax.set_xlim(0, rmax)
        fig.tight_layout()
        png = outdir / "amorphous_rdf.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png}")
    except ImportError as e:
        print(f"  Matplotlib not available ({e}); skipping plot.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melt-quench amorphous Ge RDF via LAMMPS"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.almtp)")
    parser.add_argument("--T-melt", type=float, default=1800.0,
                        help="Melt temperature in K (default: 1800; Ge Tm = 1211 K, and a "
                             "surface-free crystal must be superheated well above Tm to melt "
                             "within ~20 ps — checked afterwards via msd.dat)")
    parser.add_argument("--quench-rate", type=float, default=1e12,
                        help="Quench rate in K/s (default: 1e12; use 1e13 for --quick)")
    parser.add_argument("--quick", action="store_true",
                        help="Use 1e13 K/s quench rate for a quick validation run")
    parser.add_argument("--n-cells", type=int, default=3,
                        help="Supercell repeat (8·n³ atoms, default: 3→216)")
    parser.add_argument("--rmax", type=float, default=8.0)
    parser.add_argument("--nbins", type=int, default=200)
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/tests/<pot's parent dir "
                             "name>/amorphous_rdf, e.g. --pot results/potentials/pot_660277/pot.almtp "
                             "-> results/tests/pot_660277/amorphous_rdf)")
    parser.add_argument("--lammps", default=None,
                        help="LAMMPS command (default: lmp_mpi for --backend lammps-mlip, "
                             f"{DEFAULT_LAMMPS_NLH} for --backend lammps-nlh/lammps-sw)")
    parser.add_argument("--backend", choices=["lammps-mlip", "lammps-nlh", "lammps-sw"], default="lammps-mlip",
                        help="'lammps-mlip' (default): pair_style mlip. 'lammps-nlh': pair_style "
                             "hybrid/overlay mtp nlh, for mtp+nlh .mtp potentials. 'lammps-sw': SW + "
                             "NLH table, --pot = its directory (src/utils/lammps_pair.py).")
    parser.add_argument("--no-energy", action="store_true",
                        help="Skip the 0 K E_a - E_c / density / coordination report")
    parser.add_argument("--from-run", default=None,
                        help="Only compute the 0 K energy report for an existing run directory "
                             "(amorphous_rdf outdir or its lammps_run/); no MD. Output goes to "
                             "--outdir if given, else next to lammps_run/.")
    parser.add_argument("--a0-diamond", type=float, default=5.7567,
                        help="Starting lattice constant of the c-Ge reference cell (Å); the box "
                             "is relaxed anyway (default: 5.7567)")
    parser.add_argument("--n-cells-diamond", type=int, default=3,
                        help="c-Ge reference supercell repeat (default: 3 -> 216 atoms)")
    parser.add_argument("--cn-cutoff", type=float, default=3.2,
                        help="Coordination cutoff for the 0 K report (default: 3.2 Å)")
    parser.add_argument("--np", type=int, default=1,
                        help="Number of MPI processes (default: 1, wraps with mpirun if >1)")
    parser.add_argument("--ref-highbound", default="src/physical_validation/reference/thermal_properties_EXP/a-Ge_rdf_highbound.dat",
                        help="Reference experimental high-bound RDF .dat file")
    parser.add_argument("--ref-lowbound", default="src/physical_validation/reference/thermal_properties_EXP/a-Ge_rdf_lowbound.dat",
                        help="Reference experimental low-bound RDF .dat file")
    args = parser.parse_args()
    if args.backend == "lammps-sw":
        lammps_pair.check_sw(args.pot)
    lammps = args.lammps or (LAMMPS_DEFAULT if args.backend == "lammps-mlip" else DEFAULT_LAMMPS_NLH)

    if args.from_run is not None:
        run_energy_only(
            pot=args.pot, from_run=Path(args.from_run),
            outdir=Path(args.outdir) if args.outdir else None,
            lammps=lammps, backend=args.backend, np_cores=args.np,
            a0_diamond=args.a0_diamond, n_cells_diamond=args.n_cells_diamond,
            cn_cutoff=args.cn_cutoff,
        )
        return

    if args.quick and args.quench_rate != 1e12:
        print("  NOTE: --quick overrides --quench-rate; using 1e13 K/s")
    quench_rate = 1e13 if args.quick else args.quench_rate

    if args.outdir is None:
        run_name = Path(args.pot).resolve().parent.name
        outdir = Path("results/tests") / run_name / "amorphous_rdf"
    else:
        outdir = Path(args.outdir)

    run(
        pot=args.pot,
        outdir=outdir,
        lammps=lammps,
        np_cores=args.np,
        n_cells=args.n_cells,
        T_melt=args.T_melt,
        quench_rate=quench_rate,
        rmax=args.rmax,
        nbins=args.nbins,
        ref_highbound=args.ref_highbound,
        ref_lowbound=args.ref_lowbound,
        backend=args.backend,
        energy=not args.no_energy,
        a0_diamond=args.a0_diamond,
        n_cells_diamond=args.n_cells_diamond,
        cn_cutoff=args.cn_cutoff,
    )


if __name__ == "__main__":
    main()
