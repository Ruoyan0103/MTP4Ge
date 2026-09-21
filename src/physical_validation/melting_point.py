"""
Melting point of diamond-cubic Ge via two-phase coexistence method.

Protocol, a0 = 5.7567 A (MTP equilibrated):
  1. NPT equilibration: 500 K, 1 bar, 20 ps, dt = 2 fs
  2. Selective melting: bottom half fixed, top half NVT at 1500 K, 20 ps
  3. NpH coexistence: 1 bar, 200 ps — temperature evolves freely
  4. Tm = mean T over last 100 ps of coexistence

Repeated N times with independent random seeds to obtain mean Tm ± σ.

Two LAMMPS runs per repeat:
  Phase 1: NPT equilibration → extract z-midpoint from log
  Phase 2: melt + NpH coexistence with hardcoded z-midpoint

Usage (local, sequential over all repeats):
    python src/physical_validation/melting_point.py --pot results/potentials/pot_al.almtp
    python src/physical_validation/melting_point.py --pot results/potentials/pot_al.almtp \
        --steps-coexist 100000 --outdir results/tests/melting_point
    python src/physical_validation/melting_point.py --pot results/potentials/pot_al.almtp --quick

Heavy test — submit to HPC as a SLURM array (one repeat per array task):
    sbatch scripts/submit_melting_point.sh results/potentials/pot_al.almtp
    # after all array tasks finish, collect Tm across repeats:
    python src/physical_validation/melting_point.py --pot results/potentials/pot_al.almtp --aggregate
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_GE_MASS_AMU = 72.630

# ---------------------------------------------------------------------------
# Expected Tm for sanity check
# ---------------------------------------------------------------------------
GE_TM_EXP = 1211.4   # K, experimental melting point of germanium


# ---------------------------------------------------------------------------
# LAMMPS input writers (2-phase approach)
# ---------------------------------------------------------------------------

def write_stage1_input(
    workdir: Path,
    pot_path: str,
    nx: int,
    ny: int,
    nz: int,
    T_solid: float,
    steps_npt: int,
    seed: int,
    thermo_every: int,
    a0_diamond: float,
    dump_every: int,
) -> Path:
    """Write LAMMPS input for Stage 1: NPT equilibration only."""
    lammps_in = f"""# Melting point Stage 1 — NPT equilibration
# Supercell: {nx}x{ny}x{nz} = {8*nx*ny*nz} atoms

units       metal
boundary    p p p
atom_style  atomic

lattice     diamond {a0_diamond}
region      box block 0 {nx} 0 {ny} 0 {nz} units lattice
create_box  1 box
create_atoms 1 box
mass        1 {_GE_MASS_AMU}

pair_style  hybrid/overlay mtp nlh
pair_coeff  * * mtp {pot_path} 0
pair_coeff  1 1 nlh 32 32 0.05121 0.50355 0.44524 49.82156 12.56998 4.13684 1.28 2.0

neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes
timestep    0.002
thermo      {thermo_every}
thermo_style custom step temp press vol lx ly lz pe ke etotal

#---------------------------MINIMIZE------------------------------
min_style      cg
minimize       1.0e-8 1.0e-10 10000 100000
fix            1 all box/relax iso 0.0 vmax 0.001
minimize       1e-10 1e-10 10000 100000
unfix          1
run            1000

# Stage 1: NPT at {T_solid} K, 0 bar, {steps_npt * 0.002:.0f} ps
velocity    all create {T_solid} {seed} mom yes rot yes
fix         npt_eq all npt temp {T_solid} {T_solid} $(100.0*dt) iso 0.0 0.0 $(1000.0*dt)
dump        dmp1 all custom {dump_every} stage1_npt.lammpstrj id type x y z
dump_modify dmp1 sort id
run         {steps_npt}
undump      dmp1
unfix       npt_eq

write_restart stage1.restart
"""
    in_path = workdir / "in.stage1"
    in_path.write_text(lammps_in)
    return in_path


def write_stage23_input(
    workdir: Path,
    pot_path: str,
    z_mid: float,
    T_solid: float,
    T_melt: float,
    steps_melt: int,
    steps_coexist: int,
    thermo_every: int,
    dump_every: int,
) -> Path:
    """Write LAMMPS input for Stage 2 (melt) + Stage 3 (NpH coexist).

    z_mid is the box z-midpoint from Stage 1, hardcoded for region.
    Lower half is thermostated at T_solid (NVT) so Stage 3 starts near Tm.
    """
    lammps_in = f"""# Melting point Stage 2+3 — selective melt + NpH coexistence
# z_mid = {z_mid:.6f} A (from Stage 1 NPT box)

read_restart stage1.restart

pair_style  hybrid/overlay mtp nlh
pair_coeff  * * mtp {pot_path} 0
pair_coeff  1 1 nlh 32 32 0.05121 0.50355 0.44524 49.82156 12.56998 4.13684 1.28 2.0

neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes
timestep    0.002
thermo      {thermo_every}
thermo_style custom step temp press vol pe ke etotal

# ================================================================
# Stage 2: Selective melting — bottom half fixed, top half melted
# ================================================================
region      lower block INF INF INF INF INF {z_mid:.6f} units box
region      upper block INF INF INF INF {z_mid:.6f} INF units box
group       lower region lower
group       upper region upper

# Thermostat bottom half at {T_solid} K (solid stays ordered; Stage 3 starts near Tm)
fix         solid_thermo lower nvt temp {T_solid} {T_solid} $(100.0*dt) 

# Thermostat top half to {T_melt} K
fix         melt_thermo upper nvt temp {T_melt} {T_melt} $(100.0*dt) 
dump        dmp2 all custom {dump_every} stage2_melt.lammpstrj id type x y z
dump_modify dmp2 sort id
dump        final2 all custom {steps_melt} stage2_final.lammpstrj id type x y z
dump_modify final2 sort id
run         {steps_melt}
undump      dmp2
undump      final2
unfix       solid_thermo
unfix       melt_thermo

group       lower delete
group       upper delete
region      lower delete
region      upper delete

# ================================================================
# Stage 3: NpH coexistence — T evolves freely
# ================================================================
reset_timestep 0
fix         nph_coexist all nph iso 0.0 0.0 $(1000.0*dt) 
dump        dmp3 all custom {dump_every} stage3_coexist.lammpstrj id type x y z
dump_modify dmp3 sort id
run         {steps_coexist}
undump      dmp3
unfix       nph_coexist

write_data  final.data
print       "=== Coexistence run complete ==="
"""
    in_path = workdir / "in.stage23"
    in_path.write_text(lammps_in)
    return in_path



def _is_thermo_line(parts):
    """Check if a line looks like LAMMPS thermo data (all numeric, step is int)."""
    try:
        int(parts[0])
        float(parts[1])
        float(parts[2])
        return True
    except (ValueError, IndexError):
        return False

def get_zmid_from_log(log_path: Path) -> float:
    """Extract z-midpoint = lz/2 from last thermo line of Stage 1 log.

    Assumes thermo_style includes lx, ly, lz and zlo=0 for periodic box.
    """
    text = log_path.read_text()
    lines = text.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        # Columns: step temp press vol lx ly lz pe ke etotal
        if len(parts) >= 7 and _is_thermo_line(parts):
            try:
                lz = float(parts[6])
                return lz / 2.0
            except (ValueError, IndexError):
                continue
    raise RuntimeError(f"Could not extract lz from {log_path}")


# ---------------------------------------------------------------------------
# LAMMPS log parser
# ---------------------------------------------------------------------------

def parse_lammps_log(log_path: Path) -> dict[str, np.ndarray]:
    """Parse LAMMPS log file → dict of thermo arrays.

    Returns dict with keys: step, time, temp, press, vol, pe, ke, etotal.
    Each value is a 1D numpy array.
    """
    text = log_path.read_text()

    # Find the last "Step" header line (there may be multiple runs)
    lines = text.splitlines()
    header_idx = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith("Step ") and "Temp" in stripped:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"No thermo header found in {log_path}")

    header = lines[header_idx].split()
    col_map = {name: idx for idx, name in enumerate(header)}

    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != len(header):
            break
        data_lines.append([float(x) for x in parts])

    if not data_lines:
        raise RuntimeError(f"No thermo data found after header in {log_path}")

    data = np.array(data_lines)

    result_out = {}
    key_map = {
        "Step": "step", "Temp": "temp", "Press": "press",
        "Volume": "vol", "PotEng": "pe", "KinEng": "ke", "TotEng": "etotal",
    }
    for header_name, short_name in key_map.items():
        if header_name in col_map:
            result_out[short_name] = data[:, col_map[header_name]]
    result_out["time"] = data[:, col_map["Step"]] * 0.002
    return result_out


def _coord_num(pos: np.ndarray, cell: np.ndarray, r_cut: float = 3.5) -> float:
    """Compute mean coordination number for a set of atoms."""
    cell_inv = np.linalg.inv(cell)
    N = len(pos)
    if N < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    frac = diff @ cell_inv
    frac -= np.round(frac)
    diff_mic = frac @ cell
    dists = np.linalg.norm(diff_mic, axis=-1)
    np.fill_diagonal(dists, r_cut + 1.0)
    neighbours = (dists < r_cut).sum(axis=1)
    return float(np.mean(neighbours))


def verify_melting(stage2_dump: Path, n_atoms: int) -> tuple[bool, float, float]:
    """Check that the top half actually melted during Stage 2.

    Returns (melted, cn_upper, cn_lower).
      melted = True if cn_upper > 5.5 (liquid-like; diamond Ge CN=4)
    """
    if not stage2_dump.exists():
        return False, 0.0, 0.0

    text = stage2_dump.read_text()
    blocks = text.split("ITEM: TIMESTEP")
    if len(blocks) < 2:
        return False, 0.0, 0.0

    last_block = blocks[-1]
    lines = last_block.splitlines()

    try:
        box_idx = next(i for i, l in enumerate(lines) if "ITEM: BOX BOUNDS" in l)
        lx_lo, lx_hi = map(float, lines[box_idx + 1].split()[:2])
        ly_lo, ly_hi = map(float, lines[box_idx + 2].split()[:2])
        lz_lo, lz_hi = map(float, lines[box_idx + 3].split()[:2])
        cell = np.diag([lx_hi - lx_lo, ly_hi - ly_lo, lz_hi - lz_lo])
        z_mid = (lz_lo + lz_hi) / 2.0
    except (StopIteration, IndexError, ValueError):
        return False, 0.0, 0.0

    try:
        atom_idx = next(i for i, l in enumerate(lines) if "ITEM: ATOMS" in l)
    except StopIteration:
        return False, 0.0, 0.0

    pos_upper, pos_lower = [], []
    for line in lines[atom_idx + 1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        if z > z_mid:
            pos_upper.append([x, y, z])
        else:
            pos_lower.append([x, y, z])
        if len(pos_upper) + len(pos_lower) >= n_atoms:
            break

    pos_upper = np.array(pos_upper) if pos_upper else np.empty((0, 3))
    pos_lower = np.array(pos_lower) if pos_lower else np.empty((0, 3))

    cn_upper = _coord_num(pos_upper, cell) if len(pos_upper) > 1 else 0.0
    cn_lower = _coord_num(pos_lower, cell) if len(pos_lower) > 1 else 0.0

    melted = cn_upper > 5.5 and cn_lower < 5.0
    return melted, cn_upper, cn_lower


def extract_tm_from_log(log_path: Path, last_ps: float = 10.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract melting temperature from coexistence stage.

    Uses the last `last_ps` picoseconds of the thermo data.
    Returns (Tm, temps, times_ps).
    """
    thermo = parse_lammps_log(log_path)
    temps = thermo["temp"]
    times = thermo["time"]

    t_end = times[-1]
    mask = times >= (t_end - last_ps)

    tm = float(np.mean(temps[mask]))
    return tm, temps[mask], times[mask]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    all_results: list[dict],
    outdir: Path,
    discard_fraction: float = 0.5,
) -> None:
    """Plot T(t) for all repeats and Tm histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plt.rcParams['font.family'] = 'DejaVu Serif'
    n_repeats = len(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_repeats))
    for i, res in enumerate(all_results):
        times = res["times"]
        temps = res["temps"]
        tm = res["tm"]
        n = len(temps)
        disc = int(n * discard_fraction)
        ax.plot(times, temps, color=colors[i], lw=0.5, alpha=0.6)
        ax.plot(times[disc:], temps[disc:], color=colors[i], lw=1.2, alpha=0.9)
        ax.axhline(tm, color=colors[i], lw=0.8, ls=":", alpha=0.5)

    ax.axhline(GE_TM_EXP, color="black", lw=1.5, ls="--", label=f"Exp. Tm = {GE_TM_EXP} K")
    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("Temperature (K)", fontsize=11)
    ax.set_title(f"Two-Phase Coexistence T(t) — {n_repeats} repeats", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)

    ax = axes[1]
    tm_values = [r["tm"] for r in all_results]
    mean_tm = np.mean(tm_values)
    std_tm = np.std(tm_values, ddof=1) if n_repeats > 1 else 0.0
    ax.hist(tm_values, bins=max(6, n_repeats // 3), edgecolor="black",
            color="tab:blue", alpha=0.7, density=False)
    ax.axvline(mean_tm, color="tab:red", lw=2, ls="-",
               label=f"MTP Tm = {mean_tm:.1f} ± {std_tm:.1f} K")
    ax.axvline(GE_TM_EXP, color="black", lw=1.5, ls="--",
               label=f"Exp. = {GE_TM_EXP} K")
    ax.set_xlabel("Tm (K)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Melting Point Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)

    fig.tight_layout()
    png = outdir / "melting_point.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  Plot saved to {png}")


# ---------------------------------------------------------------------------
# Single repeat (2-phase LAMMPS)
# ---------------------------------------------------------------------------

def _run_lammps(lammps: str, np_cores: int, workdir: Path,
                in_name: str, log_name: str, seed: int) -> float:
    """Run LAMMPS; return elapsed wall time in seconds. Raises on failure."""
    lmp_tokens = lammps.split()
    if lmp_tokens and lmp_tokens[0] == "srun":
        lmp_cmd = ([lmp_tokens[0], "-n", str(np_cores)]
                   + lmp_tokens[1:] + ["-in", in_name, "-log", log_name])
    elif np_cores > 1:
        lmp_cmd = ["mpirun", "-np", str(np_cores)] + lmp_tokens + ["-in", in_name, "-log", log_name]
    else:
        lmp_cmd = lmp_tokens + ["-in", in_name, "-log", log_name]
    t0 = time.time()
    result = subprocess.run(lmp_cmd, cwd=workdir, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    LAMMPS stderr (last 600 chars):")
        print(result.stderr[-600:])
        raise RuntimeError(f"LAMMPS run failed with seed={seed}")
    return elapsed


def collect_repeat_result(
    workdir: Path,
    seed: int,
    n_atoms: int,
    verify_melting_cn: bool = True,
) -> dict:
    """Extract Tm (+ melting verification) from an already-completed repeat's outputs.

    Reads stage23.log (and stage2_final.lammpstrj, if present) from `workdir`.
    Shared by `run_single_repeat` (right after LAMMPS finishes) and `--aggregate`
    (reading logs left behind by earlier SLURM array tasks).
    """
    melted, cn_upper, cn_lower = False, 0.0, 0.0
    if verify_melting_cn:
        stage2_dump = workdir / "stage2_final.lammpstrj"
        melted, cn_upper, cn_lower = verify_melting(stage2_dump, n_atoms)
        if not melted:
            if cn_upper <= 5.5:
                print(f"    WARNING: Top half did not melt! CN_upper = {cn_upper:.1f} "
                      f"(expected > 5.5 for liquid). Try increasing --T-melt or --steps-melt.")
            if cn_lower >= 5.0:
                print(f"    WARNING: Bottom half is not solid! CN_lower = {cn_lower:.1f} "
                      f"(expected < 5.0 for solid). Try decreasing --T-solid.")
        else:
            print(f"    Melting verified: CN_upper = {cn_upper:.1f} (liquid), "
                  f"CN_lower = {cn_lower:.1f} (solid)")

    tm, temps, times = extract_tm_from_log(workdir / "stage23.log")
    return {"seed": seed, "tm": tm, "temps": temps, "times": times,
            "melted": melted, "cn_upper": cn_upper, "cn_lower": cn_lower}


def run_single_repeat(
    lammps: str,
    np_cores: int,
    pot_abs: str,
    workdir: Path,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    T_solid: float,
    T_melt: float,
    steps_npt: int,
    steps_melt: int,
    steps_coexist: int,
    a0: float,
    dump_every: int = 1_000,
    verify_melting_cn: bool = True,
) -> dict:
    """Run one coexistence simulation (2 LAMMPS phases).

    Phase 1: NPT equilibration → extract z_midpoint
    Phase 2: melt + NpH coexistence with hardcoded z_midpoint
    """
    # --- Phase 1: NPT equilibration ---
    write_stage1_input(
        workdir=workdir, pot_path=pot_abs,
        nx=nx, ny=ny, nz=nz, T_solid=T_solid,
        steps_npt=steps_npt, seed=seed,
        thermo_every=100, a0_diamond=a0, dump_every=dump_every,
    )
    t1 = _run_lammps(lammps, np_cores, workdir, "in.stage1", "stage1.log", seed)
    print(f"    Phase 1 (NPT) finished in {t1:.1f}s (seed={seed})")

    z_mid = get_zmid_from_log(workdir / "stage1.log")
    print(f"    Box z-midpoint = {z_mid:.3f} A")

    # --- Phase 2: Melt + Coexistence ---
    write_stage23_input(
        workdir=workdir, pot_path=pot_abs, z_mid=z_mid,
        T_solid=T_solid, T_melt=T_melt, steps_melt=steps_melt,
        steps_coexist=steps_coexist,
        thermo_every=100, dump_every=dump_every,
    )
    t2 = _run_lammps(lammps, np_cores, workdir, "in.stage23", "stage23.log", seed)
    print(f"    Phase 2 (melt+coexist) finished in {t2:.1f}s (seed={seed})")

    n_atoms = 8 * nx * ny * nz
    return collect_repeat_result(workdir, seed, n_atoms, verify_melting_cn)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def _summarize(all_results: list[dict], outdir: Path, pot: str | None, nx: int, ny: int, nz: int) -> None:
    """Aggregate Tm across repeats: print stats and write tm_summary.txt."""
    n_atoms = 8 * nx * ny * nz

    if not all_results:
        print("\n  Error: no completed repeats found.", file=sys.stderr)
        sys.exit(1)

    tm_values = np.array([r["tm"] for r in all_results])
    mean_tm = float(np.mean(tm_values))
    std_tm = float(np.std(tm_values, ddof=1)) if len(tm_values) > 1 else 0.0
    sem_tm = std_tm / np.sqrt(len(tm_values)) if len(tm_values) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  Melting point of diamond-cubic Ge (MTP)")
    print(f"  Tm = {mean_tm:.2f} +- {std_tm:.2f} K  (s, n={len(tm_values)})")
    print(f"  SEM = {sem_tm:.2f} K")
    print(f"  Exp. Tm (Ge) = {GE_TM_EXP} K")
    print(f"  dT = {mean_tm - GE_TM_EXP:+.2f} K  "
          f"({(mean_tm - GE_TM_EXP) / GE_TM_EXP * 100:+.2f}%)")
    print(f"{'='*60}")

    summary_path = outdir / "tm_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"# Melting point: two-phase coexistence method\n")
        f.write(f"# Potential: {pot if pot else 'unknown'}\n")
        f.write(f"# Supercell: {nx}x{ny}x{nz} ({n_atoms} atoms)\n")
        f.write(f"# N repeats: {len(tm_values)}\n")
        f.write(f"# Tm = {mean_tm:.4f} +- {std_tm:.4f} K (s)\n")
        f.write(f"# SEM = {sem_tm:.4f} K\n")
        f.write(f"# Exp. Tm (Ge) = {GE_TM_EXP} K\n")
        f.write(f"# dT = {mean_tm - GE_TM_EXP:+.4f} K\n")
        f.write(f"#\n")
        f.write(f"# seed  Tm (K)\n")
        for r in all_results:
            f.write(f"  {r['seed']:6d}  {r['tm']:10.4f}\n")
    print(f"  Summary written to {summary_path}")

    #plot_results(all_results, outdir)


def _aggregate(
    outdir: Path,
    pot: str | None,
    n_repeats: int,
    seed_start: int,
    nx: int,
    ny: int,
    nz: int,
    verify_melting_cn: bool,
) -> None:
    """Collect Tm from repeat_NN/ directories left behind by SLURM array tasks."""
    n_atoms = 8 * nx * ny * nz
    results_by_index: dict[int, dict] = {}
    for i in range(n_repeats):
        run_dir = outdir / f"repeat_{i:02d}"
        seed = seed_start + i
        if not (run_dir / "stage23.log").exists():
            print(f"  Repeat {i:02d} (seed={seed}): no stage23.log found — skipping")
            continue
        try:
            res = collect_repeat_result(run_dir, seed, n_atoms, verify_melting_cn)
            results_by_index[i] = res
            print(f"  Repeat {i:02d} (seed={seed}): Tm = {res['tm']:.2f} K "
                  f"(exp = {GE_TM_EXP} K, dT = {res['tm'] - GE_TM_EXP:+.1f} K)")
        except Exception as e:
            print(f"  Repeat {i:02d} (seed={seed}): FAILED to parse ({e})")

    all_results = [results_by_index[i] for i in sorted(results_by_index)]
    _summarize(all_results, outdir, pot, nx, ny, nz)


def run(
    pot: str,
    outdir: Path,
    lammps: str,
    np_cores: int,
    n_repeats: int,
    nx: int,
    ny: int,
    nz: int,
    T_solid: float,
    T_melt: float,
    steps_npt: int,
    steps_melt: int,
    steps_coexist: int,
    a0: float,
    seed_start: int,
    dump_every: int = 1_000,
    verify_melting_cn: bool = True,
    repeat_index: int | None = None,
    aggregate: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if aggregate:
        print(f"=== Melting Point: aggregating {n_repeats} repeats from {outdir} ===")
        _aggregate(outdir, pot, n_repeats, seed_start, nx, ny, nz, verify_melting_cn)
        return

    n_atoms = 8 * nx * ny * nz
    pot_abs = str(Path(pot).resolve())

    total_ps_npt = steps_npt * 0.002
    total_ps = (steps_npt + steps_melt + steps_coexist) * 0.002

    print(f"=== Melting Point: Two-Phase Coexistence ===")
    print(f"  Supercell: {nx}x{ny}x{nz} = {n_atoms} atoms")
    print(f"  Potential: {pot}")
    print(f"  Protocol: NPT {total_ps_npt:.0f} ps + melt {steps_melt*0.002:.0f} ps "
          f"+ NpH {steps_coexist*0.002:.0f} ps = {total_ps:.0f} ps")
    print(f"  Repeats: {n_repeats}  |  Seed start: {seed_start}")
    print(f"  LAMMPS: {lammps}  |  MPI tasks/repeat: {np_cores}")
    if repeat_index is not None:
        print(f"  Running single repeat index {repeat_index} (SLURM array task)")

    indices = [repeat_index] if repeat_index is not None else list(range(n_repeats))

    results_by_index: dict[int, dict] = {}
    for i in indices:
        seed = seed_start + i
        run_dir = outdir / f"repeat_{i:02d}"
        run_dir.mkdir(exist_ok=True)
        try:
            res = run_single_repeat(
                lammps=lammps,
                np_cores=np_cores,
                pot_abs=pot_abs,
                workdir=run_dir,
                seed=seed,
                nx=nx, ny=ny, nz=nz,
                T_solid=T_solid,
                T_melt=T_melt,
                steps_npt=steps_npt,
                steps_melt=steps_melt,
                steps_coexist=steps_coexist,
                a0=a0,
                dump_every=dump_every,
                verify_melting_cn=verify_melting_cn,
            )
            results_by_index[i] = res
            print(f"\n--- Repeat {i+1}/{n_repeats} (seed={seed}) done: "
                  f"Tm = {res['tm']:.2f} K "
                  f"(exp = {GE_TM_EXP} K, dT = {res['tm'] - GE_TM_EXP:+.1f} K)")
        except Exception as e:
            print(f"\n--- Repeat {i+1}/{n_repeats} (seed={seed}) FAILED: {e}")

    if repeat_index is not None:
        # SLURM array task: only this repeat runs here. Aggregation happens in
        # a separate `--aggregate` pass once every array task has finished.
        if not results_by_index:
            sys.exit(1)
        return

    all_results = [results_by_index[i] for i in sorted(results_by_index)]
    _summarize(all_results, outdir, pot, nx, ny, nz)


# ---------------------------------------------------------------------------
# Quick mode defaults
# ---------------------------------------------------------------------------

QUICK = {
    "nx": 4, "ny": 4, "nz": 8,
    "n_repeats": 1,
    "steps_npt": 2_500,
    "steps_melt": 2_500,
    "steps_coexist": 25_000,
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melting point of diamond Ge via two-phase coexistence (LAMMPS + MTP)"
    )
    parser.add_argument("--pot", default=None,
                        help="Path to potential (.almtp). Required unless --aggregate is set "
                             "(aggregation only reads existing logs; --pot is recorded in "
                             "tm_summary.txt if given).")

    parser.add_argument("--nx", type=int, default=8,
                        help="x repeats (default: 8)")
    parser.add_argument("--ny", type=int, default=8,
                        help="y repeats")
    parser.add_argument("--nz", type=int, default=16,
                        help="z repeats (interface normal)")

    parser.add_argument("--T-solid", type=float, default=500.0,
                        help="NPT equilibration T (default: 500 K)")
    parser.add_argument("--T-melt", type=float, default=1900.0,
                        help="NVT melting T (default: 1900 K)")
    parser.add_argument("--steps-npt", type=int, default=10_000,
                        help="NPT steps at dt=2 fs (default: 10000 = 20 ps)")
    parser.add_argument("--steps-melt", type=int, default=10_000,
                        help="NVT melt steps (default: 10000 = 20 ps)")
    parser.add_argument("--steps-coexist", type=int, default=70_000,
                        help="NpH coexist steps (default: 100000 = 140 ps)")

    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Number of repeats (default: 10)")
    parser.add_argument("--repeat-index", type=int, default=None,
                        help="Run only this single repeat (0-based index), writing to "
                             "outdir/repeat_NN/. Set to $SLURM_ARRAY_TASK_ID by a SLURM "
                             "array job (see scripts/submit_melting_point.sh) so each task "
                             "runs one repeat. If omitted, all --n-repeats run sequentially.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Skip running LAMMPS; collect Tm from existing repeat_NN/ "
                             "directories under --outdir and write tm_summary.txt. Run "
                             "this after all SLURM array tasks have completed.")
    parser.add_argument("--seed-start", type=int, default=42,
                        help="Starting random seed")
    parser.add_argument("--dump-every", type=int, default=1000,
                        help="Dump trajectory every N steps (default: 1000)")

    parser.add_argument("--lammps", default="lmp_mpi",
                        help="LAMMPS binary (default: lmp_mpi)")
    parser.add_argument("--np", type=int, default=1,
                        help="MPI processes per repeat (default: 1)")

    parser.add_argument("--a0", type=float, default=5.7567,
                        help="Diamond lattice constant in A (default: 5.7567, MTP eq.)")

    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/tests/<pot's parent dir "
                             "name>/melting_point, e.g. --pot results/potentials/pot_660277/pot.almtp "
                             "-> results/tests/pot_660277/melting_point; falls back to "
                             "results/tests/melting_point if --pot is omitted with --aggregate)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 4x4x8 cell, 1 repeat, short runs")
    parser.add_argument("--no-verify-melting", action="store_true",
                        help="Skip CN check after Stage 2")

    args = parser.parse_args()

    if not args.aggregate and args.pot is None:
        parser.error("--pot is required unless --aggregate is set")

    if args.quick:
        print("=== Quick mode ===")
        nx, ny, nz = QUICK["nx"], QUICK["ny"], QUICK["nz"]
        n_repeats = QUICK["n_repeats"]
        steps_npt = QUICK["steps_npt"]
        steps_melt = QUICK["steps_melt"]
        steps_coexist = QUICK["steps_coexist"]
    else:
        nx, ny, nz = args.nx, args.ny, args.nz
        n_repeats = args.n_repeats
        steps_npt = args.steps_npt
        steps_melt = args.steps_melt
        steps_coexist = args.steps_coexist

    if args.outdir is None:
        if args.pot is not None:
            run_name = Path(args.pot).resolve().parent.name
            outdir = Path("results/tests") / run_name / "melting_point"
        else:
            outdir = Path("results/tests/melting_point")
    else:
        outdir = Path(args.outdir)

    run(
        pot=args.pot,
        outdir=outdir,
        lammps=args.lammps,
        np_cores=args.np,
        n_repeats=n_repeats,
        nx=nx, ny=ny, nz=nz,
        T_solid=args.T_solid,
        T_melt=args.T_melt,
        steps_npt=steps_npt,
        steps_melt=steps_melt,
        steps_coexist=steps_coexist,
        a0=args.a0,
        seed_start=args.seed_start,
        dump_every=args.dump_every,
        verify_melting_cn=not args.no_verify_melting,
        repeat_index=args.repeat_index,
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()
