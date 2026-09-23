"""
Active learning loop for MTP potential refinement.

Follows the MLIP-2 tutorial-2 workflow (md_al_mtp.sh). All AL state lives
under <pot_path's folder>/AL/: updated_pot.almtp and updated_train.cfg are
initialized once (copied from --pot / init_train_cfg) and then updated in
place every iteration — no per-iteration copies. Each iteration lives in
AL/iter_N/, holding that iteration's own MD/selection artifacts plus a
labeling subfolder AL/iter_N/<tagging_engine>/ (tagging_engine is 'vasp' or
'gap', see Steps C/D below): for 'vasp' this holds one struct_NNN/ per
selected structure (one VASP run each); for 'gap' it holds the batched
turbogap predict inputs/output directly, flat, since one call covers all
selected structures at once.

  A. LAMMPS MD        — run MD with MLIP selection; extrapolative configs
                        written to iter_dir/preselected.cfg
  B. select_add       — pick most informative subset → iter_dir/selected.cfg
  C/D. Labeling       — single-point reference labels on each selected
                        structure, via one of two engines chosen by
                        config/active_learning.yaml's tagging_engine:
                          'vasp' (default) — DFT single-point via ASE Vasp,
                                              one VASP run per structure
                                              (run_dft_labeling)
                          'gap'             — TurboGAP GAP potential,
                                              one turbogap predict call over
                                              all selected structures
                                              (run_gap_labeling, reuses
                                              src/tagging_structs/GAP_settings.py)
  E. retrain          — append iter_dir/new_added.cfg to updated_train.cfg,
                        retrain updated_pot.almtp (warm start)

Usage:
    python src/active_learning_structs/active_learning.py
    python src/active_learning_structs/active_learning.py --pot results/potentials/pot.almtp --max-iter 10
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force line-buffering so Python prints appear before subprocess output in SLURM logs.
sys.stdout.reconfigure(line_buffering=True)

# Ensure repo root is on sys.path so `from src.X import` works when invoked as
# `python src/active_learning_structs/active_learning.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tagging_structs"))

import numpy as np
import yaml

DEFAULT_TRAIN_CONFIG = "config/training.yaml"
DEFAULT_AL_CONFIG = "config/active_learning.yaml"

# Reverse of convert.py SPECIES_MAP: type_index → element symbol
_IDX_TO_SYMBOL: dict[int, str] = {0: "Ge"}
_SYMBOL_TO_IDX: dict[str, int] = {v: k for k, v in _IDX_TO_SYMBOL.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _count_cfg(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text().count("BEGIN_CFG")


def _truncate_cfg(path: Path, n: int) -> None:
    """Keep only the first n config blocks in a CFG file, overwriting it."""
    blocks = re.split(r"(?=BEGIN_CFG\b)", path.read_text())
    blocks = [b for b in blocks if b.strip().startswith("BEGIN_CFG")]
    path.write_text("\n".join(blocks[:n]) + "\n")


def _record_trajectory_info(
    al_root: Path,
    tagging_engine: str,
    initial_potential: str,
    initial_train_cfg: Path,
    trajectories: list[dict] | None,
) -> None:
    """Append a record of this AL run's inputs to <al_root>/trajectory_info.txt.

    Lives at the AL folder's top level alongside updated_pot.almtp and
    updated_train.cfg, so the provenance of the structures accumulated into
    those two files stays traceable. One block per distinct tagging_engine
    ('vasp'/'gap' — guarded so resuming an already-started run doesn't
    duplicate its entry) recording the initial potential, the initial
    training dataset it was copied from, and the trajectories config used to
    generate the MD structures.
    """
    info_path = al_root / "trajectory_info.txt"
    header = f"[tagging_engine: {tagging_engine}]"
    existing = info_path.read_text() if info_path.exists() else ""
    if header in existing:
        return
    traj_text = yaml.dump(trajectories, default_flow_style=False, sort_keys=False) if trajectories else "(none)\n"
    block = (
        f"{header}\n"
        f"initial_potential: {initial_potential}\n"
        f"initial_training_dataset: {initial_train_cfg}\n"
        f"trajectories:\n"
        + "".join(f"  {line}\n" for line in traj_text.splitlines())
        + "\n"
    )
    with info_path.open("a") as f:
        f.write(block)
    print(f"  Trajectory info recorded: {info_path}")


# ---------------------------------------------------------------------------
# MLIP commands
# ---------------------------------------------------------------------------


def select_add(mlp: str, pot: str, train_cfg: str, preselected: str, selected: Path) -> int:
    # Wrap with srun -n $SLURM_NTASKS: mlp select_add is the same MPI-capable
    # build as `mlp train` (see retrain()'s equivalent override) but was
    # otherwise always invoked as a single, unwrapped process — slow for a
    # large preselected.cfg (the MaxVol selection scan is the bottleneck).
    # OMP_NUM_THREADS=1 is required here: submit_active_learning.sh exports
    # OMP_NUM_THREADS=$SLURM_NTASKS for the whole driver job, and without
    # pinning it down per rank, every one of the N MPI ranks below would also
    # spawn N OpenBLAS/OpenMP threads for its own MaxVol linear algebra —
    # this oversubscription is what OOM-killed a prior run at -n 128
    # (scripts/submit_train.sh already sets OMP_NUM_THREADS=1 for this same
    # reason before its own srun-wrapped `mlp train` call).
    n_tasks = os.environ.get("SLURM_NTASKS", "1")
    mlp_launch = f"srun -n {n_tasks} {mlp}"
    cmd = shlex.split(mlp_launch) + ["select_add", pot, train_cfg, preselected, str(selected)]
    print("  Select:", " ".join(cmd))
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return _count_cfg(selected)


# ---------------------------------------------------------------------------
# LAMMPS MD (Step A)
# ---------------------------------------------------------------------------


def _collect_preselected(run_dir: Path) -> Path:
    """Merge MLIP-numbered preselected.cfg.N files into a canonical preselected.cfg.

    MLIP's save_extrapolative_to appends a numeric suffix (.0, .1, …) rather
    than writing plain preselected.cfg.  Concatenate all numbered files into
    one so downstream code has a consistent target.  Returns the path
    (which may not exist if MLIP saved nothing).
    """
    canonical = run_dir / "preselected.cfg"
    parts = sorted(run_dir.glob("preselected.cfg.[0-9]*"))
    if not parts:
        return canonical
    with canonical.open("w") as out:
        for p in parts:
            out.write(p.read_text())
    return canonical


def _tag_cfg(content: str, feature: str, value: str) -> str:
    """Insert 'Feature <feature> <value>' into every BEGIN_CFG block."""
    return re.sub(r"(BEGIN_CFG\b)", rf"\1\n    Feature {feature} {value}", content)


def _split_cfg_by_feature(cfg: Path, feature: str, match_value: str, iter_dir: Path) -> tuple[Path, Path]:
    """Split a CFG file into two files: blocks where Feature==match_value and the rest."""
    blocks = re.split(r"(?=BEGIN_CFG\b)", cfg.read_text())
    blocks = [b for b in blocks if b.strip().startswith("BEGIN_CFG")]
    matched, rest = [], []
    pattern = re.compile(rf"Feature\s+{re.escape(feature)}\s+{re.escape(match_value)}\b")
    for b in blocks:
        (matched if pattern.search(b) else rest).append(b)
    matched_path = iter_dir / f"selected_{feature}_{match_value}.cfg"
    rest_path    = iter_dir / f"selected_{feature}_other.cfg"
    matched_path.write_text(("\n".join(matched) + "\n") if matched else "")
    rest_path.write_text(("\n".join(rest) + "\n") if rest else "")
    return matched_path, rest_path


def _write_lammps_structure(lat_param: float, path: Path, supercell_size: int = 1, cubic: bool = False) -> Path:
    """Write a Ge diamond LAMMPS data file to path and return its absolute path.

    cubic=False → primitive cell (2 atoms); cubic=True → conventional cubic cell (8 atoms).
    supercell_size N tiles the cell N×N×N before writing.
    """
    from ase.build import bulk
    from ase.io import write as ase_write
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = bulk("Ge", "diamond", a=lat_param, cubic=cubic)
    if supercell_size > 1:
        atoms = atoms.repeat(supercell_size)
    ase_write(str(path), atoms, format="lammps-data", atom_style="atomic")
    return path.resolve()


def _system_setup_block(read_data: str | None) -> str:
    """Return the ${SYSTEM_SETUP} block for md_nvt.in (solid template).

    If read_data is given, the box comes from that LAMMPS data file and the
    native lattice-creation commands are commented out. Otherwise the box is
    built in-place from ${LAT_PARAM}/${SUPERCELL_SIZE} (substituted by the
    caller's later .replace() calls) and read_data is commented out instead.
    """
    if read_data:
        data_path = Path(read_data).resolve()
        return (
            "#variable    lat_param equal ${LAT_PARAM}   # Ge diamond cubic lattice constant (Å)\n"
            "#lattice     diamond ${lat_param}\n"
            "#region      box block 0 ${SUPERCELL_SIZE} 0 ${SUPERCELL_SIZE} 0 ${SUPERCELL_SIZE} units lattice\n"
            "#create_box  1 box\n"
            "#create_atoms 1 box\n"
            f"read_data   {data_path}"
        )
    return (
        "variable    lat_param equal ${LAT_PARAM}   # Ge diamond cubic lattice constant (Å)\n"
        "lattice     diamond ${lat_param}\n"
        "region      box block 0 ${SUPERCELL_SIZE} 0 ${SUPERCELL_SIZE} 0 ${SUPERCELL_SIZE} units lattice\n"
        "create_box  1 box\n"
        "create_atoms 1 box\n"
        "#read_data   ${data_file}"
    )


def run_lammps_md(
    lammps: str,
    template: Path,
    T: float,
    run_dir: Path,
    pot_path: str,
    threshold: float,
    grade_break: float = 0.0,
    lat_param: float = 5.76,
    supercell_size: int = 1,
    cubic: bool = False,
    read_data: str | None = None,
) -> Path:
    """Patch the LAMMPS template and run MD from run_dir.

    This LAMMPS build's pair_MLIP reads settings as inline key=value args
    on the pair_style line (no separate ini file). The template uses
    ${POT_PATH}, ${GRADE_THRESHOLD}, ${GRADE_BREAK}, and ${SYSTEM_SETUP}
    as placeholders. LAMMPS runs with cwd=run_dir so
    select:save_selected_to=preselected.cfg resolves to run_dir/preselected.cfg.
    Returns the path where MLIP writes extrapolative configs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    system_setup = _system_setup_block(read_data)
    text = (
        template.read_text()
        .replace("${T_RUN}", str(T))
        .replace("${POT_PATH}", str(Path(pot_path).resolve()))
        .replace("${GRADE_THRESHOLD}", str(threshold))
        .replace("${GRADE_BREAK}", f"{(grade_break if grade_break > 0 else threshold * 5):.1f}")
        .replace("${SYSTEM_SETUP}", system_setup)
        .replace("${SUPERCELL_SIZE}", str(supercell_size))
        .replace("${LAT_PARAM}", str(lat_param))
    )
    (run_dir / "md.in").write_text(text)
    cmd = shlex.split(lammps) + ["-in", "md.in", "-log", "log.lammps", "-screen", "none"]
    print("  LAMMPS:", " ".join(cmd), f"(cwd={run_dir})")
    result = subprocess.run(cmd, cwd=str(run_dir))
    if result.returncode != 0:
        print(f"  Warning: LAMMPS exited {result.returncode}")

    return _collect_preselected(run_dir)

 
# ---------------------------------------------------------------------------
# LAMMPS MD — parallel (Step A with N trajectories)
# ---------------------------------------------------------------------------

def _liquid_lat_param(T: float) -> float:
    """Diamond cubic lattice constant (Å) from liquid Ge density at temperature T (K).

    ρ(T) = −4.42×10⁻⁴·T + 6.17  g/cm³  (experimental fit)
    a = (8·M / (Nₐ·ρ))^(1/3) × 10⁸  Å
    """
    import math
    rho = -4.42e-4 * T + 6.17       # g/cm³
    M_Ge = 72.630                    # g/mol
    N_A  = 6.02214076e23
    return (8 * M_Ge / (N_A * rho)) ** (1 / 3) * 1e8


def _lat_param_from_density(density_g_cm3: float) -> float:
    """Diamond cubic lattice constant (Å) for a target density (g/cm³).

    a = (8·M / (Nₐ·ρ))^(1/3) × 10⁸  Å   (8-atom conventional cubic cell)
    """
    M_Ge = 72.630   # g/mol
    N_A  = 6.02214076e23
    return (8 * M_Ge / (N_A * density_g_cm3)) ** (1 / 3) * 1e8


def _quench_steps(T_high: float, T_low: float, rate_K_per_s: float,
                   timestep_ps: float = 0.001) -> int:
    """Number of MD steps to ramp from T_high to T_low at rate_K_per_s (K/s)."""
    delta_T = T_high - T_low
    time_ps = delta_T / rate_K_per_s * 1e12   # seconds → picoseconds
    return max(1000, int(time_ps / timestep_ps))


# Fixed melt-quench protocol — must match the hardcoded `variable T_run` /
# `variable T_melt` in config/lammps/md_melt_quench.in. Only density and
# quench_rate vary per trajectory; everything else in that template (stage
# lengths, T_run, T_melt) is the same for every melt_quench run.
_MELT_QUENCH_T_RUN = 100.0
_MELT_QUENCH_T_MELT = 1900.0


def run_parallel_lammps_md(
    lammps: str,
    trajectories: list[dict],
    iter_dir: Path,
    pot_path: str,
    default_threshold: float,
    default_grade_break: float = 0.0,
    default_lat_param: float = 5.76,
    default_supercell_size: int = 1,
    default_cubic: bool = False,
) -> Path:
    """Launch one LAMMPS job per trajectory in parallel; merge preselected configs.

    Each trajectory dict requires 'template'. 'solid'/'liquid' trajectories
    additionally require 'temperature' and 'lat_param'. Optional keys:
      template        : 'solid' (default), 'liquid', or 'melt_quench' — selects
                        md_nvt.in, md_nvt_liquid.in (3-step melt→cool→equil
                        protocol), or md_melt_quench.in (fixed-volume 4-stage
                        equilibrate→heat/melt→quench→anneal protocol for
                        amorphous structures at a chosen density)
      grade_threshold : per-trajectory γ_select (overrides default_threshold)
      grade_break     : per-trajectory γ_break; set equal to grade_threshold for
                        at-most-one-config-per-trajectory behaviour on liquid runs
                        (0 → threshold × 5)
      read_data       : solid trajectories only — path to a pre-built LAMMPS data
                        file; if given, md_nvt.in reads it directly instead of
                        building the box from lat_param/supercell_size/cubic

    melt_quench trajectories are a fixed protocol: T_run/T_melt/stage lengths
    are hardcoded in md_melt_quench.in itself (see _MELT_QUENCH_T_RUN/_T_MELT,
    which must match it) and are the same for every melt_quench trajectory.
    Only two keys vary per trajectory:
      density         : target density in g/cm³ — converted to a diamond cubic
                        lattice constant (a = (8·M/(Nₐ·ρ))^(1/3)) and used to
                        build an N×N×N supercell of the 8-atom conventional
                        cubic cell (cubic=True always; supercell_size N → 8·N³
                        atoms), which is fixed for the whole NVT trajectory
      quench_rate     : cooling rate in K/s from T_melt down to T_run;
                        converted to a step count via _quench_steps()

    Each job runs in iter_dir/md_NNN/ and writes preselected.cfg there.
    All non-empty per-trajectory preselected.cfg files are concatenated into
    iter_dir/preselected.cfg, which is returned.
    """
    solid_tmpl       = Path("config/lammps/md_nvt.in")
    liquid_tmpl      = Path("config/lammps/md_nvt_liquid.in")
    melt_quench_tmpl = Path("config/lammps/md_melt_quench.in")

    # Split the job's SLURM allocation evenly across the concurrent
    # trajectories (srun job steps packed into one allocation, same pattern
    # as _run_vasp_inline's inline VASP parallelism). Falls back to running
    # lammps_binary directly, unwrapped, outside a SLURM allocation (e.g.
    # local testing) where SLURM_NTASKS isn't set.
    allocated = os.environ.get("SLURM_NTASKS")
    if allocated and trajectories:
        n_per_traj = max(1, int(allocated) // len(trajectories))
        # Plain srun -n (no --overlap, no --exact): both were tried and
        # rejected. --overlap let different trajectories' ranks land on the
        # same physical CPUs (confirmed via taskset), tanking each process to
        # ~10-20% CPU instead of ~94%. --exact was meant to avoid the "step
        # creation still disabled, retrying (Requested nodes are busy)"
        # admission delay, but empirically made no difference on this
        # cluster — steps still admitted ~one at a time. So we accept that
        # delay: it's a one-time cost at the start of each trajectory, and
        # this plain form is the one confirmed (via taskset/ps CPU%) to give
        # each trajectory clean, dedicated cores once it does start.
        lammps_launch_cmd = (
            ["srun", "--nodes=1", f"--ntasks={n_per_traj}", "--cpus-per-task=1"]
            + shlex.split(lammps)
        )
        print(f"  Splitting {allocated} allocated tasks across {len(trajectories)} "
              f"trajectories: {n_per_traj} task(s) each.")
        # Pin OMP threads to 1 per MPI task so LAMMPS doesn't inherit the
        # driver job's own OMP_NUM_THREADS (set to the *full* allocation size
        # in submit_active_learning.sh) and oversubscribe every task.
        lammps_env = {**os.environ, "OMP_NUM_THREADS": "1"}
    else:
        lammps_launch_cmd = shlex.split(lammps)
        lammps_env = None

    procs: list[subprocess.Popen] = []
    md_dirs: list[Path] = []
    log_fhs: list = []

    for i, traj in enumerate(trajectories):
        md_dir = iter_dir / f"md_{i:03d}"
        md_dir.mkdir(parents=True, exist_ok=True)

        tmpl_type = traj.get("template", "solid")

        if tmpl_type == "melt_quench":
            # Fixed protocol: T_run/T_melt/stage lengths live in
            # md_melt_quench.in itself; only density and quench_rate vary.
            T            = _MELT_QUENCH_T_RUN
            T_melt       = _MELT_QUENCH_T_MELT
            density      = float(traj["density"])
            lat_param    = _lat_param_from_density(density)
            coef         = 0.0
            quench_rate  = float(traj["quench_rate"])
            quench_steps = _quench_steps(T_melt, T, quench_rate)
        else:
            T = float(traj["temperature"])
            if "lattice_constant" in traj:
                # Hard override: use as-is, skip the density calc and coef scaling below.
                lat_param = float(traj["lattice_constant"])
                coef = 0.0
            else:
                if "lat_param" in traj:
                    lat_param = float(traj["lat_param"])
                elif tmpl_type == "liquid":
                    lat_param = _liquid_lat_param(T)
                else:
                    lat_param = default_lat_param
                if tmpl_type == "liquid":
                    coef = float(traj.get("coef", 0.0))
                    if coef:
                        lat_param *= (1.0 + coef)
                else:
                    coef = 0.0
        supercell_size = int(traj.get("supercell_size", default_supercell_size))
        cubic          = True if tmpl_type == "melt_quench" else bool(traj.get("cubic", default_cubic))
        threshold = float(traj.get("grade_threshold", default_threshold))
        gb_raw    = float(traj.get("grade_break", default_grade_break))
        grade_break_val = gb_raw if gb_raw > 0 else threshold * 5

        if tmpl_type == "liquid":
            template = liquid_tmpl
        elif tmpl_type == "melt_quench":
            template = melt_quench_tmpl
        else:
            template = solid_tmpl

        if tmpl_type in ("liquid", "melt_quench"):
            read_data_line = f"read_data   {_write_lammps_structure(lat_param, md_dir / 'structure.lammps', supercell_size, cubic)}"
            system_setup = ""
        else:
            read_data_line = ""
            system_setup = _system_setup_block(traj.get("read_data"))

        text = (
            template.read_text()
            .replace("${T_RUN}", str(T))
            .replace("${POT_PATH}", str(Path(pot_path).resolve()))
            .replace("${GRADE_THRESHOLD}", str(threshold))
            .replace("${GRADE_BREAK}", f"{grade_break_val:.1f}")
            .replace("${READ_DATA_LINE}", read_data_line)
            .replace("${SYSTEM_SETUP}", system_setup)
            .replace("${SUPERCELL_SIZE}", str(supercell_size))
            .replace("${LAT_PARAM}", str(lat_param))
        )
        if tmpl_type == "melt_quench":
            text = text.replace("${QUENCH_STEPS}", str(quench_steps))
        (md_dir / "md.in").write_text(text)

        cmd = lammps_launch_cmd + ["-in", "md.in", "-log", "log.lammps", "-screen", "none"]
        lat_src = "" if ("lattice_constant" in traj or "lat_param" in traj or tmpl_type == "melt_quench") else " (derived)"
        coef_str = f"  coef={coef:+.3f}" if coef else ""
        extra = (
            f"  ρ={density:.2f}g/cc  T_melt={T_melt:.0f}K  quench={quench_rate:.1e}K/s→{quench_steps}steps"
            if tmpl_type == "melt_quench" else ""
        )
        print(
            f"  LAMMPS [{i:03d}] T={T:.0f}K  lat={lat_param:.4f}Å{lat_src}{coef_str}  ({tmpl_type})"
            f"  γ_sel={threshold}  γ_brk={grade_break_val:.1f}{extra}"
            f"  (cwd={md_dir})"
        )
        lammps_out = open(md_dir / "lammps.out", "w")
        proc = subprocess.Popen(cmd, cwd=str(md_dir), stdout=lammps_out, stderr=lammps_out, env=lammps_env)
        procs.append(proc)
        md_dirs.append(md_dir)
        log_fhs.append(lammps_out)

    sys.stdout.flush()

    # Wait for all trajectories to finish.
    for i, (proc, fh) in enumerate(zip(procs, log_fhs)):
        rc = proc.wait()
        fh.close()
        if rc != 0:
            print(f"  Warning: LAMMPS trajectory {i:03d} exited {rc}")

    # Collect numbered MLIP output files (preselected.cfg.0, .1, …) → preselected.cfg
    for md_dir in md_dirs:
        _collect_preselected(md_dir)

    # Concatenate all per-trajectory preselected.cfg into one merged file.
    # Tag each block with Feature strain 1/0 so step B.5 can strain selectively.
    merged = iter_dir / "preselected.cfg"
    with merged.open("w") as out:
        for i, md_dir in enumerate(md_dirs):
            partial = md_dir / "preselected.cfg"
            if partial.exists() and partial.stat().st_size > 0:
                content = partial.read_text()
                strain_flag = "1" if trajectories[i].get("strain", False) else "0"
                content = _tag_cfg(content, "strain", strain_flag)
                content = _tag_cfg(content, "type", trajectories[i].get("template", "solid"))
                n = content.count("BEGIN_CFG")
                out.write(content)
                print(f"  Trajectory {i:03d}: {n} config(s) added to merged pool (strain={strain_flag}).")
    print(f"  Merged preselected.cfg: {_count_cfg(merged)} total config(s).")
    return merged


# ---------------------------------------------------------------------------
# CFG reader  (moved to src/utils/convert_format.py — imported below — so that
# src/tagging_structs/VASP_settings.py can also use it without importing this
# module, which would create a circular import: this module already imports
# VASP_settings.set_cal)
# ---------------------------------------------------------------------------

from convert_format import read_cfg  # noqa: E402


# ---------------------------------------------------------------------------
# VASP DFT (Steps D/E)
# ---------------------------------------------------------------------------

def _write_vasp_input(atoms, vasp_dir: Path, vasp_settings: dict) -> None:
    """Write VASP input files (POSCAR, INCAR, KPOINTS, POTCAR) without running.

    DFT physics parameters (ENCUT, EDIFF, smearing, functional, precision,
    k-spacing, ...) come from src/tagging_structs/VASP_settings.set_cal(),
    the single source of truth shared with the structure-generation scripts.
    Only cluster/job-specific settings (parallelization, pseudopotential
    path/version, output directory) are overridden per call from
    config/active_learning.yaml's vasp_* job settings.
    """
    from VASP_settings import set_cal

    vasp_dir.mkdir(parents=True, exist_ok=True)
    pp_path = vasp_settings.get("vasp_pp_path", "")
    if pp_path:
        os.environ["VASP_PP_PATH"] = pp_path

    overrides = dict(directory=str(vasp_dir))
    if "vasp_pp_version" in vasp_settings:
        overrides["pp_version"] = vasp_settings["vasp_pp_version"]
    if "vasp_kpar" in vasp_settings:
        overrides["kpar"] = vasp_settings["vasp_kpar"]
    if "vasp_ncore" in vasp_settings:
        overrides["ncore"] = vasp_settings["vasp_ncore"]
    if "vasp_command" in vasp_settings:
        overrides["command"] = vasp_settings["vasp_command"]

    calc = set_cal(**overrides)
    calc.write_input(atoms)


def _submit_vasp_array_and_wait(
    dft_iter_dir: Path,
    n: int,
    vasp_settings: dict,
) -> None:
    """Write a SLURM array job script, submit it, and block until all tasks finish."""
    account   = vasp_settings.get("slurm_account", "")
    partition = vasp_settings.get("slurm_partition", "small")
    time_dft  = vasp_settings.get("slurm_time_dft", "01:00:00")
    ntasks    = int(vasp_settings.get("slurm_ntasks_dft", 1))
    setup     = vasp_settings.get("slurm_setup_dft", "")
    vasp_cmd  = vasp_settings.get("vasp_command", "vasp_std")

    account_line = f"#SBATCH --account={account}" if account else ""
    abs_iter_dir = dft_iter_dir.resolve()
    script_path = dft_iter_dir / "vasp_array.sh"
    script_path.write_text(f"""\
#!/bin/bash -l
#SBATCH --job-name=vasp_dft
{account_line}
#SBATCH --partition={partition}
#SBATCH --time={time_dft}
#SBATCH --ntasks={ntasks}
#SBATCH --output=job.out
#SBATCH --error=job.err

{setup}

export OMP_NUM_THREADS=1

# This sbatch call runs from inside the AL driver's own SLURM job. SLURM_*
# variables are always propagated into a new job's environment regardless of
# --export (see `man sbatch`), so the driver job's own SLURM_MEM_PER_CPU
# survives here alongside whatever memory variable (e.g. SLURM_MEM_PER_NODE)
# this job's own allocation sets — srun then refuses to start because they're
# mutually exclusive. Unset them so only this job's own allocation applies.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

STRUCT_DIR="{abs_iter_dir}/struct_$(printf '%03d' $SLURM_ARRAY_TASK_ID)"
cd "$STRUCT_DIR"
{vasp_cmd} >"{abs_iter_dir}/vasp_$SLURM_ARRAY_TASK_ID.out" 2>"{abs_iter_dir}/vasp_$SLURM_ARRAY_TASK_ID.err"
""")

    def _is_done(i):
        outcar = dft_iter_dir / f"struct_{i:03d}" / "OUTCAR"
        return outcar.exists() and "General timing" in outcar.read_text(errors="ignore")

    pending = [i for i in range(n) if not _is_done(i)]
    if not pending:
        print(f"  All {n} VASP results already exist — skipping sbatch.")
        return
    if len(pending) < n:
        print(f"  {n - len(pending)} structs already done; submitting {len(pending)} remaining.")

    array_spec = ",".join(str(i) for i in pending)
    cmd = ["sbatch", f"--array={array_spec}", str(script_path)]
    print(f"  Submitting VASP array job: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch submission failed: {result.stderr.strip() or result.stdout.strip()}")
    # Parse job ID from "Submitted batch job XXXX"
    job_id = result.stdout.strip().split()[-1]
    print(f"  {result.stdout.strip()}")

    poll_interval = 1200  # seconds between squeue checks
    while True:
        sq = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        job_alive = bool(sq.stdout.strip())
        if not job_alive:
            # Job gone — check each struct and report outcome.
            done_ids, failed_structs = [], []
            for i in range(n):
                if _is_done(i):
                    done_ids.append(i)
                else:
                    reason = "unknown"
                    err_file = dft_iter_dir / f"vasp_{i}.err"
                    if err_file.exists():
                        err_text = err_file.read_text(errors="ignore")
                        if "TIME LIMIT" in err_text:
                            reason = "time limit"
                        elif "CANCELLED" in err_text:
                            reason = "cancelled"
                        elif err_text.strip():
                            reason = err_text.strip().splitlines()[-1]
                    print(f"  struct_{i:03d}: FAILED ({reason})")
                    failed_structs.append(f"struct_{i:03d}")
            if done_ids:
                ids_str = ", ".join(f"{i:03d}" for i in done_ids)
                print(f"  struct_[{ids_str}] are done")
            if failed_structs:
                raise RuntimeError(
                    f"SLURM job {job_id}: {len(done_ids)}/{n} structs done, "
                    f"{len(failed_structs)} failed: {', '.join(failed_structs)}"
                )
            break
        time.sleep(poll_interval)
    print(f"  All {n} VASP array tasks finished.")


def _read_vasp_result(vasp_dir: Path, atoms_orig):
    """Read energy/forces/stress from a completed VASP directory.

    Uses OUTCAR (sets free_energy) as primary; falls back to vasprun.xml and
    manually aliases 'energy' → 'free_energy' so write_cfg can find it.
    """
    from ase.io import read as ase_read
    from ase.calculators.singlepoint import SinglePointCalculator

    outcar   = vasp_dir / "OUTCAR"
    vasprun  = vasp_dir / "vasprun.xml"

    if outcar.exists():
        atoms_out = ase_read(str(outcar))
    elif vasprun.exists():
        atoms_out = ase_read(str(vasprun))
    else:
        raise FileNotFoundError(f"No VASP output found in {vasp_dir}")

    calc = atoms_out.calc
    if calc is not None and "free_energy" not in calc.results:
        calc.results["free_energy"] = calc.results.get("energy")

    return atoms_out


def _remove_ghost_atoms(atoms):
    """Remove LAMMPS ghost (periodic image) atoms written by MLIP Wrapper.

    Uses the 'ghost' flag stored by read_cfg when the CFG has a ghost column.
    Falls back to fractional-coordinate deduplication only if the flag is absent
    (e.g. CFGs that pre-date the ghost column).
    """
    if "ghost" in atoms.arrays:
        real = np.where(~atoms.arrays["ghost"])[0]
        return atoms[real]
    # Legacy fallback: deduplicate by fractional position rounded to 3 dp.
    scaled = atoms.get_scaled_positions(wrap=True)
    seen: dict = {}
    unique: list = []
    for i, s in enumerate(scaled):
        key = tuple(np.round(s % 1.0, 3))
        if key not in seen:
            seen[key] = i
            unique.append(i)
    return atoms[unique]


def _run_vasp_inline(
    dft_iter_dir: Path,
    n: int,
    vasp_settings: dict,
) -> None:
    """Run VASP calculations sequentially using the current allocation."""
    vasp_cmd  = vasp_settings.get("vasp_command", "vasp_std")
    setup     = vasp_settings.get("slurm_setup_dft", "")
    ntasks    = int(vasp_settings.get("slurm_ntasks_dft", 1))
    # Build per-step srun command, pinning tasks and OMP threads explicitly.
    if vasp_cmd.startswith("srun"):
        step_cmd = (f"OMP_NUM_THREADS=1 srun --nodes=1 --ntasks={ntasks} "
                    f"--cpus-per-task=1 {vasp_cmd[len('srun'):].lstrip()}")
    else:
        step_cmd = vasp_cmd

    def _is_done(i):
        outcar = dft_iter_dir / f"struct_{i:03d}" / "OUTCAR"
        return outcar.exists() and "General timing" in outcar.read_text(errors="ignore")

    pending = [i for i in range(n) if not _is_done(i)]
    if not pending:
        print(f"  All {n} VASP results already exist — skipping.")
        return
    if len(pending) < n:
        print(f"  {n - len(pending)} structs already done; running {len(pending)} remaining.")

    total_needed = len(pending) * ntasks
    allocated = int(os.environ.get("SLURM_NTASKS", 0))
    if allocated and total_needed > allocated:
        print(f"  WARNING: need {total_needed} tasks ({len(pending)} × {ntasks}) "
              f"but only {allocated} allocated — jobs will run sequentially.")

    # Write a single wrapper script that launches all steps as background
    # srun processes and waits — avoids Python-level step creation races.
    setup_line = setup.strip()
    script_lines = ["#!/bin/bash -l"]
    if setup_line:
        script_lines.append(setup_line)
    for i in pending:
        struct_dir = dft_iter_dir / f"struct_{i:03d}"
        out = dft_iter_dir / f"vasp_{i}.out"
        err = dft_iter_dir / f"vasp_{i}.err"
        script_lines.append(
            f"( cd {struct_dir} && {step_cmd} ) >{out} 2>{err} &"
        )
    script_lines.append("wait")
    wrapper = dft_iter_dir / "run_vasp_parallel.sh"
    wrapper.write_text("\n".join(script_lines) + "\n")
    wrapper.chmod(0o755)

    print(f"  Launching {len(pending)} VASP jobs in parallel ({ntasks} tasks each):")
    for i in pending:
        print(f"    struct_{i:03d} — submitted")
    sys.stdout.flush()

    rc = subprocess.run(["bash", str(wrapper)]).returncode
    for i in pending:
        done = _is_done(i)
        print(f"    struct_{i:03d} — {'done' if done else 'FAILED'}")
        sys.stdout.flush()
    print(f"  All {len(pending)} VASP jobs finished.")


def run_dft_labeling(
    selected_cfg: Path,
    dft_iter_dir: Path,
    vasp_settings: dict,
) -> Path:
    """Label structures in selected_cfg via VASP; write labelled.cfg.

    1. Write VASP input files for every structure.
    2. Run VASP: 'sbatch' (default) submits a SLURM array job;
                 'inline' runs jobs in parallel as subprocesses.
    3. Collect results from each OUTCAR and write labelled.cfg.
    """
    from convert_format import write_cfg

    vasp_mode = vasp_settings.get("vasp_mode", "sbatch")
    dft_iter_dir.mkdir(parents=True, exist_ok=True)
    atoms_list = read_cfg(selected_cfg)

    # Step 1: write all VASP inputs (skip structs already successfully completed).
    struct_info = []
    for i, (atoms, meta) in enumerate(atoms_list):
        atoms = _remove_ghost_atoms(atoms)
        vasp_dir = dft_iter_dir / f"struct_{i:03d}"
        outcar = vasp_dir / "OUTCAR"
        already_done = outcar.exists() and "General timing" in outcar.read_text(errors="ignore")
        if not already_done:
            print(f"  Writing VASP input: struct_{i:03d} ({len(atoms)} atoms)")
            _write_vasp_input(atoms, vasp_dir, vasp_settings)
        struct_info.append((i, atoms, vasp_dir, meta))

    # Step 2: run VASP.
    if vasp_mode == "inline":
        _run_vasp_inline(dft_iter_dir, len(struct_info), vasp_settings)
    else:
        _submit_vasp_array_and_wait(dft_iter_dir, len(struct_info), vasp_settings)

    # Step 3: collect results.
    labeled = []
    for i, atoms_orig, vasp_dir, meta in struct_info:
        try:
            atoms_out = _read_vasp_result(vasp_dir, atoms_orig)
            atoms_out.info["config_type"] = meta.get("type", "unknown")
            labeled.append((meta.get("orig_index", str(i)), atoms_out))
        except Exception as exc:
            print(f"  WARNING: VASP failed for struct {i}: {exc} — skipping")

    labelled_path = dft_iter_dir / "labelled.cfg"
    write_cfg(labeled, labelled_path)
    return labelled_path


def run_gap_labeling(
    selected_cfg: Path,
    gap_iter_dir: Path,
    gap_settings: dict,
) -> Path:
    """Label structures in selected_cfg via the GAP potential; write labelled.cfg.

    Counterpart of run_dft_labeling for tagging_engine: gap. Reuses
    src/tagging_structs/GAP_settings.py's turbogap-predict workflow (same GAP
    potential/backend as the standalone tagging script) instead of VASP DFT —
    single `turbogap predict` call on all selected structures at once, no
    per-structure SLURM array job needed.
    """
    from convert_format import write_cfg
    from GAP_settings import _merge_labels, run_predict

    gap_iter_dir.mkdir(parents=True, exist_ok=True)
    atoms_list = read_cfg(selected_cfg)

    orig_frames = []
    for i, (atoms, meta) in enumerate(atoms_list):
        atoms = _remove_ghost_atoms(atoms)
        atoms.info["config_type"] = meta.get("type", "unknown")
        atoms.info["orig_index"] = meta.get("orig_index", str(i))
        orig_frames.append(atoms)

    print(f"  Running turbogap predict on {len(orig_frames)} structures...")
    traj_out = run_predict(orig_frames, gap_iter_dir, gap_settings)

    from ase.io import read as ase_read
    gap_frames_raw = ase_read(str(traj_out), index=":")
    if len(gap_frames_raw) != len(orig_frames):
        raise RuntimeError(
            f"frame count mismatch: {len(orig_frames)} selected vs "
            f"{len(gap_frames_raw)} turbogap output"
        )

    gap_frames = _merge_labels(orig_frames, gap_frames_raw)
    labeled = [
        (orig.info.get("orig_index", str(i)), gap)
        for i, (orig, gap) in enumerate(zip(orig_frames, gap_frames))
    ]

    labelled_path = gap_iter_dir / "labelled.cfg"
    write_cfg(labeled, labelled_path)
    return labelled_path


# ---------------------------------------------------------------------------
# Training-set management
# ---------------------------------------------------------------------------

def append_new_configs(new_cfg: Path, train_data: Path) -> int:
    """Append every config block in new_cfg to train_data. Returns count appended."""
    blocks = re.split(r"(?=BEGIN_CFG\b)", new_cfg.read_text())
    blocks = [b.strip() for b in blocks if b.strip().startswith("BEGIN_CFG")]
    if not blocks:
        print("  No configs to add.")
        return 0
    with open(train_data, "a") as f:
        f.write("\n\n".join(blocks) + "\n")
    print(f"  Added {len(blocks)} new config(s) to {train_data}")
    return len(blocks)


def retrain(train_cfg_path: str, pot_path: str, train_config: str, iteration: int, log_path: str = "") -> None:
    from src.train import load_config, run_training
    cfg = load_config(train_config)
    # Wrap mlp_binary with `srun -n $SLURM_NTASKS` for this retrain call only
    # (matching scripts/submit_train.sh's own MPI launch), so the retrain step
    # uses the AL job's full allocation instead of running on a single core.
    # select_add/check_errors calls elsewhere in this file read mlp_binary
    # straight from config/training.yaml and are unaffected by this override.
    n_tasks = os.environ.get("SLURM_NTASKS", "1")
    mlp_binary = f"srun -n {n_tasks} {cfg['mlp_binary']}"
    overrides = {
        "train_cfg": train_cfg_path,
        "mtp_template": pot_path,   # warm-start from current AL potential
        "output_potential": pot_path,
        "init_random": False,
        "mlp_binary": mlp_binary,
        "log_path": log_path or None,
    }
    # Pin OMP_NUM_THREADS=1 for this call, same reason as select_add()'s
    # override: submit_active_learning.sh exports OMP_NUM_THREADS=$SLURM_NTASKS
    # for the whole driver job, and run_training's subprocess.Popen (in
    # src/train.py) has no env parameter to override per-call, so it inherits
    # that unless we temporarily patch it here — otherwise each of the
    # n_tasks MPI ranks below also spawns OMP_NUM_THREADS-many OpenBLAS
    # threads of its own (this OOM-killed select_add() at -n 128; retrain()
    # has the same exposure even though it happened not to trigger it yet).
    prev_omp = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        run_training(cfg, overrides)
    finally:
        if prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev_omp


# ---------------------------------------------------------------------------
# Straining (Step B.5)
# ---------------------------------------------------------------------------

def _write_cfg_block(atoms, meta: dict, extra: dict | None = None) -> str:
    """Serialize one ASE Atoms object as a minimal MLIP CFG block."""
    cell = atoms.get_cell()
    pos  = atoms.get_positions()
    lines = ["BEGIN_CFG", " Size", f"    {len(atoms)}", " Supercell"]
    for row in cell:
        lines.append(f"    {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
    lines.append(" AtomData: id type cartes_x cartes_y cartes_z")
    for i, (sym, xyz) in enumerate(zip(atoms.symbols, pos)):
        tid = _SYMBOL_TO_IDX.get(sym, 0)
        lines.append(f"    {i+1} {tid} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    for k, v in meta.items():
        lines.append(f" Feature {k} {v}")
    if extra:
        for k, v in extra.items():
            lines.append(f" Feature {k} {v}")
    lines.append("END_CFG")
    return "\n".join(lines)



#_STRAIN_EPS = [-0.05, -0.04, -0.03, -0.02, -0.01, +0.01, +0.02, +0.03, +0.04, +0.05] #strained_dft_v1

#_STRAIN_EPS = [-0.016, -0.012, -0.008, -0.004, 0.000, 0.004, 0.008, 0.012, 0.016] #strained_dft_v2

#_STRAIN_EPS = [-0.018, -0.014, -0.006, -0.002, 0.002, 0.006, 0.014, 0.018] #strained_dft_v3

_STRAIN_EPS = [-0.065, -0.055, -0.045, -0.035, -0.025, +0.025, +0.035, +0.045, +0.055, +0.065] #strained_dft_v4

def _strain_atoms_blocks(atoms, meta: dict) -> list[str]:
    """Return CFG block strings: 1 equilibrium + 3 modes x nonzero eps values."""
    cell0 = atoms.get_cell().copy()
    a0    = float(np.linalg.norm(cell0[0]))
    blocks: list[str] = []

    # equilibrium (eps=0) reference — generated once, not per-mode
    #at = atoms.copy()
    #blocks.append(_write_cfg_block(at, meta, {"strain_type": "equilibrium", "strain_eps": "0.000"}))

    for eps in _STRAIN_EPS:
        #if eps == 0.0:
            #continue  # already added above
        # 1. Hydrostatic: scale all vectors uniformly
        hc = (1.0 + eps) * cell0
        at = atoms.copy(); at.set_cell(hc, scale_atoms=True)
        blocks.append(_write_cfg_block(at, meta, {"strain_type": "hydrostatic", "strain_eps": f"{eps:.3f}"}))
        # 2. Uniaxial [100]: scale only the a vector
        uc = cell0.copy(); uc[0] = (1.0 + eps) * cell0[0]
        at = atoms.copy(); at.set_cell(uc, scale_atoms=True)
        blocks.append(_write_cfg_block(at, meta, {"strain_type": "uniaxial_100", "strain_eps": f"{eps:.3f}"}))
        # 3. Shear: add ε×|a₀| to the y-component of the a vector
        sc = cell0.copy(); sc[0, 1] += eps * a0
        at = atoms.copy(); at.set_cell(sc, scale_atoms=True)
        blocks.append(_write_cfg_block(at, meta, {"strain_type": "shear", "strain_eps": f"{eps:.3f}"}))
    return blocks



def _apply_strain(selected_cfg: Path, iter_dir: Path) -> Path:
    """Strain all AL-selected structures (3 modes × 10 ε = 30 per input).

    Reads selected_cfg, calls _strain_atoms_blocks for each structure, and
    writes iter_dir/selected_strained.cfg.
    """
    strained_path = iter_dir / "selected_strained.cfg"
    atoms_list = read_cfg(selected_cfg)
    blocks: list[str] = []
    for atoms, meta in atoms_list:
        blocks.extend(_strain_atoms_blocks(_remove_ghost_atoms(atoms), meta))
    strained_path.write_text("\n".join(blocks) + "\n")
    print(f"  Strained: {len(atoms_list)} structure(s) × 30 = {len(blocks)} configs → {strained_path}")
    return strained_path


def _make_no_al_strained_cfg(no_al_trajs: list[dict], out_path: Path, lat_param: float) -> None:
    """Build an S×S×S diamond cubic cell and apply strain; write to out_path.

    Constructs the conventional 8-atom Ge cubic cell using lat_param (global
    default, or per-trajectory override via 'lat_param' key), then tiles it
    into an S×S×S supercell ('supercell_size' key, default 1). Produces 30
    strained variants per no_al trajectory entry.
    """
    from ase.build import bulk
    blocks: list[str] = []
    for i, traj in enumerate(no_al_trajs):
        a = float(traj.get("lat_param", lat_param))
        S = int(traj.get("supercell_size", 1))
        atoms = bulk("Ge", "diamond", a=a, cubic=True)  # 8-atom conventional cell
        if S > 1:
            atoms = atoms.repeat(S)
        meta = {"type": "solid"}
        traj_blocks = _strain_atoms_blocks(atoms, meta)
        blocks.extend(traj_blocks)
        print(f"  no_al[{i:03d}] diamond {S}×{S}×{S} ({len(atoms)} atoms) a={a:.4f}Å: {len(traj_blocks)} strained configs.")
    out_path.write_text("\n".join(blocks) + "\n")
    print(f"  Total no_al strained: {len(blocks)} configs → {out_path}")


# Lattice-constant rates (a_new = a*(1+r)) sampled around equilibrium for the
# frozen-phonon force-constant probes. Range chosen from the actual thermal
# expansion result: integrating alpha_L(T) from 0->1000 K gives only ~0.57%
# (DFT) to ~0.99% (this project's own MTP fit) linear strain, so +/-0.03 gives
# a comfortable ~3-5x margin over the real excursion without spending points
# far outside the regime that matters (unlike src/physical_validation/thermal_properties.py's
# own +/-0.05 QHA-fit convention, which is deliberately wider for curvature
# robustness rather than to match the physical excursion). Point count/spacing
# (7, denser near 0) still differs from that script's evenly-spaced
# --n-volumes sweep, so it remains an independent held-out check.
_PHONON_LATTICE_RATES = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
# Displacement distance (Å) — deliberately different from the 0.01 Å default
# used by src/physical_validation/phonon_dispersion.py / src/physical_validation/thermal_properties.py, for the
# same held-out-evaluation reason.
_PHONON_DISPLACEMENT = 0.02


def _make_no_al_phonon_cfg(no_al_trajs: list[dict], out_path: Path, lat_param: float) -> None:
    """Build symmetry-complete frozen-phonon displacements at several volumes.

    For each 'phonon' no_al trajectory entry, builds a Ge diamond cell (using
    lat_param, or a per-trajectory 'lat_param' override), scales it to each
    lattice constant in _PHONON_LATTICE_RATES, tiles it into an S×S×S supercell
    ('supercell_size', default 2), and uses phonopy to generate the minimal
    symmetry-inequivalent displaced structure(s) (distance=_PHONON_DISPLACEMENT)
    at that volume. These are single-point force-constant probes — IBRION=-1
    (no ionic relaxation) is physically correct here, unlike the shear-strain
    case, since no external strain is imposed beyond the volume itself.
    """
    from ase import Atoms
    from ase.build import bulk
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    def _ase_to_phonopy(atoms):
        return PhonopyAtoms(
            symbols=list(atoms.get_chemical_symbols()),
            cell=atoms.get_cell().array.copy(),
            scaled_positions=atoms.get_scaled_positions(),
        )

    def _phonopy_to_ase(ph_atoms):
        return Atoms(
            symbols=list(ph_atoms.symbols),
            cell=ph_atoms.cell.copy(),
            positions=ph_atoms.positions.copy(),
            pbc=True,
        )

    blocks: list[str] = []
    for i, traj in enumerate(no_al_trajs):
        a = float(traj.get("lat_param", lat_param))
        S = int(traj.get("supercell_size", 3))
        cubic = bool(traj.get("cubic", False))
        unitcell = bulk("Ge", "diamond", a=a, cubic=cubic)
        meta = {"type": "phonon_fc"}

        n_this_traj = 0
        for rate in _PHONON_LATTICE_RATES:
            scaled = unitcell.copy()
            scaled.set_cell(scaled.get_cell() * (1.0 + rate), scale_atoms=True)

            phonon = Phonopy(
                _ase_to_phonopy(scaled),
                supercell_matrix=[[S, 0, 0], [0, S, 0], [0, 0, S]],
            )
            phonon.generate_displacements(distance=_PHONON_DISPLACEMENT)
            for sc in phonon.supercells_with_displacements:
                blocks.append(_write_cfg_block(
                    _phonopy_to_ase(sc), meta, {"lattice_rate": f"{rate:+.4f}"}
                ))
                n_this_traj += 1
        print(f"  no_al[{i:03d}] phonon a={a:.4f}Å S={S}: "
              f"{n_this_traj} displaced configs across {len(_PHONON_LATTICE_RATES)} volumes.")
    out_path.write_text("\n".join(blocks) + "\n")
    print(f"  Total no_al phonon: {len(blocks)} configs → {out_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_active_learning(
    train_cfg: dict,
    al_cfg: dict,
    pot_path: str,
    max_iter: int,
    al_pot_path: str = "",
    dft_dir: str = "",
    max_tagging_per_iter: int = 0,
    al_root: str = "",
) -> None:
    mlp = train_cfg["mlp_binary"]

    # All AL state lives under al_root, initialized once from the
    # caller-supplied potential/training set and updated in place every
    # iteration thereafter (no per-iteration copies, no date-stamped names).
    # Defaults to <pot_path's folder>/AL for backward compatibility, but can
    # be pointed at a sibling round folder (e.g. AL-2) so that continuing
    # from a previous round's output (AL-1/updated_pot.almtp) doesn't nest a
    # new AL/ folder inside it.
    al_root = Path(al_root) if al_root else Path(pot_path).parent / "AL"
    al_root.mkdir(parents=True, exist_ok=True)

    train_data = al_root / "updated_train.cfg"
    base_train = Path(al_cfg.get("init_train_cfg", "data/train-init.cfg"))
    if not train_data.exists():
        shutil.copy(base_train, train_data)
        print(f"  Created training set: {train_data} (copied from {base_train})")

    grade_threshold = al_cfg.get("grade_threshold", 2.0)
    grade_break     = al_cfg.get("grade_break", 0.0)
    temperature    = al_cfg.get("temperature", 300)
    lammps         = al_cfg.get("lammps_binary", "lmp")
    lat_param      = al_cfg.get("lat_param", 5.76)
    supercell_size = int(al_cfg.get("supercell_size", 1))
    cubic          = bool(al_cfg.get("cubic", False))
    trajectories   = al_cfg.get("trajectories", None)
    md_template    = Path("config/lammps/md_nvt.in")
    al_outdir      = al_root

    # AL writes to a separate potential file so the initial pot.almtp is never
    # overwritten. al_pot_path can be specified explicitly (e.g. to share one
    # output potential across multiple AL scripts); defaults to AL/updated_pot.almtp.
    if not al_pot_path:
        al_pot_path = str(al_root / "updated_pot.almtp")
    if not Path(al_pot_path).exists() or Path(al_pot_path).resolve() != Path(pot_path).resolve():
        shutil.copy(pot_path, al_pot_path)

    tagging_engine = al_cfg.get("tagging_engine", "vasp").lower()
    if tagging_engine not in ("vasp", "gap"):
        print(f"ERROR: unknown tagging_engine {tagging_engine!r} (expected 'vasp' or 'gap')", file=sys.stderr)
        sys.exit(1)
    vasp_settings  = {k: v for k, v in al_cfg.items() if k.startswith("vasp_") or k.startswith("slurm_")}
    gap_settings   = {k: v for k, v in al_cfg.items() if k.startswith("turbogap_") or k.startswith("slurm_")}

    _record_trajectory_info(al_root, tagging_engine, pot_path, base_train, trajectories)

    print("=== Active Learning ===")
    print(f"  Initial potential : {pot_path}")
    print(f"  AL potential      : {al_pot_path}")
    print(f"  Tagging engine    : {tagging_engine}")
    print(f"  Training set  : {train_data}  ({_count_cfg(train_data)} configs)")

    print(f"  Max iterations: {max_iter}")
    print(f"  Max tagging per iter: {max_tagging_per_iter if max_tagging_per_iter > 0 else 'unlimited'}")
    sys.stdout.flush()

    if not train_data.exists():
        print(f"ERROR: training set not found: {train_data}", file=sys.stderr)
        sys.exit(1)

    do_strain   = bool(trajectories and any(
        t.get("strain", False) for t in trajectories
        if t.get("template", "solid") == "solid"
    ))
    al_trajs    = [t for t in trajectories if not t.get("no_al", False)] if trajectories else []
    no_al_trajs = [t for t in trajectories if     t.get("no_al", False)] if trajectories else []
    no_al_strain_trajs = [t for t in no_al_trajs if t.get("template", "solid") != "phonon"]
    no_al_phonon_trajs = [t for t in no_al_trajs if t.get("template", "solid") == "phonon"]

    for iteration in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration}/{max_iter}")
        print(f"{'='*60}")

        iter_dir = al_outdir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        iteration_marker = iter_dir / ".iteration_complete"
        if iteration_marker.exists():
            print(f"\nIteration {iteration} already complete (DFT-labelled data already merged "
                  f"into training set) — skipping.")
            continue

        # Step A: LAMMPS MD with MLIP selection enabled (skipped for no_al-only runs)
        preselected = iter_dir / "preselected.cfg"
        if preselected.exists():
            print("\nStep A — skipping (preselected.cfg already exists).")
        elif al_trajs:
            print("\nStep A — LAMMPS MD with active learning selection...")
            sys.stdout.flush()
            print(f"  Parallel mode: {len(al_trajs)} AL trajectories.")
            preselected = run_parallel_lammps_md(
                lammps, al_trajs, iter_dir, al_pot_path,
                grade_threshold, grade_break,
                default_lat_param=lat_param,
                default_supercell_size=supercell_size,
                default_cubic=cubic,
            )
        elif not trajectories:
            print("\nStep A — LAMMPS MD with active learning selection...")
            sys.stdout.flush()
            preselected = run_lammps_md(
                lammps, md_template, temperature, iter_dir, al_pot_path, grade_threshold,
                grade_break=grade_break,
                lat_param=lat_param,
                supercell_size=supercell_size,
                cubic=cubic,
            )
        else:
            # All trajectories have no_al=True; no LAMMPS needed.
            print("\nStep A — skipping (all trajectories have no_al=True).")

        n_pre = _count_cfg(preselected) if preselected.exists() else 0
        if n_pre == 0 and not no_al_trajs:
            print("\nActive learning converged: no extrapolative structures found.")
            break
        if n_pre == 0:
            print("  No extrapolative structures from AL trajectories.")

        # Step B: select_add — pick most informative subset (only when AL found configs)
        iter_selected = iter_dir / "selected.cfg"
        n_selected = 0
        if n_pre > 0:
            if iter_selected.exists():
                print("\nStep B — skipping (selected.cfg already exists in iter_dir).")
                n_selected = _count_cfg(iter_selected)
            else:
                print("\nStep B — select_add...")
                sys.stdout.flush()
                n_selected = select_add(mlp, al_pot_path, str(train_data), str(preselected), iter_selected)
            if n_selected == 0 and not no_al_trajs:
                print("\nActive learning converged: select_add chose 0 structures.")
                break
            if n_selected > 0:
                print(f"  {n_selected} structures selected.")

        # Optionally cap how many AL-selected structures go to tagging this iteration.
        n_for_tagging = n_selected
        if n_selected > 0 and max_tagging_per_iter > 0 and n_selected > max_tagging_per_iter:
            _truncate_cfg(iter_selected, max_tagging_per_iter)
            n_for_tagging = max_tagging_per_iter
            print(f"  Capped to {n_for_tagging} structures for tagging (--max-tagging-per-iter={max_tagging_per_iter}).")

        # Step B.5: strain AL-selected structures from strain=True trajectories only
        cfg_for_tagging: Path | None = iter_selected if n_selected > 0 else None
        if do_strain and n_selected > 0:
            strained_cfg = iter_dir / "selected_strained.cfg"
            if strained_cfg.exists():
                print("\nStep B.5 — skipping (selected_strained.cfg already exists).")
                cfg_for_tagging = strained_cfg
            else:
                print("\nStep B.5 — straining structures from strain=True trajectories only...")
                sys.stdout.flush()
                to_strain, direct = _split_cfg_by_feature(iter_selected, "strain", "1", iter_dir)
                n_to_strain = _count_cfg(to_strain)
                n_direct    = _count_cfg(direct)
                if n_to_strain > 0:
                    expanded = _apply_strain(to_strain, iter_dir)
                    if n_direct > 0:
                        strained_cfg.write_text(expanded.read_text() + direct.read_text())
                    else:
                        strained_cfg = expanded
                else:
                    strained_cfg = direct
                cfg_for_tagging = strained_cfg
                print(f"  {n_to_strain} structure(s) strained × 30, {n_direct} passed through directly.")
            n_for_tagging = _count_cfg(cfg_for_tagging)

        # Step B.6: no_al trajectories — load structure, strain/displace directly, merge
        if no_al_trajs:
            no_al_cfg = iter_dir / "no_al_strained.cfg"
            if no_al_cfg.exists():
                print(f"\nStep B.6 — skipping (no_al_strained.cfg already exists).")
            else:
                parts: list[str] = []
                if no_al_strain_trajs:
                    print(f"\nStep B.6 — straining {len(no_al_strain_trajs)} no_al structure(s) directly...")
                    sys.stdout.flush()
                    strain_path = iter_dir / "no_al_strained_only.cfg"
                    _make_no_al_strained_cfg(no_al_strain_trajs, strain_path, lat_param)
                    parts.append(strain_path.read_text())
                if no_al_phonon_trajs:
                    print(f"\nStep B.6 — generating {len(no_al_phonon_trajs)} no_al phonon structure(s) directly...")
                    sys.stdout.flush()
                    phonon_path = iter_dir / "no_al_phonon_only.cfg"
                    _make_no_al_phonon_cfg(no_al_phonon_trajs, phonon_path, lat_param)
                    parts.append(phonon_path.read_text())
                no_al_cfg.write_text("".join(parts))
            if cfg_for_tagging is not None and _count_cfg(cfg_for_tagging) > 0:
                merged = iter_dir / "merged_for_tagging.cfg"
                if not merged.exists():
                    merged.write_text(cfg_for_tagging.read_text() + no_al_cfg.read_text())
                cfg_for_tagging = merged
            else:
                cfg_for_tagging = no_al_cfg
            n_for_tagging = _count_cfg(cfg_for_tagging)

        # Steps C/D: single-point labeling (VASP DFT or GAP)
        if cfg_for_tagging is None or n_for_tagging == 0:
            print("\nNo structures for tagging this iteration — skipping.")
            continue
        print(f"\nSteps C/D — {tagging_engine.upper()} labeling on {n_for_tagging} structures...")
        sys.stdout.flush()
        tagging_iter_dir = (
            Path(dft_dir) / f"iter_{iteration}"
            if dft_dir
            else iter_dir / tagging_engine
        )
        if tagging_engine == "gap":
            labelled = run_gap_labeling(cfg_for_tagging, tagging_iter_dir, gap_settings)
        else:
            labelled = run_dft_labeling(cfg_for_tagging, tagging_iter_dir, vasp_settings)
        n_labelled = _count_cfg(labelled)
        if n_labelled == 0:
            print("  No labelled configs from DFT — evaluating errors on current potential and stopping.")
            sys.stdout.flush()
            err_out = iter_dir / "check_errors.out"
            result = subprocess.run(
                shlex.split(mlp) + ["check_errors", al_pot_path, str(train_data)],
                stdout=err_out.open("w"), stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                print(f"  Warning: mlp check_errors exited {result.returncode}")
            else:
                print(f"  Errors written to {err_out}")
            break
        new_added = iter_dir / "new_added.cfg"
        shutil.copy(labelled, new_added)
        print(f"  Labelled configs written to {new_added}")

        n_new = _count_cfg(new_added)
        # Appending is idempotent via this marker so a crash/restart between
        # the append and the end of Step E (e.g. resuming a killed retrain)
        # doesn't double-append new_added.cfg into train_data.
        append_marker = iter_dir / ".new_added_appended"
        if n_new > 0 and not append_marker.exists():
            append_new_configs(new_added, train_data)
            append_marker.touch()
        elif n_new > 0:
            print(f"  {new_added} already appended to {train_data} — skipping re-append.")
        n_train = _count_cfg(train_data)

        # Step E: retrain
        if n_new == 0:
            print("\nStep E — skipping retrain (no new configs added to training set).")
            sys.stdout.flush()
        else:
            print(f"\nStep E — retraining on {n_train} configs...")
            sys.stdout.flush()
            retrain(str(train_data), al_pot_path, DEFAULT_TRAIN_CONFIG, iteration,
                    log_path=str(iter_dir / "retrain.log"))

            # Check errors on the training set immediately after retraining.
            err_out = iter_dir / "check_errors.out"
            print(f"\n  check_errors on {train_data} ({n_train} configs)...")
            sys.stdout.flush()
            err_result = subprocess.run(
                shlex.split(mlp) + ["check_errors", al_pot_path, str(train_data)],
                capture_output=True, text=True,
            )
            err_out.write_text(err_result.stdout + err_result.stderr)
            if err_result.returncode != 0:
                print(f"  Warning: mlp check_errors exited {err_result.returncode}")
            print(err_result.stdout or err_result.stderr, end="")
            print(f"  Errors written to {err_out}")
            sys.stdout.flush()

        iteration_marker.touch()

        if not al_trajs:
            print("\nno_al-only run: stopping after one iteration.")
            break

    print(f"\nActive learning complete.")
    print(f"  AL potential      : {al_pot_path}")
    print(f"  Final training set: {_count_cfg(train_data)} configs")

    # Final error report on the accumulated training set.
    if train_data.exists() and _count_cfg(train_data) > 0:
        outdir = al_root / "errors"
        outdir.mkdir(parents=True, exist_ok=True)
        report   = outdir / "errors.txt"
        log_file = outdir / "error.log"
        print(f"\n  check_errors [train] → {outdir}")
        sys.stdout.flush()
        result = subprocess.run(
            shlex.split(mlp) + [
                "check_errors", al_pot_path, str(train_data),
                f"--log={log_file}",
                f"--report_to={report}",
            ],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  Warning: mlp check_errors exited {result.returncode}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Active learning loop for MTP refinement")
    parser.add_argument("--config",    default=DEFAULT_TRAIN_CONFIG, help="Training YAML config")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG,    help="Active learning YAML config")
    parser.add_argument("--pot",       help="Potential path (overrides config)")
    parser.add_argument("--max-iter",  type=int, help="Max AL iterations (overrides al-config)")
    parser.add_argument("--al-pot",      help="Explicit output path for the evolving AL potential (default: <pot's folder>/AL/updated_pot.almtp)")
    parser.add_argument("--al-root",     help="Explicit AL working directory (overrides config; default: <pot's folder>/AL). Use to keep sequential AL rounds as sibling folders, e.g. AL-2, instead of nesting under the seed potential's folder")
    parser.add_argument("--temperature", type=float, help="NVT temperature in K (overrides al-config temperature)")
    parser.add_argument("--lat-param",   type=float, help="Diamond cubic lattice constant (Å) for lattice creation (overrides al-config lat_param)")
    parser.add_argument("--dft-dir",              help="Root directory for VASP DFT calculations (e.g. data/defect_dft or data/thermal_dft)")
    parser.add_argument("--max-tagging-per-iter", type=int, default=0, help="Cap DFT/GAP labeling jobs per iteration (0 = no cap)")
    args = parser.parse_args()

    train_cfg = _load(args.config)
    al_cfg    = _load(args.al_config)

    if args.temperature:
        al_cfg["temperature"] = args.temperature
    if args.lat_param:
        al_cfg["lat_param"] = args.lat_param

    pot_path             = args.pot or al_cfg.get("init_train_pot") or train_cfg["output_potential"]
    max_iter             = args.max_iter or al_cfg.get("max_iterations", 20)
    max_tagging_per_iter = args.max_tagging_per_iter or int(al_cfg.get("max_tagging_per_iter", 0))

    al_root = args.al_root or al_cfg.get("al_root", "")

    run_active_learning(train_cfg, al_cfg, pot_path, max_iter, al_pot_path=args.al_pot or "", dft_dir=args.dft_dir or "", max_tagging_per_iter=max_tagging_per_iter, al_root=al_root)


if __name__ == "__main__":
    main()
