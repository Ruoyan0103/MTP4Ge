"""Threshold displacement energy (Ed) of diamond-cubic Ge via recoil MD.

Ed(direction) is the lowest recoil kinetic energy given to one lattice atom
(the primary knock-on atom, PKA) along that direction that leaves a stable
Frenkel pair after the lattice has cooled and been relaxed.

Protocol (LAMMPS only, pair_style hybrid/overlay mtp nlh):
  0. Perfect lattice: n×n×n conventional diamond cells at --a0, then
     `fix box/relax iso` + CG minimisation, so the reference sites and the box
     are the potential's own 0 K lattice (a0 printed in the summary).
  1. Thermalisation (per repeat, only if --temperature > 0): NVT (Nose-Hoover)
     at T for --t-equil ps with velocity seed --seed + repeat. At T = 0 the
     perfect lattice is used directly and --repeats is forced to 1.
  2. One LAMMPS run per (direction, energy, repeat): the PKA (the sublattice-A
     atom closest to the box centre) gets |v| = sqrt(2E/m) along the
     direction; NVE with `fix dt/reset` (adaptive timestep, max --dt-max ps,
     max displacement --xmax Å per step) for --t-md ps, optionally with a
     Langevin thermostat at T on a border shell of thickness --border Å
     (default 0 = plain NVE); then CG minimisation (ftol --ftol eV/Å).
  3. Wigner-Seitz occupancy analysis (numpy/scipy, periodic nearest-site
     assignment against the perfect lattice, after removing the rigid
     centre-of-mass drift): vacancies = empty sites, interstitials = sum of
     (occupancy - 1). A run "makes a defect" if n_vac >= 1.
  4. Energy scan from --e-min to --e-max in --e-step eV, stopping at the first
     energy that leaves a defect (per direction & repeat); then optional
     bisection between the last defect-free energy and that energy down to
     --refine eV. Bisection assumes defect production is monotonic in E near
     threshold (not strictly true; the coarse scan result is kept in the CSV).

Direction convention: the PKA sits on sublattice A (fractional (0,0,0) of the
conventional cell); its four nearest neighbours lie along <111> = [111],
[1-1-1], [-11-1], [-1-11]. Hence [111] is a head-on (bond) direction while
[-1-1-1] points into the open channel towards the tetrahedral site —
they are inequivalent in diamond, so both are scanned by default.

Each (direction, energy, repeat) is an independent LAMMPS run in its own
directory; runs are parallelised with a thread pool (--workers) and each run
may itself be parallel via the --lammps command (e.g. "srun --exact -n 4
lmp_mpi"). Finished runs (result.json present) are skipped on restart.

Reference: TODO (DFT/GAP to be computed). Literature values for Ge vary
between methods and directions; none are tabulated here until checked.

Outputs (in --outdir):
  threshold_displacement.txt   per-direction Ed table (mean ± std over repeats)
  threshold_displacement.png   Ed by direction
  threshold_displacement.json  machine-readable summary
  runs.csv                     every LAMMPS run: E, n_vac, n_int, ΔE_relaxed, ...
  prep/, runs/<dir>_rNN/E_xxx.xx/   LAMMPS inputs, logs, final relaxed dumps

Usage:
    python src/physical_validation/threshold_displacement.py --pot results/potentials/pot_1019/bal.mtp
    python src/physical_validation/threshold_displacement.py --pot results/potentials/pot_1019/bal.mtp \\
        --supercell 6 --temperature 10 --repeats 3 --n-random 20 --workers 96 --batch 4 \\
        --lammps "srun --exact -n 4 -c 1 /projappl/project_2012355/CODE/lammps-MTP/src/lmp_mpi"
    # quick pipeline check (explicit energies, no refinement):
    python src/physical_validation/threshold_displacement.py --pot results/potentials/pot_1019/bal.mtp \\
        --supercell 3 --directions 100 --energies 5,20,60 --batch 3 --t-md 0.5 --no-refine
Heavy test — submit to HPC (one node, many concurrent runs):
    sbatch scripts/submit_threshold_displacement.sh results/potentials/pot_1019/bal.mtp
"""

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from energy_volume import DEFAULT_LAMMPS

GE_MASS = 72.630                 # amu
EV_PER_AMU_TO_A2_PS2 = 9648.533  # 1 eV/amu = 9648.533 Å²/ps²

NLH_COEFF = "nlh 32.0 32.0 0.05121 0.50355 0.44524 49.82156 12.56998 4.13684 1.28 2.0"

DEFAULT_DIRECTIONS = "100,110,111,-1-1-1,210,211,221,310"

# Ed reference values [eV] by direction label; empty until a verified
# DFT/GAP/experimental number is available.  reference: TODO (DFT/GAP to be computed)
DFT_REF: dict[str, float] = {}
GAP_REF: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def parse_direction(token: str) -> tuple[str, np.ndarray]:
    """'111' -> ([1,1,1]), '-1-1-1' -> ([-1,-1,-1]), '1,-1,0' style not used.

    Returns (label, unit vector). Label replaces '-' by 'm' (filesystem-safe).
    """
    comps = re.findall(r"-?\d", token.strip())
    if len(comps) != 3:
        raise ValueError(f"Cannot parse direction {token!r}: need 3 single-digit "
                         f"Miller components, e.g. 110 or -1-1-1")
    v = np.array([int(c) for c in comps], dtype=float)
    if not np.any(v):
        raise ValueError(f"Zero direction {token!r}")
    return "".join(comps).replace("-", "m"), v / np.linalg.norm(v)


def random_directions(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        v = rng.normal(size=3)
        out.append((f"rand{i:02d}", v / np.linalg.norm(v)))
    return out


# ---------------------------------------------------------------------------
# LAMMPS helpers
# ---------------------------------------------------------------------------

def _ff_block(pot_abs: str) -> str:
    return (f"pair_style  hybrid/overlay mtp nlh\n"
            f"pair_coeff  * * mtp {pot_abs} 0\n"
            f"pair_coeff  1 1 {NLH_COEFF}\n")


def _run_lammps(lammps_cmd: str, workdir: Path, infile: str = "in.lmp") -> None:
    cmd = shlex.split(lammps_cmd) + ["-in", infile, "-log", "log.lammps", "-screen", "none"]
    res = subprocess.run(cmd, cwd=str(workdir), stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LAMMPS exited {res.returncode} in {workdir}: "
                           f"{res.stderr.strip()[-500:]} (see log.lammps)")


def _read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a single-frame `write_dump ... id x y z modify sort id` file.

    Returns (ids, positions[N,3], box[3,2]).
    """
    lines = path.read_text().splitlines()
    n = int(lines[3])
    box = np.array([[float(x) for x in lines[5 + i].split()[:2]] for i in range(3)])
    data = np.loadtxt(lines[9:9 + n], ndmin=2)
    return data[:, 0].astype(int), data[:, 1:4], box


def _log_value(log: Path, key: str) -> float:
    m = re.findall(rf"^{key}\s+([-\d.eE+]+)", log.read_text(), flags=re.M)
    if not m:
        raise RuntimeError(f"{key} not found in {log}")
    return float(m[-1])


def prepare_perfect(outdir: Path, pot_abs: str, n: int, a0_guess: float,
                    lammps_cmd: str) -> dict:
    """Stage 0: relaxed perfect n×n×n diamond lattice (0 K, zero pressure)."""
    wd = outdir / "prep" / "perfect"
    done = wd / "perfect.json"
    if done.exists():
        return json.loads(done.read_text())
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "in.lmp").write_text(f"""# Threshold displacement — perfect lattice, box/relax at 0 K
units       metal
boundary    p p p
atom_style  atomic
lattice     diamond {a0_guess}
region      box block 0 {n} 0 {n} 0 {n} units lattice
create_box  1 box
create_atoms 1 box
mass        1 {GE_MASS}
{_ff_block(pot_abs)}
neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes
thermo_style custom step pe press lx
fix         br all box/relax iso 0.0 vmax 0.001
min_style   cg
minimize    1.0e-12 1.0e-8 10000 100000
unfix       br
minimize    1.0e-12 1.0e-8 10000 100000
print       "PE_PERFECT $(pe:%.10f)"
print       "LX_PERFECT $(lx:%.10f)"
write_data  perfect.data
write_dump  all custom perfect.dump id x y z modify sort id
""")
    _run_lammps(lammps_cmd, wd)
    info = {
        "pe": _log_value(wd / "log.lammps", "PE_PERFECT"),
        "a0": _log_value(wd / "log.lammps", "LX_PERFECT") / n,
        "n_atoms": 8 * n ** 3,
    }
    done.write_text(json.dumps(info, indent=2))
    return info


def prepare_equil(outdir: Path, pot_abs: str, T: float, t_equil: float,
                  seed: int, rep: int, lammps_cmd: str) -> Path:
    """Stage 1: thermalised start structure for one repeat (T > 0)."""
    perfect = (outdir / "prep" / "perfect" / "perfect.data").resolve()
    if T <= 0:
        return perfect
    wd = outdir / "prep" / f"equil_r{rep:02d}"
    out = wd / "equil.data"
    if out.exists():
        return out.resolve()
    wd.mkdir(parents=True, exist_ok=True)
    dt = 0.002
    (wd / "in.lmp").write_text(f"""# Threshold displacement — NVT thermalisation, repeat {rep}
units       metal
boundary    p p p
atom_style  atomic
read_data   {perfect}
mass        1 {GE_MASS}
{_ff_block(pot_abs)}
neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes
timestep    {dt}
thermo      100
thermo_style custom step temp pe ke press
velocity    all create {2.0 * T} {seed} mom yes rot yes dist gaussian
fix         nvt all nvt temp {T} {T} 0.1
run         {max(1, int(round(t_equil / dt)))}
write_data  equil.tmp.data
""")
    _run_lammps(lammps_cmd, wd)
    (wd / "equil.tmp.data").replace(out)   # atomic: never read a half-written file
    return out.resolve()


def write_recoil_input(wd: Path, pot_abs: str, start_data: Path, pka_id: int,
                       vel: np.ndarray, T: float, border: float, seed: int,
                       t_md: float, dt_max: float, xmax: float, ftol: float) -> None:
    border_block = ""
    if border > 0:
        b = border
        border_block = f"""region      inner block $(xlo+{b}) $(xhi-{b}) $(ylo+{b}) $(yhi-{b}) $(zlo+{b}) $(zhi-{b}) units box
group       inner region inner
group       border subtract all inner
fix         lang border langevin {T} {T} 0.1 {seed} zero yes
"""
    wd.joinpath("in.lmp").write_text(f"""# Threshold displacement — single recoil run
units       metal
boundary    p p p
atom_style  atomic
read_data   {start_data}
mass        1 {GE_MASS}
{_ff_block(pot_abs)}
neighbor    1.0 bin
neigh_modify delay 0 every 1 check yes
timestep    {dt_max}

group       PKA id {pka_id}
velocity    PKA set {vel[0]:.8f} {vel[1]:.8f} {vel[2]:.8f} units box
{border_block}fix         integ all nve
fix         dtr all dt/reset 1 NULL {dt_max} {xmax} units box
variable    t equal time
fix         halt all halt 20 v_t > {t_md} error continue
thermo      200
thermo_style custom step dt time temp pe ke
run         100000000
unfix       halt
unfix       dtr
unfix       integ
{"unfix       lang" if border > 0 else ""}
print       "MD_STEPS $(step)"
print       "MD_TIME $(time:%.6f)"
min_style   cg
minimize    0.0 {ftol} 5000 50000
print       "PE_FINAL $(pe:%.10f)"
write_dump  all custom final.dump id x y z modify sort id
""")


# ---------------------------------------------------------------------------
# Wigner-Seitz analysis
# ---------------------------------------------------------------------------

def wigner_seitz(sites: np.ndarray, pos: np.ndarray, box: np.ndarray) -> dict:
    """Wigner-Seitz occupancy in an orthogonal periodic box.

    sites: perfect-lattice site positions (N,3); atom i started on sites[i]
    pos:   relaxed final positions (N,3), same atom order as sites
    The rigid drift of the whole crystal (PKA momentum, border thermostat) is
    removed first, using the median minimum-image displacement of all atoms.
    """
    from scipy.spatial import cKDTree
    lo, L = box[:, 0], box[:, 1] - box[:, 0]
    d = pos - sites
    d -= L * np.round(d / L)
    shift = np.median(d, axis=0)
    p = np.mod(pos - shift - lo, L)
    s = np.mod(sites - lo, L)
    p = np.where(p >= L, p - L, p)      # np.mod can return L for tiny negatives
    s = np.where(s >= L, s - L, s)
    _, idx = cKDTree(s, boxsize=L).query(p)
    occ = np.bincount(idx, minlength=len(s))
    return {
        "n_vac": int(np.sum(occ == 0)),
        "n_int": int(np.sum(np.clip(occ - 1, 0, None))),
        # atoms not on their own starting site (replacements + interstitials)
        "n_displaced": int(np.sum(idx != np.arange(len(s)))),
        "drift": round(float(np.linalg.norm(shift)), 4),
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    outdir: Path
    pot_abs: str
    lammps: str
    sites: np.ndarray
    site_ids: np.ndarray
    pe_perfect: float
    pka_id: int
    T: float
    border: float
    t_md: float
    dt_max: float
    xmax: float
    ftol: float
    seed: int
    starts: dict = field(default_factory=dict)   # rep -> start data path


def run_one(ctx: Ctx, label: str, vec: np.ndarray, rep: int, E: float) -> dict:
    wd = ctx.outdir / "runs" / f"{label}_r{rep:02d}" / f"E_{E:07.2f}"
    res_file = wd / "result.json"
    if res_file.exists():
        return json.loads(res_file.read_text())
    wd.mkdir(parents=True, exist_ok=True)
    speed = math.sqrt(2.0 * E / GE_MASS * EV_PER_AMU_TO_A2_PS2)   # Å/ps
    write_recoil_input(wd, ctx.pot_abs, ctx.starts[rep], ctx.pka_id, speed * vec,
                       ctx.T, ctx.border, ctx.seed + 1000 * rep + 7,
                       ctx.t_md, ctx.dt_max, ctx.xmax, ctx.ftol)
    t0 = time.time()
    _run_lammps(ctx.lammps, wd)
    ids, pos, box = _read_dump(wd / "final.dump")
    if not np.array_equal(ids, ctx.site_ids):
        raise RuntimeError(f"atom ids changed in {wd} (lost atoms?)")
    ws = wigner_seitz(ctx.sites, pos, box)
    log = wd / "log.lammps"
    res = {
        "direction": label, "repeat": rep, "E": E,
        **ws,
        "defect": ws["n_vac"] > 0,
        "dE_relaxed": _log_value(log, "PE_FINAL") - ctx.pe_perfect,
        "md_steps": int(_log_value(log, "MD_STEPS")),
        "md_time_ps": _log_value(log, "MD_TIME"),
        "wall_s": round(time.time() - t0, 1),
    }
    res_file.write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# Scan driver (per (direction, repeat) state machine on a shared pool)
# ---------------------------------------------------------------------------

class Series:
    """Energy scan + bisection for one (direction, repeat)."""

    def __init__(self, label, vec, rep, energies, batch, refine):
        self.label, self.vec, self.rep = label, vec, rep
        self.queue = list(energies)          # coarse energies, ascending
        self.batch, self.refine = batch, refine
        self.results: dict[float, dict] = {}
        self.inflight: set[float] = set()
        self.hit: float | None = None        # lowest coarse E with a defect
        self.phase = "coarse"
        self.lo = self.hi = None
        self.bisect_job: float | None = None  # energy of the running bisection step

    def next_jobs(self) -> list[float]:
        jobs = []
        if self.phase == "coarse":
            while (self.queue and len(self.inflight) + len(jobs) < self.batch
                   and (self.hit is None or self.queue[0] < self.hit)):
                jobs.append(self.queue.pop(0))
        elif self.phase == "refine" and self.bisect_job is None:
            if self.hi - self.lo > self.refine + 1e-9:
                mid = round(0.5 * (self.lo + self.hi), 4)
                if mid in self.results:          # already known (e.g. from coarse grid)
                    self.update(mid, self.results[mid], bisect=True)
                    return self.next_jobs()
                self.bisect_job = mid
                jobs.append(mid)
            else:
                self.phase = "done"
        return jobs

    def update(self, E: float, res: dict, bisect: bool = False) -> None:
        self.inflight.discard(E)
        self.results[E] = res
        if E == self.bisect_job:
            self.bisect_job, bisect = None, True
        if self.phase == "coarse":
            if res["defect"] and (self.hit is None or E < self.hit):
                self.hit = E
            pending_below = any(e < (self.hit if self.hit is not None else math.inf)
                                for e in self.inflight)
            if self.hit is not None and not pending_below:
                clean = [e for e, r in self.results.items() if e < self.hit and not r["defect"]]
                self.coarse_ed = self.hit
                if clean and self.refine > 0:
                    self.lo, self.hi, self.phase = max(clean), self.hit, "refine"
                else:
                    self.lo = max(clean) if clean else None
                    self.hi, self.phase = self.hit, "done"
            elif self.hit is None and not self.queue and not self.inflight:
                self.coarse_ed, self.phase = math.nan, "done"
        elif self.phase == "refine" and bisect:
            # late coarse results above the hit (batch > 1) are only recorded
            if res["defect"]:
                self.hi = E
            else:
                self.lo = E

    @property
    def ed(self) -> float:
        return self.hi if self.hi is not None else math.nan


def run_scan(ctx: Ctx, series: list[Series], workers: int) -> None:
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}

        def feed():
            for s in series:
                for E in s.next_jobs():
                    s.inflight.add(E)
                    futs[pool.submit(run_one, ctx, s.label, s.vec, s.rep, E)] = (s, E)

        feed()
        while futs:
            done, _ = wait(list(futs), return_when=FIRST_COMPLETED)
            for f in done:
                s, E = futs.pop(f)
                try:
                    res = f.result()
                except Exception as exc:   # treat a crashed run as "no result"
                    print(f"  [{s.label} r{s.rep}] E={E:.2f} eV FAILED: {exc}", flush=True)
                    res = {"direction": s.label, "repeat": s.rep, "E": E, "defect": False,
                           "n_vac": -1, "n_int": -1, "n_displaced": -1, "failed": True}
                with lock:
                    s.update(E, res)
                print(f"  [{s.label:>7s} r{s.rep:02d}] E={E:7.2f} eV  vac={res['n_vac']:3d} "
                      f"int={res['n_int']:3d} displaced={res['n_displaced']:3d}"
                      + (f"  dE={res['dE_relaxed']:.3f} eV  {res['md_steps']} steps "
                         f"{res['wall_s']} s" if "dE_relaxed" in res else ""), flush=True)
            feed()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(outdir: Path, series: list[Series], meta: dict) -> None:
    # all runs
    rows = sorted((r for s in series for r in s.results.values()),
                  key=lambda r: (r["direction"], r["repeat"], r["E"]))
    keys = ["direction", "repeat", "E", "defect", "n_vac", "n_int", "n_displaced",
            "dE_relaxed", "md_steps", "md_time_ps", "drift", "wall_s"]
    with open(outdir / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    labels = list(dict.fromkeys(s.label for s in series))
    per_dir = {}
    for lab in labels:
        ss = [s for s in series if s.label == lab]
        eds = np.array([s.ed for s in ss])
        res_ = [(s.hi - s.lo) if (s.lo is not None and s.hi is not None) else math.nan
                for s in ss]
        thr = [s.results.get(s.hi, {}) if s.hi is not None else {} for s in ss]
        per_dir[lab] = {
            "vector": [round(float(x), 6) for x in ss[0].vec],
            "Ed_repeats": [None if math.isnan(e) else float(e) for e in eds],
            "Ed_mean": float(np.nanmean(eds)) if np.any(~np.isnan(eds)) else math.nan,
            "Ed_std": float(np.nanstd(eds)) if np.sum(~np.isnan(eds)) > 1 else math.nan,
            "resolution": float(np.nanmax(res_)) if np.any(~np.isnan(res_)) else math.nan,
            "n_FP_at_Ed": [t.get("n_vac") for t in thr],
            "dE_relaxed_at_Ed": [t.get("dE_relaxed") for t in thr],
        }
    all_eds = np.array([e if e is not None else math.nan
                        for d in per_dir.values() for e in d["Ed_repeats"]], dtype=float)
    lattice = [lab for lab in labels if not lab.startswith("rand")]
    rand = [lab for lab in labels if lab.startswith("rand")]

    def _avg(labs):
        v = np.array([per_dir[l]["Ed_mean"] for l in labs], dtype=float)
        return (float(np.nanmean(v)), int(np.sum(~np.isnan(v)))) if len(v) and np.any(~np.isnan(v)) \
            else (math.nan, 0)

    m_lat, n_lat = _avg(lattice)
    m_rand, n_rand = _avg(rand)

    def fmt(x, w=8, p=2):
        return f"{x:{w}.{p}f}" if x is not None and not (isinstance(x, float) and math.isnan(x)) \
            else f"{'n/a':>{w}s}"

    lines = [
        "Threshold displacement energy (Ed) — Ge diamond, recoil MD + Wigner-Seitz",
        f"Potential   : {meta['pot']}",
        f"Supercell   : {meta['supercell']}^3 conventional cells = {meta['n_atoms']} atoms, "
        f"a0 (0 K relaxed) = {meta['a0']:.5f} Å",
        f"PKA         : atom id {meta['pka_id']} (sublattice A, nearest to box centre)",
        f"T           : {meta['T']} K   repeats: {meta['repeats']}   border thermostat: "
        f"{meta['border']} Å",
        f"MD          : NVE + dt/reset (dt_max {meta['dt_max']} ps, xmax {meta['xmax']} Å), "
        f"{meta['t_md']} ps, then CG (ftol {meta['ftol']} eV/Å)",
        f"Scan        : {meta['energies_desc']}; bisection to {meta['refine']} eV"
        if meta["refine"] > 0 else f"Scan        : {meta['energies_desc']}; no refinement",
        "Ed = lowest recoil energy leaving >= 1 vacancy after relaxation. "
        "'n/a' = no defect up to E_max.",
        "Reference   : TODO (DFT/GAP to be computed)",
        "",
        f"{'dir':>8s} {'Ed_mean':>8s} {'Ed_std':>8s} {'res':>6s}  {'Ed per repeat':<28s} "
        f"{'nFP@Ed':<10s} {'DFT_ref':>8s} {'GAP_ref':>8s}",
    ]
    for lab, d in per_dir.items():
        reps = " ".join(fmt(e, 6, 2).strip() for e in d["Ed_repeats"])
        nfp = ",".join(str(x) for x in d["n_FP_at_Ed"])
        lines.append(f"{lab:>8s} {fmt(d['Ed_mean'])} {fmt(d['Ed_std'])} {fmt(d['resolution'], 6, 2)}"
                     f"  {reps:<28s} {nfp:<10s} {fmt(DFT_REF.get(lab, math.nan))} "
                     f"{fmt(GAP_REF.get(lab, math.nan))}")
    lines += [
        "",
        f"Mean over lattice directions ({n_lat}): {fmt(m_lat).strip()} eV",
        f"Mean over random directions  ({n_rand}): {fmt(m_rand).strip()} eV",
        f"Min Ed over all runs              : {fmt(float(np.nanmin(all_eds)) if np.any(~np.isnan(all_eds)) else math.nan).strip()} eV",
        "",
        "Note: n_vac counts vacancies (= Frenkel pairs) in the relaxed final cell; "
        "dE_relaxed = E_relaxed - E_perfect is in runs.csv.",
    ]
    text = "\n".join(lines) + "\n"
    (outdir / "threshold_displacement.txt").write_text(text)
    print("\n" + text)

    summary = {"meta": meta, "directions": per_dir,
               "mean_lattice_dirs": m_lat, "mean_random_dirs": m_rand,
               "DFT_REF": DFT_REF, "GAP_REF": GAP_REF}
    (outdir / "threshold_displacement.json").write_text(
        json.dumps(summary, indent=2, default=lambda x: None))
    plot(outdir, per_dir, m_lat, m_rand, meta)


def plot(outdir: Path, per_dir: dict, m_lat: float, m_rand: float, meta: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = list(per_dir)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.6 * len(labs) + 2), 4))
    for i, lab in enumerate(labs):
        d = per_dir[lab]
        reps = [e for e in d["Ed_repeats"] if e is not None]
        if reps:
            ax.scatter([i] * len(reps), reps, color="0.6", s=14, zorder=2)
        if not math.isnan(d["Ed_mean"]):
            ax.errorbar(i, d["Ed_mean"], yerr=0 if math.isnan(d["Ed_std"]) else d["Ed_std"],
                        fmt="o", color="C0", capsize=3, zorder=3)
        else:
            ax.annotate(f">{meta['e_max']:g}", (i, meta["e_max"]), ha="center",
                        va="bottom", fontsize=8, color="C3")
        ref = DFT_REF.get(lab)
        if ref is not None and not math.isnan(ref):
            ax.plot(i, ref, "k_", ms=14, mew=2, zorder=4)
    if not math.isnan(m_lat):
        ax.axhline(m_lat, color="C0", ls="--", lw=1, label=f"mean lattice dirs {m_lat:.1f} eV")
    if not math.isnan(m_rand):
        ax.axhline(m_rand, color="C1", ls=":", lw=1, label=f"mean random dirs {m_rand:.1f} eV")
    ax.set_xticks(x)
    ax.set_xlim(-0.6, len(labs) - 0.4)
    ax.set_xticklabels([f"[{l.replace('m', '-')}]" if not l.startswith("rand") else l
                        for l in labs], rotation=45 if len(labs) > 10 else 0, fontsize=8)
    ax.set_ylabel("$E_d$ (eV)")
    ax.set_title(f"Threshold displacement energy, Ge ({meta['n_atoms']} atoms, "
                 f"{meta['T']} K)", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "threshold_displacement.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_pka(ids: np.ndarray, pos: np.ndarray, box: np.ndarray, a0: float) -> int:
    """Sublattice-A atom (fractional (0,0,0) of the conventional cell) closest to the centre."""
    lo = box[:, 0]
    q = np.rint(4.0 * (pos - lo) / a0).astype(int) % 4
    on_a = np.all(q % 2 == 0, axis=1)
    centre = lo + 0.5 * (box[:, 1] - box[:, 0])
    d = np.linalg.norm(pos - centre, axis=1)
    d[~on_a] = np.inf
    return int(ids[int(np.argmin(d))])


def run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pot_abs = str(Path(args.pot).resolve())
    n = args.supercell

    T = max(0.0, args.temperature)
    repeats = args.repeats
    if T == 0 and repeats > 1:
        print("  T = 0 K: all repeats would be identical -> using --repeats 1")
        repeats = 1

    if args.energies:
        energies = sorted(float(e) for e in args.energies.split(","))
        e_desc = "E = " + ", ".join(f"{e:g}" for e in energies) + " eV"
    else:
        energies = [round(e, 4) for e in
                    np.arange(args.e_min, args.e_max + 1e-9, args.e_step)]
        e_desc = f"E = {args.e_min:g} → {args.e_max:g} eV in {args.e_step:g} eV steps"

    dirs = []
    if args.directions.strip():
        dirs += [parse_direction(t) for t in args.directions.split(",") if t.strip()]
    dirs += random_directions(args.n_random, args.random_seed)
    if not dirs:
        sys.exit("No directions given")

    print("=== Threshold displacement energy ===")
    print(f"  Potential : {args.pot}")
    print(f"  Supercell : {n}x{n}x{n} = {8 * n ** 3} atoms  |  T = {T} K  |  repeats = {repeats}")
    print(f"  Directions: {', '.join(l for l, _ in dirs)}")
    print(f"  Energies  : {e_desc}")
    print(f"  LAMMPS    : {args.lammps}  |  workers = {args.workers}  batch = {args.batch}")

    perfect = prepare_perfect(outdir, pot_abs, n, args.a0, args.lammps)
    ids, sites, box = _read_dump(outdir / "prep" / "perfect" / "perfect.dump")
    pka_id = pick_pka(ids, sites, box, perfect["a0"])
    print(f"  a0 (relaxed) = {perfect['a0']:.5f} Å   E_perfect = {perfect['pe']:.4f} eV   "
          f"PKA id = {pka_id} at {sites[ids == pka_id][0].round(3)}")

    ctx = Ctx(outdir=outdir, pot_abs=pot_abs, lammps=args.lammps, sites=sites, site_ids=ids,
              pe_perfect=perfect["pe"], pka_id=pka_id, T=T, border=args.border,
              t_md=args.t_md, dt_max=args.dt_max, xmax=args.xmax, ftol=args.ftol,
              seed=args.seed)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, repeats))) as pool:
        starts = list(pool.map(
            lambda r: prepare_equil(outdir, pot_abs, T, args.t_equil, args.seed + r, r,
                                    args.lammps), range(repeats)))
    ctx.starts = dict(enumerate(starts))

    refine = 0.0 if args.no_refine else args.refine
    series = [Series(lab, vec, rep, energies, max(1, args.batch), refine)
              for lab, vec in dirs for rep in range(repeats)]
    t0 = time.time()
    run_scan(ctx, series, args.workers)
    print(f"  Scan finished in {time.time() - t0:.0f} s")

    meta = {
        "pot": args.pot, "supercell": n, "n_atoms": perfect["n_atoms"], "a0": perfect["a0"],
        "pe_perfect": perfect["pe"], "pka_id": pka_id, "T": T, "repeats": repeats,
        "border": args.border, "t_md": args.t_md, "dt_max": args.dt_max, "xmax": args.xmax,
        "ftol": args.ftol, "refine": refine, "energies_desc": e_desc,
        "e_max": max(energies), "seed": args.seed,
    }
    write_outputs(outdir, series, meta)


def main():
    p = argparse.ArgumentParser(
        description="Threshold displacement energy of Ge via recoil MD (LAMMPS mtp+nlh).")
    p.add_argument("--pot", required=True, help="Path to potential (.mtp/.almtp)")
    p.add_argument("--supercell", type=int, default=6,
                   help="Conventional diamond cells per edge (default 6 -> 1728 atoms)")
    p.add_argument("--a0", type=float, default=5.76,
                   help="Initial lattice constant guess [Å]; box is relaxed at 0 K (default 5.76)")
    p.add_argument("--temperature", "-T", type=float, default=10.0,
                   help="Lattice temperature [K]; 0 = perfect static lattice (default 10)")
    p.add_argument("--t-equil", type=float, default=2.0,
                   help="NVT thermalisation time per repeat [ps] (default 2.0)")
    p.add_argument("--repeats", type=int, default=1,
                   help="Independent thermal seeds per direction (T > 0 only; default 1)")
    p.add_argument("--seed", type=int, default=12345, help="Base random seed (default 12345)")
    p.add_argument("--directions", default=DEFAULT_DIRECTIONS,
                   help=f"Comma-separated Miller directions (default {DEFAULT_DIRECTIONS}); "
                        f"'' for none")
    p.add_argument("--n-random", type=int, default=0,
                   help="Additional uniformly random directions (default 0)")
    p.add_argument("--random-seed", type=int, default=2026,
                   help="Seed for random directions (default 2026)")
    p.add_argument("--e-min", type=float, default=4.0, help="Scan start [eV] (default 4)")
    p.add_argument("--e-max", type=float, default=60.0, help="Scan end [eV] (default 60)")
    p.add_argument("--e-step", type=float, default=2.0, help="Scan step [eV] (default 2)")
    p.add_argument("--energies", default=None,
                   help="Explicit comma-separated energy list [eV]; overrides --e-min/max/step")
    p.add_argument("--refine", type=float, default=0.5,
                   help="Bisection resolution [eV] after the coarse scan (default 0.5)")
    p.add_argument("--no-refine", action="store_true", help="Skip bisection refinement")
    p.add_argument("--t-md", type=float, default=3.0, help="Recoil MD time [ps] (default 3.0)")
    p.add_argument("--dt-max", type=float, default=0.002,
                   help="Max timestep for fix dt/reset [ps] (default 0.002)")
    p.add_argument("--xmax", type=float, default=0.05,
                   help="Max displacement per step for fix dt/reset [Å] (default 0.05)")
    p.add_argument("--border", type=float, default=0.0,
                   help="Thickness [Å] of a Langevin border shell at T (default 0 = pure NVE)")
    p.add_argument("--ftol", type=float, default=1e-4,
                   help="Force tolerance of the final CG minimisation [eV/Å] (default 1e-4)")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent LAMMPS runs (default 4)")
    p.add_argument("--batch", type=int, default=1,
                   help="Coarse-scan energies run concurrently per (direction, repeat); "
                        "1 = strictly sequential scan, larger = faster but wasted runs above Ed "
                        "(default 1)")
    p.add_argument("--lammps", default=DEFAULT_LAMMPS,
                   help="LAMMPS command, may include launcher, e.g. "
                        "'srun --exact -n 4 -c 1 lmp_mpi' (default: serial lmp_mpi)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default results/tests/<pot parent dir>/threshold_displacement)")
    args = p.parse_args()

    if args.outdir is None:
        args.outdir = str(Path("results/tests") / Path(args.pot).resolve().parent.name
                          / "threshold_displacement")
    run(args)


if __name__ == "__main__":
    main()
