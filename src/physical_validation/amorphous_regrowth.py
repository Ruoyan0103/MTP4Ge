"""Solid-phase epitaxial regrowth (SPER) of amorphous Ge on a (001), (110) or (111) diamond template.

Measures the velocity of the amorphous/crystal (a/c) interface along the slab normal
(--orient, default [001]) at several anneal temperatures, and an Arrhenius activation energy.

Protocol (LAMMPS only; mtp+nlh potentials via `pair_style hybrid/overlay mtp nlh`, or with
--backend lammps-sw the SW + NLH table, --pot = its directory, src/utils/lammps_pair.py):
  prep   (one run, <outdir>/prep/)
    1. Orthogonal diamond slab of nx × ny × nz repeat lengths, lattice constant --a0, z = --orient:
       001: [100]/[010]/[001], 6 × 6 × 24 = 6912 atoms (default);
       110: [-110]/[001]/[110], 8 × 6 × 34 = 6528 atoms;
       111: [1-10]/[11-2]/[111], 8 × 5 × 14 = 6720 atoms (all ≈ 35 × 35 × 138 Å at a0 = 5.76).
    2. Central half along z (Lz/4 < z < 3Lz/4) melted: NVT at --T-melt for
       --steps-melt, crystal halves frozen (setforce 0, zero velocity), box fixed.
    3. Central region quenched --T-melt → 300 K in --steps-quench (NVT), still
       with frozen crystal.
    4. Whole box released: NPT 300 K, 0 bar (--barostat) for --steps-relax.
       Written to prep/a_c_slab.lammps. The diamond fraction of the central
       half is checked with OVITO IdentifyDiamondModifier (should be ≈ 0).
  anneal (one independent run per temperature, <outdir>/T_<T>K/; parallelisable)
    5. NPT at T (default 800, 900, 1000 K), 0 bar, `aniso` (x, y, z independent)
       or `z` (only Lz barostatted, x/y kept at the prep lateral size), for
       --steps-anneal at --dt-anneal; dump every --dump-every steps (positions only).
    6. Every dump frame is classified with OVITO IdentifyDiamondModifier
       (crystalline = cubic or hexagonal diamond incl. 1st/2nd-neighbour
       variants). The crystalline fraction of the outer crystal (|z - zc| > 3Lz/8)
       f_ref(t) normalises out thermal misidentification:
           L_a(t) = (1 - f(t) / f_ref(t)) · Lz(t)     (amorphous thickness, Å)
       Two a/c interfaces → v = -½ dL_a/dt (linear fit), 1 Å/ps = 100 m/s.
       Fit window: t ≥ --fit-skip-ps and L_a ≥ --fit-min-frac · L_a(0).
       Positive v = regrowth; negative v = the crystal amorphises/melts
       (expected above the MTP Tm). Interface positions z_lo, z_hi are also
       located from the z-profile of the normalised crystalline fraction
       (0.5 crossing, bins of a0/2) and written for diagnostics.
       Outputs: T_<T>K/fraction.dat, profile.dat, velocity.txt.
  analyze (<outdir>/regrowth_summary.txt, regrowth.png)
    7. Collect all T_<T>K/fraction.dat, refit, Arrhenius fit
       ln v = ln v0 - E_a / (k_B T) over temperatures with v > 0 (needs ≥ 3).

Caveats: MD regrowth is only observable at high T (close to Tm), where random
nucleation in the amorphous layer can compete with epitaxial regrowth — check
profile.dat / the dumps: nucleation shows up as crystalline bins inside the
amorphous layer rather than motion of the two interfaces. The amorphous layer
is made by a fast quench of the melt next to the template (no a-Ge relaxation
anneal), so its state depends on --steps-quench.

References:
  MTP Tm (two-phase coexistence, melting_point.py) ≈ 960 K; experimental Ge
  Tm = 1211 K. Experimental SPER velocity / activation energy of Ge(001):
  reference: TODO.

Usage:
    # Everything sequentially (prep, all temperatures, analysis):
    python src/physical_validation/amorphous_regrowth.py --pot results/potentials/pot_1019/bal.mtp

    # Split (as scripts/submit_amorphous_regrowth.sh does; anneals in parallel):
    python src/physical_validation/amorphous_regrowth.py --pot POT --stage prep --np 32
    python src/physical_validation/amorphous_regrowth.py --pot POT --stage anneal --T 800 --np 32 &
    python src/physical_validation/amorphous_regrowth.py --pot POT --stage anneal --T 900 --np 32 &
    wait
    python src/physical_validation/amorphous_regrowth.py --pot POT --stage analyze

    sbatch scripts/submit_amorphous_regrowth.sh results/potentials/pot_1019/bal.mtp
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from energy_volume import DEFAULT_LAMMPS
import lammps_pair

_GE_MASS_AMU = 72.630
_KB_EV = 8.617333262e-5          # eV/K
_APS_TO_MS = 100.0               # 1 Å/ps = 100 m/s
MTP_TM_K = 960.0   # pot_1019/bal Tm = 963 ± 4 K (results/tests/pot_1019/bal/melting_point); messages only

PAIR_BLOCK = """pair_style  hybrid/overlay mtp nlh
pair_coeff  * * mtp {pot} 0
pair_coeff  1 1 nlh 32 32 0.05121 0.50355 0.44524 49.82156 12.56998 4.13684 1.28 2.0"""


def _pair_block(pot: str) -> str:
    """mtp+nlh block, or the SW block if `pot` is the SW directory (lammps_pair.apply)."""
    return lammps_pair.apply(PAIR_BLOCK.format(pot=pot) + "\n", pot).rstrip("\n")

_HEADER = """units       metal
boundary    p p p
atom_style  atomic"""

_NEIGH = """neighbor    2.0 bin
neigh_modify delay 0 every 1 check yes"""


# Slab orientations: LAMMPS orient vectors (x, y, z = growth direction), repeat length of each
# axis in units of a0, atoms per repeat cell, default repeats (≈ 35 × 35 × 138 Å, as 6 × 6 × 24 for 001).
ORIENTS = {
    "001": {"orient": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "period": (1.0, 1.0, 1.0), "n_cell": 8, "reps": (6, 6, 24)},
    "110": {"orient": ((-1, 1, 0), (0, 0, 1), (1, 1, 0)),
            "period": (2 ** -0.5, 1.0, 2 ** -0.5), "n_cell": 4, "reps": (8, 6, 34)},
    "111": {"orient": ((1, -1, 0), (1, 1, -2), (1, 1, 1)),
            "period": (2 ** -0.5, 1.5 ** 0.5, 3 ** 0.5), "n_cell": 12, "reps": (8, 5, 14)},
}


def _baro(barostat: str, pdamp: float) -> str:
    if barostat == "aniso":
        return f"aniso 0.0 0.0 {pdamp}"
    if barostat == "z":
        return f"z 0.0 0.0 {pdamp}"
    raise ValueError(f"unknown barostat {barostat!r}")


# ---------------------------------------------------------------------------
# LAMMPS input writers
# ---------------------------------------------------------------------------


def write_prep_input(workdir: Path, pot: str, a0: float, nx: int, ny: int, nz: int,
                     T_melt: float, steps_melt: int, steps_quench: int, steps_relax: int,
                     barostat: str, seed: int, orient: str = "001",
                     thermo_every: int = 500) -> dict:
    """Build the slab (growth direction = z = `orient`) and amorphise its central half.
    nx, ny, nz count repeat lengths of the oriented axes (ORIENTS). Returns geometry info."""
    o = ORIENTS[orient]
    Lx, Ly, Lz = (n * p * a0 for n, p in zip((nx, ny, nz), o["period"]))
    z_lo, z_hi = 0.25 * Lz, 0.75 * Lz
    ovec = " ".join(f"orient {ax} {' '.join(map(str, v))}" for ax, v in zip("xyz", o["orient"]))
    text = f"""# a/c slab preparation ({orient}) — generated by amorphous_regrowth.py
{_HEADER}

lattice     diamond {a0} {ovec}
region      box block 0 {Lx:.6f} 0 {Ly:.6f} 0 {Lz:.6f} units box
create_box  1 box
create_atoms 1 box
mass        1 {_GE_MASS_AMU}

{_pair_block(pot)}
{_NEIGH}
timestep    0.001
thermo      {thermo_every}

region      amor block INF INF INF INF {z_lo:.6f} {z_hi:.6f} units box
group       amor region amor
group       xtal subtract all amor
compute     Tamor amor temp
thermo_style custom step time c_Tamor temp pe press lx ly lz

# ── melt the central half, crystal halves frozen, box fixed ──
velocity    xtal set 0.0 0.0 0.0
fix         freeze xtal setforce 0.0 0.0 0.0
velocity    amor create {T_melt} {seed} mom yes rot no dist gaussian
fix         melt amor nvt temp {T_melt} {T_melt} 0.1
run         {steps_melt}
unfix       melt

# ── quench the central region {T_melt:.0f} K -> 300 K ──
fix         quench amor nvt temp {T_melt} 300.0 0.1
run         {steps_quench}
unfix       quench
unfix       freeze

# ── release everything: NPT 300 K, 0 bar ──
velocity    all create 300.0 {seed + 1} mom yes rot yes dist gaussian
fix         relax all npt temp 300.0 300.0 0.1 {_baro(barostat, 1.0)}
run         {steps_relax}
unfix       relax

write_data  a_c_slab.lammps nocoeff
"""
    (workdir / "in.prep").write_text(text, encoding="utf-8")
    return {"a0": a0, "orient": orient, "nx": nx, "ny": ny, "nz": nz, "Lz": Lz,
            "z_amor_lo": z_lo, "z_amor_hi": z_hi,
            "n_atoms": o["n_cell"] * nx * ny * nz, "T_melt": T_melt, "steps_melt": steps_melt,
            "steps_quench": steps_quench, "steps_relax": steps_relax, "barostat": barostat}


def write_anneal_input(workdir: Path, pot: str, slab: Path, T: float, steps: int, dt: float,
                       dump_every: int, barostat: str, seed: int, thermo_every: int) -> None:
    text = f"""# a/c regrowth anneal at {T:.0f} K — generated by amorphous_regrowth.py
{_HEADER}

read_data   {slab}
mass        1 {_GE_MASS_AMU}
{_pair_block(pot)}
{_NEIGH}
timestep    {dt}
thermo      {thermo_every}
thermo_style custom step time temp pe press lx ly lz

velocity    all create {T} {seed} mom yes rot yes dist gaussian
fix         anneal all npt temp {T} {T} {100 * dt} {_baro(barostat, 1000 * dt)}
dump        dmp all custom {dump_every} dump_anneal.lammpstrj id type x y z
dump_modify dmp sort id
run         {steps}
write_data  final.lammps nocoeff
"""
    (workdir / "in.anneal").write_text(text, encoding="utf-8")


def _run_lammps(lammps: str, np_cores: int, workdir: Path, in_name: str, log_name: str) -> float:
    """Run LAMMPS in workdir; `srun ...` launchers get `-n np`, else mpirun if np > 1."""
    tokens = shlex.split(lammps)
    if tokens and tokens[0] == "srun":
        cmd = [tokens[0], "-n", str(np_cores)] + tokens[1:]
    elif np_cores > 1:
        cmd = ["mpirun", "-np", str(np_cores)] + tokens
    else:
        cmd = tokens
    cmd += ["-in", in_name, "-log", log_name]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if res.returncode != 0:
        print("  LAMMPS stderr (last 600 chars):")
        print(res.stderr[-600:] or res.stdout[-600:])
        raise RuntimeError(f"LAMMPS failed in {workdir} (see {log_name})")
    return time.time() - t0


# ---------------------------------------------------------------------------
# Structure identification (OVITO)
# ---------------------------------------------------------------------------


def _diamond_flags(path: Path):
    """Yield (timestep, cell lengths (3,), z origin, z positions, crystalline bool array)."""
    from ovito.io import import_file
    from ovito.modifiers import IdentifyDiamondModifier

    pipeline = import_file(str(path))
    pipeline.modifiers.append(IdentifyDiamondModifier())
    for frame in range(pipeline.source.num_frames):
        data = pipeline.compute(frame)
        st = np.asarray(data.particles["Structure Type"])
        xtal = st != IdentifyDiamondModifier.Type.OTHER
        cell = np.asarray(data.cell[...])
        lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
        z = np.asarray(data.particles.positions[:, 2]) - cell[2, 3]
        z = np.mod(z, lengths[2])
        step = int(data.attributes.get("Timestep", frame))
        yield step, lengths, z, xtal


def _profile_interfaces(prof: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Locate the two a/c interfaces from a normalised crystalline profile.

    Amorphous bins: prof < 0.5. Takes the longest periodic run of amorphous
    bins; returns (z_lo, z_hi) at the linearly interpolated 0.5 crossings.
    NaN if no amorphous bin (or no crystalline bin) is present.
    """
    n = len(prof)
    amor = prof < 0.5
    if amor.all() or not amor.any():
        return float("nan"), float("nan")
    start = int(np.argmin(amor))           # a crystalline bin: unroll from here
    order = (np.arange(n) + start) % n
    best, cur, best_end = 0, 0, -1
    for k, i in enumerate(order):
        cur = cur + 1 if amor[i] else 0
        if cur > best:
            best, best_end = cur, k
    first_k, last_k = best_end - best + 1, best_end
    w = edges[1] - edges[0]
    centres = edges[:-1] + 0.5 * w
    Lz = edges[-1] - edges[0]

    def cross(k_x, k_a):        # crystalline bin k_x next to amorphous bin k_a
        i_x, i_a = order[k_x % n], order[k_a % n]
        p_x, p_a = prof[i_x], prof[i_a]
        frac = (p_x - 0.5) / (p_x - p_a) if p_x != p_a else 0.5
        z_x = centres[i_x]
        z_a = centres[i_a]
        dz = (z_a - z_x + 0.5 * Lz) % Lz - 0.5 * Lz     # minimum image, ±w
        return (z_x + frac * dz) % Lz

    return cross(first_k - 1, first_k), cross(last_k + 1, last_k)


def analyze_dump(dump: Path, dt_ps: float, out_dir: Path, nbins: int) -> np.ndarray:
    """Crystalline-fraction time series → fraction.dat, profile.dat. Returns the table."""
    rows, prof_rows = [], []
    for step, L, z, xtal in _diamond_flags(dump):
        Lz = L[2]
        # Outer crystal reference: farther than 3Lz/8 from the slab centre (z = Lz/2).
        outer = np.abs(z - 0.5 * Lz) > 0.375 * Lz
        f = xtal.mean()
        f_ref = xtal[outer].mean() if outer.any() else 1.0
        f_norm = min(f / f_ref, 1.0) if f_ref > 0 else 0.0
        L_a = (1.0 - f_norm) * Lz
        edges = np.linspace(0.0, Lz, nbins + 1)
        n_all, _ = np.histogram(z, bins=edges)
        n_x, _ = np.histogram(z[xtal], bins=edges)
        prof = np.divide(n_x, n_all, out=np.zeros(nbins), where=n_all > 0) / max(f_ref, 1e-9)
        z_lo, z_hi = _profile_interfaces(prof, edges)
        rows.append([step * dt_ps, f, f_ref, L_a, Lz, z_lo, z_hi])
        prof_rows.append(np.concatenate([[step * dt_ps], prof]))
    tab = np.array(rows)
    np.savetxt(out_dir / "fraction.dat", tab, fmt="%.6f", encoding="utf-8",
               header="t_ps  f_xtal  f_xtal_outer_ref  L_amor_A  Lz_A  z_if_lo_A  z_if_hi_A\n"
                      "f_xtal: IdentifyDiamond (cubic+hex, incl. neighbour variants); "
                      "L_amor = (1 - f/f_ref)*Lz; z_if from the 0.5 crossing of profile.dat")
    np.savetxt(out_dir / "profile.dat", np.array(prof_rows), fmt="%.4f", encoding="utf-8",
               header=f"t_ps  then {nbins} bins along z (0..Lz) of crystalline fraction / f_ref")
    return tab


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_velocity(tab: np.ndarray, skip_ps: float, min_frac: float) -> dict:
    """Linear fit of L_a(t): v = -½ dL_a/dt. Also per-interface fits from z_if."""
    t, L_a = tab[:, 0] - tab[0, 0], tab[:, 3]
    L0 = L_a[0]
    mask = (t >= skip_ps) & (L_a >= min_frac * L0)
    out = {"L_a0_A": float(L0), "n_fit": int(mask.sum()), "v_ms": float("nan"),
           "v_err_ms": float("nan"), "t_fit_min": float("nan"), "t_fit_max": float("nan"),
           "v_lo_ms": float("nan"), "v_hi_ms": float("nan")}
    if mask.sum() < 3:
        return out
    tt, LL = t[mask], L_a[mask]
    (slope, icpt), cov = np.polyfit(tt, LL, 1, cov=True)
    out.update(v_ms=-0.5 * slope * _APS_TO_MS, v_err_ms=0.5 * np.sqrt(cov[0, 0]) * _APS_TO_MS,
               t_fit_min=float(tt[0]), t_fit_max=float(tt[-1]), slope_A_ps=float(slope),
               icpt_A=float(icpt))
    # Per-interface (diagnostic): the lower interface moves +z, the upper −z on regrowth.
    for col, key, sign in ((5, "v_lo_ms", 1.0), (6, "v_hi_ms", -1.0)):
        z = tab[mask, col]
        ok = np.isfinite(z)
        if ok.sum() >= 3:
            out[key] = sign * np.polyfit(tt[ok], np.unwrap(z[ok], period=tab[0, 4]), 1)[0] * _APS_TO_MS
    return out


def fit_arrhenius(Ts: list[float], vs: list[float]) -> dict | None:
    pts = [(T, v) for T, v in zip(Ts, vs) if np.isfinite(v) and v > 0]
    if len(pts) < 3:
        return None
    T = np.array([p[0] for p in pts])
    v = np.array([p[1] for p in pts])
    x = 1.0 / (_KB_EV * T)
    (slope, icpt), cov = np.polyfit(x, np.log(v), 1, cov=True)
    return {"E_a_eV": -slope, "E_a_err_eV": float(np.sqrt(cov[0, 0])),
            "v0_ms": float(np.exp(icpt)), "T_used": T.tolist()}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _tdir(outdir: Path, T: float) -> Path:
    return outdir / f"T_{T:.0f}K"


def stage_prep(pot: str, outdir: Path, lammps: str, np_cores: int, a0: float,
               nx: int, ny: int, nz: int, T_melt: float, steps_melt: int,
               steps_quench: int, steps_relax: int, barostat: str, seed: int,
               orient: str = "001") -> Path:
    workdir = outdir / "prep"
    workdir.mkdir(parents=True, exist_ok=True)
    info = write_prep_input(workdir, str(Path(pot).resolve()), a0, nx, ny, nz, T_melt,
                            steps_melt, steps_quench, steps_relax, barostat, seed, orient)
    print(f"  prep ({orient}): {info['n_atoms']} atoms, {nx}x{ny}x{nz} cells, amorphise "
          f"{info['z_amor_lo']:.1f} < z < {info['z_amor_hi']:.1f} Å at {T_melt:.0f} K")
    wall = _run_lammps(lammps, np_cores, workdir, "in.prep", "prep.log")
    slab = workdir / "a_c_slab.lammps"

    # Check: central half must be (nearly) non-crystalline after the quench.
    _, L, z, xtal = next(_diamond_flags(slab))
    if len(xtal) != info["n_atoms"]:     # oriented box not commensurate with the lattice
        raise RuntimeError(f"{slab}: {len(xtal)} atoms, expected {info['n_atoms']}")
    centre = np.abs(z - 0.5 * L[2]) < 0.25 * L[2] - 0.5 * a0     # away from the interfaces
    outer = np.abs(z - 0.5 * L[2]) > 0.25 * L[2] + 0.5 * a0
    info.update(pot=str(pot), f_xtal_centre=float(xtal[centre].mean()),
                f_xtal_outer=float(xtal[outer].mean()), wall_s=wall)
    (workdir / "slab_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  prep done ({wall:.0f} s): diamond fraction centre = {info['f_xtal_centre']:.3f}, "
          f"outer crystal = {info['f_xtal_outer']:.3f}")
    if info["f_xtal_centre"] > 0.2:
        print("  WARNING: the central region is > 20 % diamond after the quench — it did not "
              "amorphise (raise --T-melt / --steps-melt, or shorten --steps-quench).")
    return slab


def stage_anneal(pot: str, outdir: Path, lammps: str, np_cores: int, T: float, steps: int,
                 dt: float, dump_every: int, barostat: str, seed: int, nbins: int | None,
                 skip_ps: float, min_frac: float, delete_dump: bool) -> dict:
    slab = (outdir / "prep" / "a_c_slab.lammps").resolve()
    if not slab.exists():
        raise FileNotFoundError(f"{slab} missing — run --stage prep first")
    info = json.loads((outdir / "prep" / "slab_info.json").read_text())
    nbins = nbins or round(2 * info.get("Lz", info["nz"] * info["a0"]) / info["a0"])  # a0/2 bins
    workdir = _tdir(outdir, T)
    workdir.mkdir(parents=True, exist_ok=True)
    write_anneal_input(workdir, str(Path(pot).resolve()), slab, T, steps, dt, dump_every,
                       barostat, seed + int(T), thermo_every=max(dump_every // 5, 1))
    print(f"  anneal {T:.0f} K: {steps} steps x {dt} ps = {steps * dt:.1f} ps, "
          f"dump every {dump_every * dt:.2f} ps")
    wall = _run_lammps(lammps, np_cores, workdir, "in.anneal", "anneal.log")
    dump = workdir / "dump_anneal.lammpstrj"
    tab = analyze_dump(dump, dt, workdir, nbins)
    res = fit_velocity(tab, skip_ps, min_frac)
    res.update(T=T, wall_s=wall)
    _write_velocity(workdir / "velocity.txt", res)
    print(f"  {T:.0f} K ({wall:.0f} s): L_a {tab[0, 3]:.1f} -> {tab[-1, 3]:.1f} Å over "
          f"{tab[-1, 0] - tab[0, 0]:.1f} ps;  v = {res['v_ms']:.3g} ± {res['v_err_ms']:.2g} m/s "
          f"({res['n_fit']} points)")
    if delete_dump:
        dump.unlink()
    return res


def _write_velocity(path: Path, res: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# a/c regrowth velocity (v > 0: regrowth; 1 m/s = 1 nm/ns = 0.01 Å/ps)\n")
        for k, v in res.items():
            f.write(f"{k}  {v}\n")


def stage_analyze(outdir: Path, pot: str | None, skip_ps: float, min_frac: float,
                  tm: float = MTP_TM_K) -> None:
    tdirs = sorted(outdir.glob("T_*K"), key=lambda p: float(p.name[2:-1]))
    results = []
    for d in tdirs:
        frac = d / "fraction.dat"
        if not frac.exists():
            print(f"  skipping {d.name}: no fraction.dat")
            continue
        tab = np.loadtxt(frac, ndmin=2)
        res = fit_velocity(tab, skip_ps, min_frac)
        res["T"] = float(d.name[2:-1])
        res["tab"] = tab
        results.append(res)
    if not results:
        raise RuntimeError(f"no T_*K/fraction.dat under {outdir}")

    arr = fit_arrhenius([r["T"] for r in results], [r["v_ms"] for r in results])
    info_path = outdir / "prep" / "slab_info.json"
    orient = json.loads(info_path.read_text()).get("orient", "001") if info_path.exists() else "001"
    lines = [f"# Amorphous/crystal regrowth (SPER) along [{orient}]  |  pot: {pot}",
             f"# v = -1/2 dL_a/dt, fit t >= {skip_ps} ps and L_a >= {min_frac} L_a(0); "
             f"Tm ~ {tm:.0f} K; experimental reference: TODO",
             "# T_K   v_m/s   v_err_m/s   v_lo_if_m/s   v_hi_if_m/s   L_a0_A   L_a_end_A   "
             "t_end_ps   n_fit"]
    for r in results:
        tab = r["tab"]
        lines.append(f"{r['T']:6.0f}  {r['v_ms']:9.4g}  {r['v_err_ms']:9.3g}  {r['v_lo_ms']:9.4g}  "
                     f"{r['v_hi_ms']:9.4g}  {r['L_a0_A']:7.2f}  {tab[-1, 3]:7.2f}  "
                     f"{tab[-1, 0] - tab[0, 0]:8.1f}  {r['n_fit']:4d}")
    if arr is None:
        lines.append("# Arrhenius: needs >= 3 temperatures with v > 0 — not computed")
    else:
        lines.append(f"# Arrhenius (T = {arr['T_used']}): E_a = {arr['E_a_eV']:.3f} ± "
                     f"{arr['E_a_err_eV']:.3f} eV, v0 = {arr['v0_ms']:.3g} m/s  "
                     f"[reference: TODO]")
    txt = outdir / "regrowth_summary.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join("  " + l for l in lines))
    print(f"  Summary written to {txt}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncol = 2 if arr is not None else 1
        fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 4.5), squeeze=False)
        ax = axes[0, 0]
        for r in results:
            tab = r["tab"]
            t = tab[:, 0] - tab[0, 0]
            (line,) = ax.plot(t, tab[:, 3], "o-", ms=3, lw=1, label=f"{r['T']:.0f} K: "
                              f"v = {r['v_ms']:.3g} m/s")
            if np.isfinite(r["v_ms"]):
                tf = np.array([r["t_fit_min"], r["t_fit_max"]])
                ax.plot(tf, r["icpt_A"] + r["slope_A_ps"] * tf, "--",
                        color=line.get_color(), lw=1.5)
        ax.set_xlabel("anneal time (ps)")
        ax.set_ylabel("amorphous thickness $L_a$ (Å)")
        ax.set_title(f"a/c regrowth, Ge({orient})")
        ax.legend(fontsize=8)
        if arr is not None:
            ax2 = axes[0, 1]
            T = np.array([r["T"] for r in results if np.isfinite(r["v_ms"]) and r["v_ms"] > 0])
            v = np.array([r["v_ms"] for r in results if np.isfinite(r["v_ms"]) and r["v_ms"] > 0])
            x = 1.0 / (_KB_EV * T)
            ax2.semilogy(x, v, "o")
            xx = np.linspace(x.min(), x.max(), 50)
            ax2.semilogy(xx, arr["v0_ms"] * np.exp(-arr["E_a_eV"] * xx), "--",
                         label=f"$E_a$ = {arr['E_a_eV']:.2f} eV")
            ax2.set_xlabel("1 / (k$_B$T)  (1/eV)")
            ax2.set_ylabel("v (m/s)")
            ax2.legend()
        fig.tight_layout()
        png = outdir / "regrowth.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png}")
    except ImportError as e:
        print(f"  Matplotlib not available ({e}); skipping plot.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="a/c solid-phase epitaxial regrowth of Ge(001) "
                                            "(LAMMPS mtp+nlh + OVITO IdentifyDiamond)")
    p.add_argument("--pot", default=None, help="mtp+nlh potential (.mtp), or the SW directory with "
                                                "--backend lammps-sw; required except for --stage analyze")
    p.add_argument("--backend", choices=["lammps", "lammps-sw"], default="lammps",
                   help="'lammps' (default): pair_style mtp+nlh. 'lammps-sw': SW + NLH table, "
                        "--pot = its directory (src/utils/lammps_pair.py); ~4 GB per MPI rank")
    p.add_argument("--tm", type=float, default=MTP_TM_K,
                   help=f"Tm of the potential, K, for the T > Tm note only (default: {MTP_TM_K:.0f})")
    p.add_argument("--stage", choices=["all", "prep", "anneal", "analyze"], default="all",
                   help="all (default): prep, every --T, analyze, sequentially. 'anneal' runs "
                        "the --T list against an existing prep/ (one T per job for parallelism).")
    p.add_argument("--T", type=float, nargs="+", default=[800.0, 900.0, 1000.0],
                   help="Anneal temperatures in K (default: 800 900 1000; MTP Tm ~ 960 K)")
    p.add_argument("--orient", choices=sorted(ORIENTS), default="001",
                   help="growth direction = z: 001 (default; x [100], y [010]), 110 (x [-110], "
                        "y [001]) or 111 (x [1-10], y [11-2])")
    p.add_argument("--nx", type=int, default=None,
                   help="repeat lengths along x (default per --orient: 001 6, 110 8, 111 8)")
    p.add_argument("--ny", type=int, default=None,
                   help="repeat lengths along y (default per --orient: 001 6, 110 6, 111 5)")
    p.add_argument("--nz", type=int, default=None,
                   help="repeat lengths along z (default per --orient: 001 24, 110 34, 111 14; "
                        "repeat = a0, a0/sqrt2, a0*sqrt3)")
    p.add_argument("--a0", type=float, default=5.7609,
                   help="Diamond lattice constant (Å) for the slab (default: 5.7609, pot_1019 "
                        "0 K; the NPT stages relax the box)")
    p.add_argument("--T-melt", type=float, default=1800.0,
                   help="Melt temperature of the central region (default: 1800 K)")
    p.add_argument("--steps-melt", type=int, default=20_000, help="1 fs steps (default: 20 ps)")
    p.add_argument("--steps-quench", type=int, default=20_000,
                   help="1 fs steps for T_melt -> 300 K (default: 20 ps, ~7.5e13 K/s)")
    p.add_argument("--steps-relax", type=int, default=10_000,
                   help="1 fs steps of whole-box NPT at 300 K (default: 10 ps)")
    p.add_argument("--steps-anneal", type=int, default=500_000,
                   help="anneal steps (default: 500000 x 2 fs = 1 ns)")
    p.add_argument("--dt-anneal", type=float, default=0.002, help="anneal timestep, ps (0.002)")
    p.add_argument("--dump-every", type=int, default=5_000,
                   help="anneal dump interval in steps (default: 5000 = 10 ps)")
    p.add_argument("--barostat", choices=["aniso", "z"], default="aniso",
                   help="NPT coupling: aniso (default; x, y, z independent) or z (only Lz)")
    p.add_argument("--nbins", type=int, default=None,
                   help="z bins for profile.dat / interface positions (default: Lz / (a0/2))")
    p.add_argument("--fit-skip-ps", type=float, default=20.0,
                   help="ignore the first N ps of each anneal in the fit (default: 20)")
    p.add_argument("--fit-min-frac", type=float, default=0.1,
                   help="stop fitting once L_a < this fraction of L_a(0) (default: 0.1)")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--delete-dump", action="store_true",
                   help="delete dump_anneal.lammpstrj after analysis (fraction.dat is kept)")
    p.add_argument("--lammps", default=DEFAULT_LAMMPS,
                   help=f"LAMMPS command (default: {DEFAULT_LAMMPS}); may be 'srun <binary>'")
    p.add_argument("--np", type=int, default=1, help="MPI ranks per LAMMPS run (default: 1)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: results/tests/<pot's parent dir name>/"
                        "amorphous_regrowth)")
    args = p.parse_args()
    for ax, n in zip(("nx", "ny", "nz"), ORIENTS[args.orient]["reps"]):
        if getattr(args, ax) is None:
            setattr(args, ax, n)

    if args.stage != "analyze" and args.pot is None:
        p.error("--pot is required unless --stage analyze")
    if args.backend == "lammps-sw" and args.stage != "analyze":
        lammps_pair.check_sw(args.pot)
    if args.outdir is None:
        if args.pot is None:
            p.error("--outdir is required when --pot is omitted")
        outdir = Path("results/tests") / Path(args.pot).resolve().parent.name / "amorphous_regrowth"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if any(T > args.tm for T in args.T) and args.stage in ("all", "anneal"):
        print(f"  NOTE: T > ~{args.tm:.0f} K (Tm) — the crystal may melt; v < 0 then.")

    if args.stage in ("all", "prep"):
        stage_prep(args.pot, outdir, args.lammps, args.np, args.a0, args.nx, args.ny, args.nz,
                   args.T_melt, args.steps_melt, args.steps_quench, args.steps_relax,
                   args.barostat, args.seed, args.orient)
    if args.stage in ("all", "anneal"):
        for T in args.T:
            stage_anneal(args.pot, outdir, args.lammps, args.np, T, args.steps_anneal,
                         args.dt_anneal, args.dump_every, args.barostat, args.seed, args.nbins,
                         args.fit_skip_ps, args.fit_min_frac, args.delete_dump)
    if args.stage in ("all", "analyze"):
        stage_analyze(outdir, args.pot, args.fit_skip_ps, args.fit_min_frac, args.tm)


if __name__ == "__main__":
    main()
