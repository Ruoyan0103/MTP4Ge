"""Self-interstitial (SIA) migration barriers via climbing-image NEB for Ge MTP.

Computes NEB barriers for the main self-interstitial hops in diamond Ge:

  T-T  tetrahedral → neighbouring tetrahedral site, through the hexagonal (H)
       site on the ⟨111⟩ line joining them.
  X-X  ⟨110⟩ split dumbbell translating to a nearest-neighbour lattice site
       (one dumbbell atom returns to the original site, the other forms a new
       ⟨110⟩ dumbbell with the neighbour).
  X-T  dumbbell → tetrahedral conversion (forward = X→T, reverse = T→X).

Endpoint construction (atom ordering is IDENTICAL in initial and final state;
the final state is built from the initial one by moving specific atoms, so
NEB interpolation pairs every atom with itself). Every displacement uses the
minimum-image vector and positions are never re-wrapped afterwards, so the
linear interpolation never sends an atom across the box.

  T-T: initial = defect_formation.build_T_interstitial_from (T site at
       conventional-cell fractional (1/2,1/2,1/2) of the middle block, extra
       atom appended last). The H site from build_H_interstitial_from,
       (5/8,5/8,5/8), is the midpoint of the hop: the interstitial (last atom)
       is moved to T + 2·(H − T) = (3/4,3/4,3/4), the next T site along
       [111] (the T site of the other type; the two are related by inversion
       through a bond centre, so E_final = E_initial by symmetry).
  X-X: initial = defect_formation.build_X_dumbbell_from (atom at the origin
       replaced by a pair at c ± 1.2 Å·[110]/√2, appended last as atoms a=c−d,
       b=c+d), then the whole cell is translated by a lattice vector so the
       dumbbell sits mid-cell. Final: atom a → the lattice site c; atom b and
       the nearest-neighbour lattice atom n closest to b (n = c + a0/4·[111])
       → the new dumbbell positions n ± d (same [110] orientation; the
       assignment b/n ↔ n∓d/n±d minimising the total displacement is used).
       The final state is the inversion image of the initial one through the
       c–n bond centre, so E_final = E_initial by symmetry. Three atoms move.
  X-T: initial = the same X state. Final: atom a → the lattice site c, atom b
       → the ideal T site nearest to b (c + a0/4·[11-1], which has c as one of
       its four neighbours). Two atoms move.

After both endpoints are relaxed (fixed cell, ASE BFGS), each one is
classified from its geometry (`classify_sia`): atoms are assigned to the
nearest ideal lattice site (Wigner–Seitz analysis); the one doubly occupied
site gives the defect pair. One pair atom on the site and the other within
0.7 Å of an ideal T (H) site → "T" ("H"); both pair atoms off the site, pair
centroid within 1.0 Å of it and split along ⟨110⟩ → "X" (⟨100⟩/⟨111⟩/other
splits are labelled "split<hkl>"). The 1.0 Å centroid tolerance admits the
off-centre ⟨110⟩ dumbbell (e.g. pot_1019/bal.mtp relaxes X to a [110] pair
whose centre is shifted 0.71 Å along [00-1]);
anything else (vacancy + extra atoms, bond-centre, …) → "other". A path whose
relaxed endpoints are the same configuration (max atomic displacement between
them < 0.3 Å) is flagged COLLAPSED and its NEB is skipped — the barrier would
be meaningless. Endpoints that relaxed to a different site type than built are
flagged (the barrier then refers to the relaxed endpoints). Known case: with
current MTPs the H interstitial relaxes to T and T lies below X (wrong vs DFT),
so X endpoints may relax into T or the X-T barrier may be inverted.

Reported per path: as-built and as-relaxed endpoint types, E_final − E_initial,
forward barrier (E_max − E_initial), reverse barrier (E_max − E_final), and
the formation energy of each relaxed endpoint (μ_Ge from the ideal bulk cell
at --a0, which should be the MTP's own equilibrium lattice constant).

References:
  Migration barriers: reference: TODO (DFT/GAP to be computed) — no project
    DFT SIA migration barriers exist yet.
  Endpoint energy difference: project DFT relaxed SIA energies give the ⟨110⟩
    dumbbell (X) as ground state with T +0.21 eV above X (64-atom cells;
    note this test defaults to 3×3×3 = 216+1 atoms).

Backends (--backend):
  mlp     - MTPCalculator (`mlp calculate_efs` per image/step, default).
  lammps  - MTPLammpsCalculator (LAMMPS pair_style hybrid/overlay mtp nlh per
            image/step). REQUIRED for mtp+nlh potentials ("Wrong radial basis
            type" in mlp).

Outputs (--outdir, default results/tests/<pot parent dir>/interstitial_migration):
  interstitial_migration.txt   summary table + NEB path energies
  interstitial_migration.png   ΔE vs configuration-space arc length per path
  <path>/built/                as-built endpoints (xyz/POSCAR/lammps-data/cfg)
  <path>/before/, <path>/after/  NEB images right after interpolation / after
                               optimisation (vacancy_migration._save_neb_images)
  <path>/energy.txt            per-path NEB energies

Usage:
    python src/physical_validation/interstitial_migration.py \\
        --pot results/potentials/pot_1019/bal.mtp --backend lammps --a0 5.7609
    python src/physical_validation/interstitial_migration.py \\
        --pot results/potentials/pot_1019/bal.mtp --backend lammps --a0 5.7609 \\
        --paths T-T --supercell 2 2 2 --n-images 3 --fmax 0.1
    sbatch scripts/submit_interstitial_migration.sh results/potentials/pot_1019/bal.mtp --a0 5.7609
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from mtp_calculator import MTPCalculator, MTPLammpsCalculator
from energy_volume import DEFAULT_LAMMPS
import lammps_pair
from defect_formation import (
    SUPERCELL,
    build_bulk_ref,
    build_T_interstitial_from,
    build_H_interstitial_from,
    build_X_dumbbell_from,
)
from vacancy_migration import (
    MLP_DEFAULT,
    run_neb,
    _relax_endpoint,
    _save_neb_images,
)

PATHS = ("T-T", "X-X", "X-T")
BUILT_TYPES = {"T-T": ("T", "T"), "X-X": ("X", "X"), "X-T": ("X", "T")}
DESCRIPTION = {
    "T-T": "T -> H -> T along <111>",
    "X-X": "<110> dumbbell translation to NN site",
    "X-T": "X -> T conversion (reverse = T -> X)",
}

# Barrier references: none available yet.
BARRIER_REF = {p: "TODO (DFT/GAP to be computed)" for p in PATHS}
# E_final - E_initial reference (eV): project DFT relaxed SIAs, 64-atom cells.
DE_REF = {"T-T": 0.0, "X-X": 0.0, "X-T": 0.21}
DE_REF_NOTE = {
    "T-T": "symmetric path",
    "X-X": "symmetric path",
    "X-T": "project DFT, T - X, 64-atom cells",
}

SITE_TOL = 0.7       # Å — "atom sits on this ideal site"
DUMBBELL_TOL = 1.0   # Å — max offset of a split pair's centroid from its lattice site
COLLAPSE_TOL = 0.3   # Å — max displacement below which endpoints are identical

# Ideal T and H sites in one conventional diamond cell (A atoms at fcc,
# B atoms at fcc + (1/4,1/4,1/4)).
_T_FRAC = np.array([
    [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],      # 4 B nbrs
    [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75],  # 4 A nbrs
])
_H_DIRS = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]]) / 8.0
_H_FRAC = np.mod((_T_FRAC[:4, None, :] + _H_DIRS[None, :, :]).reshape(-1, 3), 1.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mic(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image version of displacement vector(s) `vec` (..., 3)."""
    frac = vec @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def _move(pos: np.ndarray, i: int, target: np.ndarray, cell: np.ndarray) -> None:
    """Move atom i to (the periodic image of) `target` nearest its current position."""
    pos[i] = pos[i] + _mic(target - pos[i], cell)


def _ideal_sites(cell: np.ndarray, repeat) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian ideal T and H sites of the whole supercell."""
    a_eff = np.diag(cell) / np.array(repeat)
    blocks = np.array([[i, j, k] for i in range(repeat[0])
                       for j in range(repeat[1]) for k in range(repeat[2])])
    t = ((blocks[:, None, :] + _T_FRAC[None]) * a_eff).reshape(-1, 3)
    h = ((blocks[:, None, :] + _H_FRAC[None]) * a_eff).reshape(-1, 3)
    return t, h


def _atoms(cell, pos):
    from ase import Atoms
    return Atoms(symbols=["Ge"] * len(pos), cell=cell, positions=pos, pbc=True)


# ---------------------------------------------------------------------------
# Endpoint builders (see module docstring for each path's construction)
# ---------------------------------------------------------------------------

def _build_X_initial(a0: float, repeat):
    """X dumbbell (defect_formation builder) translated mid-cell.

    Returns (atoms, lattice_sites, c, idx_a, idx_b, d) with atom a = c − d,
    b = c + d (the last two atoms).
    """
    cell, pos, types = build_bulk_ref(a0, repeat=repeat)
    x_cell, x_pos, _ = build_X_dumbbell_from(cell, pos, types, repeat=repeat)
    n = len(x_pos)
    b, a = n - 2, n - 1          # builder appends center + disp, then center - disp
    shift = (np.array(repeat) // 2) * (np.diag(cell) / np.array(repeat))
    atoms = _atoms(x_cell, x_pos + shift)
    atoms.wrap()
    p = atoms.get_positions()
    d = 0.5 * _mic(p[b] - p[a], x_cell)
    c = p[a] + d
    return atoms, pos, c, a, b, d


def build_path(path: str, a0: float, repeat=SUPERCELL):
    """Return (initial, final, lattice_sites, moved_atom_indices) for one path."""
    if path == "T-T":
        cell, pos, types = build_bulk_ref(a0, repeat=repeat)
        t_cell, t_pos, _ = build_T_interstitial_from(cell, pos, types, repeat=repeat)
        _, h_pos, _ = build_H_interstitial_from(cell, pos, types, repeat=repeat)
        t, h = t_pos[-1], h_pos[-1]
        initial = _atoms(t_cell, t_pos)
        fpos = t_pos.copy()
        fpos[-1] = t + 2.0 * (h - t)
        return initial, _atoms(t_cell, fpos), pos, [len(t_pos) - 1]

    initial, lattice, c, ia, ib, d = _build_X_initial(a0, repeat)
    cell = initial.get_cell().array
    p = initial.get_positions()
    fpos = p.copy()

    if path == "X-X":
        # nearest-neighbour lattice atom (in the current structure) closest to b
        nn_dist = a0 * np.sqrt(3.0) / 4.0
        r_c = np.linalg.norm(_mic(p - c, cell), axis=1)
        cand = [i for i in np.where(np.abs(r_c - nn_dist) < 0.3)[0] if i not in (ia, ib)]
        if not cand:
            raise RuntimeError("No nearest-neighbour atom found around the dumbbell site")
        r_b = [np.linalg.norm(_mic(p[i] - p[ib], cell)) for i in cand]
        n_idx = int(cand[int(np.argmin(r_b))])
        n_site = p[n_idx].copy()
        _move(fpos, ia, c, cell)
        # assign {b, n} to {n + d, n − d} with the smaller total squared displacement
        opt1 = (np.sum(_mic(n_site + d - p[ib], cell) ** 2) + np.sum(d ** 2))
        opt2 = (np.sum(_mic(n_site - d - p[ib], cell) ** 2) + np.sum(d ** 2))
        sb = 1.0 if opt1 <= opt2 else -1.0
        _move(fpos, ib, n_site + sb * d, cell)
        _move(fpos, n_idx, n_site - sb * d, cell)
        return initial, _atoms(cell, fpos), lattice, [ia, ib, n_idx]

    if path == "X-T":
        t_sites, _ = _ideal_sites(cell, repeat)
        dt = np.linalg.norm(_mic(t_sites - p[ib], cell), axis=1)
        _move(fpos, ia, c, cell)
        _move(fpos, ib, t_sites[int(np.argmin(dt))], cell)
        return initial, _atoms(cell, fpos), lattice, [ia, ib]

    raise ValueError(f"Unknown path {path!r}; choose from {PATHS}")


# ---------------------------------------------------------------------------
# Endpoint classification
# ---------------------------------------------------------------------------

def classify_sia(atoms, lattice: np.ndarray, repeat) -> tuple[str, str]:
    """Classify a single-SIA configuration as T / H / X / split<hkl> / other.

    Returns (label, detail string). See module docstring for the criteria.
    """
    cell = atoms.get_cell().array
    pos = atoms.get_positions()
    diff = _mic(pos[:, None, :] - lattice[None, :, :], cell)
    dist = np.linalg.norm(diff, axis=2)
    site = np.argmin(dist, axis=1)
    occ = np.bincount(site, minlength=len(lattice))
    n_vac = int(np.sum(occ == 0))
    multi = np.where(occ >= 2)[0]
    if n_vac != 0 or len(multi) != 1 or occ[multi[0]] != 2:
        return "other", f"WS: {n_vac} empty, {len(multi)} multiply-occupied sites"

    s = int(multi[0])
    i, j = np.where(site == s)[0]
    u_i, u_j = diff[i, s], diff[j, s]
    r_i, r_j = np.linalg.norm(u_i), np.linalg.norm(u_j)
    centroid = np.linalg.norm(0.5 * (u_i + u_j))

    if min(r_i, r_j) < SITE_TOL:
        k = i if r_i > r_j else j
        t_sites, h_sites = _ideal_sites(cell, repeat)
        d_t = float(np.min(np.linalg.norm(_mic(t_sites - pos[k], cell), axis=1)))
        d_h = float(np.min(np.linalg.norm(_mic(h_sites - pos[k], cell), axis=1)))
        detail = f"atom {k}: d(T)={d_t:.2f} A, d(H)={d_h:.2f} A"
        if d_t < SITE_TOL and d_t <= d_h:
            return "T", detail
        if d_h < SITE_TOL:
            return "H", detail
        return "other", detail

    if centroid < DUMBBELL_TOL:
        v = (u_i - u_j) / np.linalg.norm(u_i - u_j)
        fams = {
            "110": np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
                             [0, 1, 1], [0, 1, -1]]) / np.sqrt(2),
            "100": np.eye(3),
            "111": np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]) / np.sqrt(3),
        }
        best = {f: float(np.max(np.abs(dirs @ v))) for f, dirs in fams.items()}
        fam = max(best, key=best.get)
        detail = (f"atoms {i},{j}: split {np.linalg.norm(u_i - u_j):.2f} A, "
                  f"centroid off-site {centroid:.2f} A, |cos<{fam}>|={best[fam]:.3f}")
        if best[fam] > 0.9:
            return ("X" if fam == "110" else f"split<{fam}>"), detail
        return "split", detail

    return "other", (f"atoms {i},{j}: r={r_i:.2f},{r_j:.2f} A, "
                     f"centroid off-site {centroid:.2f} A")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _config_arc(image_dir: Path) -> np.ndarray | None:
    """Cumulative 3N-dim path length (Å) from the saved post-NEB xyz images."""
    from ase.io import read as ase_read
    files = sorted(image_dir.glob("*.xyz"))
    if len(files) < 2:
        return None
    imgs = [ase_read(str(f)) for f in files]
    cell = imgs[0].get_cell().array
    steps = [np.linalg.norm(_mic(b.get_positions() - a.get_positions(), cell))
             for a, b in zip(imgs[:-1], imgs[1:])]
    return np.concatenate([[0.0], np.cumsum(steps)])


def _plot_path(ax, path: str, r: dict) -> None:
    dE = r["energies"] - r["energies"][0]
    ax.plot(r["arc"], dE, "o-", color="tab:blue", markersize=5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Configuration-space arc length (Å)")
    ax.set_ylabel("E − E_initial (eV)")
    ax.set_title(f"{path}: {r['relaxed'][0]} → {r['relaxed'][1]}  "
                 f"[fwd {r['fwd']:.3f} / rev {r['rev']:.3f} eV]", fontsize=10)
    ax.text(0.98, 0.05, "ref barrier: TODO", transform=ax.transAxes,
            ha="right", fontsize=8, color="gray")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(pot: str, outdir: Path, a0: float, paths, supercell,
        n_images: int, fmax: float, fmax_relax: float, max_steps: int,
        backend: str = "mlp", mlp: str | None = None,
        lammps_cmd: str = DEFAULT_LAMMPS) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    supercell = tuple(supercell)

    if backend == "mlp":
        make_calc = lambda: MTPCalculator(mlp=mlp, pot=pot)
    else:
        make_calc = lambda: MTPLammpsCalculator(lammps_cmd=lammps_cmd, pot=pot,
                                                save_dir=outdir / "lammps_last_call")

    # μ_Ge from the ideal bulk cell at a0 (forces vanish by symmetry)
    b_cell, b_pos, _ = build_bulk_ref(a0, repeat=supercell)
    bulk = _atoms(b_cell, b_pos)
    bulk.calc = make_calc()
    mu = bulk.get_potential_energy() / len(bulk)
    print(f"  μ_Ge = {mu:.6f} eV/atom  ({len(bulk)}-atom bulk, a0 = {a0:.4f} Å)")

    results = {}
    for path in paths:
        print(f"\n  [{path}: {DESCRIPTION[path]}]")
        pdir = outdir / path
        r = {"built": BUILT_TYPES[path], "relaxed": ("?", "?"), "status": "FAILED",
             "notes": [], "fwd": None, "rev": None, "dE": None, "Ef": (None, None)}
        results[path] = r
        try:
            initial, final, lattice, moved = build_path(path, a0, supercell)
            _save_neb_images([initial, final], pdir / "built")
            print(f"    {len(initial)} atoms; moved atoms {moved}")
            for tag, at in (("built initial", initial), ("built final", final)):
                lab, det = classify_sia(at, lattice, supercell)
                print(f"    {tag:<14}: {lab:<10} ({det})")

            print(f"    Relaxing endpoints (fmax={fmax_relax} eV/Å) ...")
            ini_r = _relax_endpoint(initial, make_calc, fmax=fmax_relax)
            fin_r = _relax_endpoint(final, make_calc, fmax=fmax_relax)
            e_i, e_f = ini_r.get_potential_energy(), fin_r.get_potential_energy()
            n = len(ini_r)
            r["Ef"] = (e_i - n * mu, e_f - n * mu)
            labels = []
            for tag, at in (("relaxed initial", ini_r), ("relaxed final", fin_r)):
                lab, det = classify_sia(at, lattice, supercell)
                labels.append(lab)
                print(f"    {tag:<16}: {lab:<10} ({det})")
            r["relaxed"] = tuple(labels)
            r["dE"] = e_f - e_i
            for end, built, got in zip(("initial", "final"), r["built"], labels):
                if built != got:
                    r["notes"].append(f"{end} relaxed {built}->{got}")

            max_disp = float(np.max(np.linalg.norm(
                _mic(fin_r.get_positions() - ini_r.get_positions(),
                     ini_r.get_cell().array), axis=1)))
            print(f"    E_final − E_initial = {r['dE']:+.4f} eV ; "
                  f"max displacement between relaxed endpoints = {max_disp:.3f} Å")
            if max_disp < COLLAPSE_TOL:
                r["status"] = "COLLAPSED"
                r["notes"].append(f"endpoints identical (max disp {max_disp:.2f} A); NEB skipped")
                print("    ⚠ Endpoints collapsed into the same configuration — "
                      "barrier meaningless, NEB skipped.")
                continue

            print(f"    Running CI-NEB ({n_images} images, fmax={fmax} eV/Å) ...")
            _, arc_atom, energies = run_neb(
                ini_r, fin_r, make_calc, n_images=n_images, fmax=fmax,
                fmax_relax=fmax_relax, max_steps=max_steps, image_dir=pdir,
            )
            arc = _config_arc(pdir / "after")
            if arc is None or len(arc) != len(energies):
                arc = arc_atom
            e_max = float(np.max(energies))
            r.update(energies=energies, arc=arc, arc_atom=arc_atom,
                     fwd=e_max - energies[0], rev=e_max - energies[-1],
                     dE=float(energies[-1] - energies[0]), status="OK")
            i_max = int(np.argmax(energies))
            if i_max in (0, len(energies) - 1):
                r["notes"].append("max at endpoint (no interior saddle)")
            with open(pdir / "energy.txt", "w") as f:
                f.write(f"# {path} CI-NEB — pot: {pot}  fwd = {r['fwd']:.6f} eV  "
                        f"rev = {r['rev']:.6f} eV\n")
                f.write("# arc_config[A]  arc_moving_atom[A]  E[eV]  E-E_initial[eV]\n")
                for s, sa, e in zip(arc, arc_atom, energies):
                    f.write(f"  {s:.4f}  {sa:.4f}  {e:.6f}  {e - energies[0]:.6f}\n")
            print(f"    Forward barrier = {r['fwd']:.3f} eV, reverse = {r['rev']:.3f} eV "
                  f"(reference: TODO)")
        except Exception as exc:  # keep going with the other paths
            r["notes"].append(f"error: {exc}")
            print(f"    FAILED: {exc}")

    # --- Summary text ---
    def _f(x, fmt="{:.3f}"):
        return fmt.format(x) if x is not None else "—"

    txt = outdir / "interstitial_migration.txt"
    engine = "mlp calculate_efs" if backend == "mlp" else f"LAMMPS {lammps_pair.label(pot)} ({lammps_cmd})"
    lines = [
        f"Self-interstitial migration barriers (CI-NEB) — pot: {pot}",
        f"Engine: {engine} | a0 = {a0:.4f} Å | supercell {supercell} "
        f"({len(bulk)}+1 atoms) | n_images = {n_images} | fmax = {fmax} eV/Å "
        f"| endpoint fmax = {fmax_relax} eV/Å",
        f"μ_Ge = {mu:.6f} eV/atom",
        "",
        "Barriers: fwd = E_max − E_initial, rev = E_max − E_final (eV). "
        "Ref barrier: TODO (DFT/GAP to be computed).",
        "",
        f"{'Path':<5} {'Built':<7} {'Relaxed':<14} {'Status':<10} {'Ef_i':>7} {'Ef_f':>7} "
        f"{'E_f-E_i':>8} {'ref dE':>7} {'fwd':>7} {'rev':>7}  {'ref barrier':<30} Notes",
        "-" * 130,
    ]
    for path, r in results.items():
        built = "->".join(r["built"])
        relaxed = "->".join(r["relaxed"])
        lines.append(
            f"{path:<5} {built:<7} {relaxed:<14} {r['status']:<10} "
            f"{_f(r['Ef'][0]):>7} {_f(r['Ef'][1]):>7} {_f(r['dE'], '{:+.3f}'):>8} "
            f"{DE_REF[path]:>+7.2f} {_f(r['fwd']):>7} {_f(r['rev']):>7}  "
            f"{BARRIER_REF[path]:<30} {'; '.join(r['notes'])}"
        )
    lines += [
        "",
        "Ef_i/Ef_f: formation energies of the relaxed endpoints, E − N·μ_Ge (eV).",
        "ref dE: " + "; ".join(f"{p}: {DE_REF_NOTE[p]}" for p in results),
        "Status COLLAPSED = relaxed endpoints are the same configuration (barrier meaningless).",
    ]
    for path, r in results.items():
        if r["status"] != "OK":
            continue
        lines += ["", f"# {path} NEB path",
                  "# arc_config[A]  E[eV]  E-E_initial[eV]"]
        for s, e in zip(r["arc"], r["energies"]):
            lines.append(f"  {s:.4f}  {e:.6f}  {e - r['energies'][0]:.6f}")
    txt.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:len(results) + 12]))
    print(f"\n  Results written to {txt}")

    # --- Plot ---
    ok = {p: r for p, r in results.items() if r["status"] == "OK"}
    if not ok:
        print("  No completed NEB paths; skipping plot.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(ok), figsize=(5.5 * len(ok), 4), squeeze=False)
        for ax, (path, r) in zip(axes[0], ok.items()):
            _plot_path(ax, path, r)
        tag = "" if backend == "mlp" else " (LAMMPS)"
        fig.suptitle(f"Ge self-interstitial migration — MTP{tag}", fontsize=12)
        fig.tight_layout()
        png = outdir / "interstitial_migration.png"
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
        description="Self-interstitial migration CI-NEB test for Ge MTP potential"
    )
    parser.add_argument("--pot", required=True, help="Path to potential (.mtp/.almtp)")
    parser.add_argument("--a0", type=float, default=5.76,
                        help="Lattice constant Å — use the MTP's 0 K equilibrium (default: 5.76)")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS),
                        help=f"Paths to compute (default: all of {' '.join(PATHS)})")
    parser.add_argument("--supercell", type=int, nargs=3, default=list(SUPERCELL),
                        metavar=("NX", "NY", "NZ"),
                        help=f"Conventional-cell repeat count (default: {list(SUPERCELL)})")
    parser.add_argument("--n-images", type=int, default=7,
                        help="Number of NEB intermediate images (default: 7)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="NEB force convergence eV/Å (default: 0.05)")
    parser.add_argument("--fmax-relax", type=float, default=0.02,
                        help="Endpoint relaxation force threshold eV/Å (default: 0.02)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max NEB BFGS steps (default: 200)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: results/tests/<pot's parent dir "
                             "name>/interstitial_migration)")
    parser.add_argument("--backend", choices=["mlp", "lammps", "lammps-sw"], default="mlp",
                        help="Evaluation engine: 'mlp' calculate_efs (default) or "
                             "'lammps' (pair_style mtp+nlh — required for mtp+nlh potentials) or "
                             "'lammps-sw' (SW + NLH table, --pot = its directory)")
    parser.add_argument("--mlp", default=MLP_DEFAULT,
                        help="mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=DEFAULT_LAMMPS,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             f"(--backend lammps only; default: {DEFAULT_LAMMPS})")
    args = parser.parse_args()
    if args.backend == "lammps-sw":
        lammps_pair.check_sw(args.pot)
        args.backend = "lammps"

    if args.outdir is None:
        run_name = Path(args.pot).resolve().parent.name
        outdir = Path("results/tests") / run_name / "interstitial_migration"
    else:
        outdir = Path(args.outdir)

    run(
        pot=args.pot, outdir=outdir, a0=args.a0, paths=args.paths,
        supercell=args.supercell, n_images=args.n_images, fmax=args.fmax,
        fmax_relax=args.fmax_relax, max_steps=args.max_steps,
        backend=args.backend, mlp=args.mlp, lammps_cmd=args.lammps,
    )


if __name__ == "__main__":
    main()
