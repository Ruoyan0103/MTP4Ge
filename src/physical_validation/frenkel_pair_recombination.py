"""Close Frenkel-pair stability and recombination barrier for Ge MTP potentials.

A Frenkel pair (FP) = one vacancy + one self-interstitial in the same cell.
For a series of vacancy–interstitial separation shells this test

  1. builds the FP in a relaxed diamond supercell (same SUPERCELL convention
     as defect_formation.py, default 3×3×3 cubic = 216 sites): the
     interstitial is created with defect_formation.py's own builders
     (X = ⟨110⟩ split dumbbell, T = tetrahedral; pick via --interstitial),
     then the lattice atom at the requested shell around the interstitial
     centre is removed (vacancy);
  2. relaxes the ions with the cell fixed at the relaxed-bulk cell
     (ISIF=2 analog, defect_formation.py's _relax_lammps / _relax_mlp);
  3. classifies the relaxed structure by Wigner–Seitz (WS) analysis against
     the relaxed perfect-crystal sites (each atom is assigned to its nearest
     perfect site, minimum image):
       recombined  - no empty WS cell  AND  E_FP − E_bulk < --recomb-tol
                     (the pair annihilated without a barrier);
       WS-clean    - no empty WS cell but E_FP − E_bulk ≥ --recomb-tol (a
                     topological bond defect — e.g. the fourfold-coordinated
                     defect / "IV pair" known in diamond-structure Si/Ge —
                     not counted as recombined);
       stable FP   - at least one empty WS cell remains;
  4. reports the FP formation energy vs separation,
         E_FP = E(FP cell, N atoms) − E(perfect cell, N atoms),
     which needs no chemical potential since both cells hold N atoms, and the
     spontaneous-recombination radius = largest initial separation whose FP
     recombined during relaxation (any stable shell inside that radius is
     flagged in the summary);
  5. for the closest --neb-shells stable FP shells, runs a climbing-image
     NEB (vacancy_migration.run_neb, which also pre-relaxes both endpoints)
     from the relaxed FP to the perfect crystal → recombination barrier
     E_b = max(E_path) − E_FP and recombination energy E_FP.

NEB atom-index correspondence
-----------------------------
ASE NEB interpolates atom i of the initial image to atom i of the final
image, so the perfect-crystal final state must list *the same atoms* in the
same order, each sitting on the lattice site it will occupy after
recombination.  The FP cell has N atoms and the perfect crystal has N sites,
so this is an assignment problem: we solve it with the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) on the cost |r_i − R_j|² (minimum-
image distance between relaxed FP atom i and perfect site j), i.e. the
mapping with the smallest total squared displacement.  For a close pair this
picks exactly the physical picture: the interstitial atom (or the dumbbell
atom on the vacancy side) ends on the vacant site, or, if that is shorter,
a short replacement chain (interstitial → neighbouring site, that atom →
vacancy) — which is what a recombination event actually does.  The final
image's positions are r_i + MIC(R_j(i) − r_i), so no atom is interpolated
across a periodic boundary.  Naively moving "the interstitial atom" (last
index) onto the vacancy would force a single long jump for the
farther shells and a non-minimal path for the dumbbell, whose two atoms are
equivalent.

References
----------
reference: TODO (DFT/GAP to be computed) — no Ge DFT/GAP values for FP
formation energy vs separation, recombination radius or recombination
barrier are included here; the table has a reference column filled with "—"
until those are computed.  For consistency checks the isolated-defect limit
is E_FP(∞) = E_f(V) + E_f(SIA), see defect_formation.py
(defect_formation_energy_DFT.dat: vacancy 2.24 eV DFT).

Backends (--backend):
  lammps  - pair_style hybrid/overlay mtp nlh (required for mtp+nlh
            potentials such as results/potentials/pot_1019/*.mtp): LAMMPS cg
            for the relaxations, MTPLammpsCalculator (one LAMMPS subprocess
            per evaluation) for the NEB.
  mlp     - `mlp relax` for relaxations, MTPCalculator for the NEB.

Usage:
    python src/physical_validation/frenkel_pair_recombination.py \\
        --pot results/potentials/pot_1019/bal.mtp --backend lammps --a0 5.7609
    python src/physical_validation/frenkel_pair_recombination.py \\
        --pot results/potentials/pot_1019/bal.mtp --backend lammps --a0 5.7609 \\
        --interstitial T --n-shells 8 --neb-shells 2 --n-images 7 --fmax 0.05
    # smoke test: 64-site cell, 3 shells, one cheap NEB
    python src/physical_validation/frenkel_pair_recombination.py \\
        --pot results/potentials/pot_1019/bal.mtp --backend lammps --a0 5.7609 \\
        --supercell 2 2 2 --n-shells 3 --neb-shells 1 --n-images 3 --fmax 0.3 \\
        --max-steps 30 --outdir /tmp/fp_smoke
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from mtp_calculator import MTPCalculator, MTPLammpsCalculator
from energy_volume import DEFAULT_LAMMPS
import lammps_pair
from defect_formation import (
    SUPERCELL, DEFAULT_TRAIN_CONFIG, build_bulk_ref,
    build_X_dumbbell_from, build_T_interstitial_from,
    _nearest_origin_idx, _relax_lammps, _relax_mlp,
)
from vacancy_migration import run_neb

TEST_NAME = "frenkel_pair_recombination"
INTERSTITIALS = {
    "X": "⟨110⟩ split dumbbell",
    "T": "tetrahedral",
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mic(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image version of displacement vector(s) `vec` (..., 3)."""
    frac = vec @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return frac @ cell


def _interstitial(kind: str, b_cell, b_pos, b_types, repeat):
    """Build the SIA with defect_formation.py's builders.

    Returns (cell, pos, types, centre, axis, n_lattice) where the first
    n_lattice atoms are unperturbed lattice atoms (candidate vacancy sites),
    `centre` is the interstitial centre (T site / dumbbell centre) and `axis`
    the dumbbell direction (None for T).
    """
    if kind == "X":
        idx = _nearest_origin_idx(b_pos, b_cell)
        centre = b_pos[idx].copy()
        cell, pos, types = build_X_dumbbell_from(b_cell, b_pos, list(b_types), repeat=repeat)
        axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        n_lat = len(pos) - 2
    elif kind == "T":
        cell, pos, types = build_T_interstitial_from(b_cell, b_pos, list(b_types), repeat=repeat)
        centre = pos[-1].copy()
        axis = None
        n_lat = len(pos) - 1
    else:
        raise ValueError(f"unknown interstitial {kind!r}")
    return cell, pos, types, centre, axis, n_lat


def _vacancy_shells(cell, pos, n_lat, centre, axis, n_shells, tol=0.02):
    """Group lattice sites (indices < n_lat) into shells around `centre`.

    Shell key = (distance, |cos| to dumbbell axis) rounded to `tol`, so that
    sites at equal distance but inequivalent w.r.t. the dumbbell orientation
    are kept apart. Returns a list of dicts (sorted by distance) with a
    representative site index, its distance, |cos| and multiplicity.
    """
    d_vec = _mic(pos[:n_lat] - centre, cell)
    dist = np.linalg.norm(d_vec, axis=1)
    if axis is None:
        cosang = np.zeros(n_lat)
    else:
        cosang = np.abs(d_vec @ axis) / np.maximum(dist, 1e-12)
    keys = {}
    for i in range(n_lat):
        if dist[i] < 0.5:     # the dumbbell's own (removed) site never appears; safety
            continue
        k = (round(dist[i] / tol), round(cosang[i] / tol))
        keys.setdefault(k, []).append(i)
    shells = []
    for k in sorted(keys, key=lambda k: (k[0], k[1])):
        idxs = keys[k]
        shells.append({"site": idxs[0], "dist": float(dist[idxs[0]]),
                       "cos": float(cosang[idxs[0]]), "mult": len(idxs)})
    return shells[:n_shells]


def ws_analysis(pos, sites, cell):
    """Wigner–Seitz occupancy of perfect `sites` by atoms at `pos`.

    Returns (occupancy[n_sites], nearest_site_of_atom[n_atoms]).
    """
    occ = np.zeros(len(sites), dtype=int)
    nearest = np.empty(len(pos), dtype=int)
    for i, r in enumerate(pos):
        d = np.linalg.norm(_mic(sites - r, cell), axis=1)
        nearest[i] = int(np.argmin(d))
        occ[nearest[i]] += 1
    return occ, nearest


def perfect_final_state(fp_pos, sites, cell):
    """Perfect-crystal NEB final state with atom-index correspondence.

    Hungarian assignment of FP atoms → perfect sites minimising the summed
    squared minimum-image displacement; returns final positions
    r_i + MIC(R_j(i) − r_i) (unwrapped relative to the FP image) and the
    assignment array j(i).
    """
    from scipy.optimize import linear_sum_assignment
    n = len(fp_pos)
    if len(sites) != n:
        raise ValueError(f"FP has {n} atoms but perfect crystal has {len(sites)} sites")
    disp = _mic(sites[None, :, :] - fp_pos[:, None, :], cell)    # (n_atoms, n_sites, 3)
    cost = np.einsum("ijk,ijk->ij", disp, disp)
    rows, cols = linear_sum_assignment(cost)
    assign = np.empty(n, dtype=int)
    assign[rows] = cols
    final = fp_pos + disp[np.arange(n), assign]
    return final, assign


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(pot: str, outdir: Path, a0: float, interstitial: str, n_shells: int,
        neb_shells: int, supercell=SUPERCELL, relax_fmax: float = 1e-3,
        relax_steps: int = 2000, recomb_tol: float = 0.05,
        n_images: int = 7, fmax: float = 0.05, fmax_relax: float = 0.02,
        max_steps: int = 200, backend: str = "lammps", mlp: str | None = None,
        lammps_cmd: str = DEFAULT_LAMMPS) -> None:
    from ase import Atoms
    outdir.mkdir(parents=True, exist_ok=True)
    repeat = tuple(supercell)

    relax_fn = _relax_mlp if backend == "mlp" else _relax_lammps
    engine = mlp if backend == "mlp" else lammps_cmd
    if backend == "mlp":
        make_calc = lambda: MTPCalculator(mlp=mlp, pot=pot)
    else:
        make_calc = lambda: MTPLammpsCalculator(lammps_cmd=lammps_cmd, pot=pot,
                                                save_dir=outdir / "neb_lammps_last")

    # ── Perfect crystal: ions + cell relax (ISIF=3 analog) ──
    print("\n  [Perfect crystal: cell + ion relaxation]")
    b_cell, b_pos, b_types = build_bulk_ref(a0, repeat=repeat)
    e_bulk, b_cell, b_pos = relax_fn(engine, pot, b_cell, b_pos, b_types, outdir / "bulk",
                                      fmax=relax_fmax, max_steps=relax_steps, fix_cell=False)
    n_atoms = len(b_pos)
    print(f"  E_bulk = {e_bulk:.6f} eV ({n_atoms} atoms) | cell diag = "
          f"{tuple(round(float(x), 4) for x in np.diag(b_cell))} Å")

    # ── Interstitial + shells of vacancy sites ──
    i_cell, i_pos, i_types, centre, axis, n_lat = _interstitial(
        interstitial, b_cell, b_pos, b_types, repeat)
    shells = _vacancy_shells(i_cell, i_pos, n_lat, centre, axis, n_shells)
    print(f"\n  Interstitial: {interstitial} ({INTERSTITIALS[interstitial]}) at "
          f"{np.round(centre, 3)} Å; {len(shells)} vacancy shells")

    results = []
    relaxed = {}
    for k, sh in enumerate(shells, start=1):
        print(f"\n  [Shell {k}: d0 = {sh['dist']:.3f} Å, |cos| = {sh['cos']:.2f}, "
              f"multiplicity {sh['mult']}]")
        pos = np.delete(i_pos, sh["site"], axis=0)
        types = i_types[:sh["site"]] + i_types[sh["site"] + 1:]
        assert len(pos) == n_atoms
        sdir = outdir / f"shell_{k:02d}"
        try:
            e_fp, _, r_pos = relax_fn(engine, pot, i_cell, pos, types, sdir,
                                      fmax=relax_fmax, max_steps=relax_steps, fix_cell=True)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({"shell": k, **sh, "E_FP": None, "status": "FAILED",
                            "d_relax": float("nan"), "n_vac": -1, "barrier": None})
            continue
        r_pos = np.asarray(r_pos)
        e_f = e_fp - e_bulk
        occ, _ = ws_analysis(r_pos, b_pos, b_cell)
        vac = np.where(occ == 0)[0]
        mult = np.where(occ > 1)[0]
        if len(vac) == 0:
            status = "recombined" if e_f < recomb_tol else "WS-clean"
            d_rel = 0.0 if status == "recombined" else float("nan")
        else:
            status = "stable FP"
            d_rel = float(min(np.linalg.norm(_mic(b_pos[v] - b_pos[m], b_cell))
                              for v in vac for m in mult)) if len(mult) else float("nan")
        print(f"    E_FP = {e_f:.4f} eV | WS: {len(vac)} vacant, {len(mult)} multiply-"
              f"occupied sites | relaxed V–I separation = {d_rel:.3f} Å → {status}")
        results.append({"shell": k, **sh, "E_FP": e_f, "status": status,
                        "d_relax": d_rel, "n_vac": int(len(vac)), "barrier": None,
                        "E_end": None})
        relaxed[k] = r_pos

    # ── NEB: stable FP → perfect crystal for the closest stable shells ──
    neb_paths = {}
    stable = [r for r in results if r["status"] == "stable FP"][:neb_shells]
    for r in stable:
        k = r["shell"]
        print(f"\n  [NEB shell {k}: stable FP → perfect crystal]")
        fp_pos = relaxed[k]
        fin_pos, assign = perfect_final_state(fp_pos, b_pos, b_cell)
        moved = np.linalg.norm(fin_pos - fp_pos, axis=1)
        top = np.argsort(moved)[::-1][:3]
        print("    Largest FP→perfect displacements: " +
              ", ".join(f"atom {i}: {moved[i]:.2f} Å" for i in top))
        initial = Atoms("Ge" * n_atoms, cell=b_cell, positions=fp_pos, pbc=True)
        final = Atoms("Ge" * n_atoms, cell=b_cell, positions=fin_pos, pbc=True)
        try:
            barrier, arc, energies = run_neb(
                initial, final, make_calc, n_images=n_images, fmax=fmax,
                fmax_relax=fmax_relax, max_steps=max_steps,
                image_dir=outdir / f"neb_shell_{k:02d}")
        except Exception as exc:
            print(f"    NEB FAILED: {exc}")
            continue
        r["barrier"] = barrier
        r["E_end"] = float(energies[0] - energies[-1])
        neb_paths[k] = (arc, energies, barrier)
        with open(outdir / f"neb_shell_{k:02d}" / "energy.txt", "w") as f:
            f.write(f"# FP shell {k} → perfect crystal NEB — pot: {pot}  "
                    f"barrier = {barrier:.6f} eV\n# arc[A]  E[eV]  dE[eV]\n")
            for s, e in zip(arc, energies):
                f.write(f"  {s:.4f}  {e:.6f}  {e - energies[0]:.6f}\n")
        print(f"    Recombination barrier = {barrier:.4f} eV | "
              f"E(FP) − E(perfect, NEB end) = {r['E_end']:.4f} eV")

    # ── Recombination radius ──
    rec = [r for r in results if r["status"] == "recombined"]
    r_rec = max((r["dist"] for r in rec), default=None)
    inside_stable = [r["shell"] for r in results
                     if r_rec is not None and r["status"] != "recombined"
                     and r["dist"] < r_rec]

    # ── Text summary ──
    txt = outdir / f"{TEST_NAME}.txt"
    engine_tag = f"LAMMPS {lammps_pair.label(pot)}" if backend == "lammps" else "mlp"
    with open(txt, "w") as f:
        f.write(f"Frenkel-pair stability and recombination — pot: {pot} ({engine_tag})\n")
        f.write(f"Interstitial: {interstitial} ({INTERSTITIALS[interstitial]}) | supercell "
                f"{repeat} = {n_atoms} sites | input a0 = {a0:.4f} Å | relaxed cell diag = "
                f"{tuple(round(float(x), 4) for x in np.diag(b_cell))} Å\n")
        f.write(f"E_bulk = {e_bulk:.6f} eV | E_FP = E(FP) − E_bulk (same N) | "
                f"recombined if no empty WS cell and E_FP < {recomb_tol} eV\n")
        f.write(f"Relaxation: fixed cell, fmax {relax_fmax} eV/Å, max {relax_steps} steps | "
                f"NEB: {n_images} images, CI, fmax {fmax} eV/Å, max {max_steps} steps\n\n")
        hdr = (f"{'Shell':>5} {'d0 (Å)':>8} {'|cos|':>6} {'mult':>5} {'d_relax (Å)':>12} "
               f"{'E_FP (eV)':>10} {'status':>11} {'E_b (eV)':>9} {'ref':>5}\n")
        f.write(hdr + "-" * (len(hdr) - 1) + "\n")
        for r in results:
            ef = f"{r['E_FP']:10.4f}" if r["E_FP"] is not None else f"{'—':>10}"
            eb = f"{r['barrier']:9.4f}" if r["barrier"] is not None else f"{'—':>9}"
            f.write(f"{r['shell']:>5} {r['dist']:8.3f} {r['cos']:6.2f} {r['mult']:>5} "
                    f"{r['d_relax']:12.3f} {ef} {r['status']:>11} {eb} {'—':>5}\n")
        f.write("\nreference: TODO (DFT/GAP to be computed)\n")
        f.write("\nSpontaneous-recombination radius (largest recombined d0): " +
                (f"{r_rec:.3f} Å\n" if r_rec is not None else "none recombined\n"))
        if inside_stable:
            f.write(f"  Note: shells {inside_stable} lie inside that radius but did NOT "
                    f"recombine (non-monotonic).\n")
        for r in results:
            if r["barrier"] is not None:
                f.write(f"Shell {r['shell']}: recombination barrier {r['barrier']:.4f} eV, "
                        f"E(FP) − E(NEB final) = {r['E_end']:.4f} eV\n")
        for k, (arc, energies, _) in neb_paths.items():
            f.write(f"\n# NEB shell {k}\n# arc[A]  E[eV]  dE[eV]\n")
            for s, e in zip(arc, energies):
                f.write(f"  {s:.4f}  {e:.6f}  {e - energies[0]:.6f}\n")
    np.savetxt(outdir / "E_FP_vs_separation.dat",
               np.array([[r["shell"], r["dist"], r["d_relax"],
                          r["E_FP"] if r["E_FP"] is not None else np.nan,
                          {"recombined": 1, "WS-clean": 2, "stable FP": 0}.get(r["status"], -1),
                          r["barrier"] if r["barrier"] is not None else np.nan]
                         for r in results]),
               header="shell d0[A] d_relax[A] E_FP[eV] status(0=stable,1=recombined,"
                      "2=WS-clean,-1=failed) barrier[eV]", fmt="%.6f")
    print(f"\n  Results written to {txt}")

    # ── Plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol = 2 if neb_paths else 1
        fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4.2), squeeze=False)
        ax = axes[0, 0]
        style = {"stable FP": ("o", "tab:blue"), "recombined": ("x", "tab:red"),
                 "WS-clean": ("s", "tab:orange")}
        for st, (m, c) in style.items():
            pts = [(r["dist"], r["E_FP"]) for r in results
                   if r["status"] == st and r["E_FP"] is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, m, color=c, ms=7, label=st, mfc="none" if m == "s" else c)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        if r_rec is not None:
            ax.axvline(r_rec, color="tab:red", lw=0.8, ls=":", label=f"r_rec = {r_rec:.2f} Å")
        ax.set_xlabel("Initial V–I separation d0 (Å)")
        ax.set_ylabel("E_FP − E_bulk (eV)")
        ax.set_title(f"Frenkel pair ({interstitial}-SIA), {n_atoms} sites")
        ax.legend(fontsize=8)
        if neb_paths:
            ax = axes[0, 1]
            for k, (arc, energies, b) in neb_paths.items():
                ax.plot(arc, energies - energies[0], "o-", ms=4,
                        label=f"shell {k}: E_b = {b:.3f} eV")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Arc length of most-displaced atom (Å)")
            ax.set_ylabel("E − E(FP) (eV)")
            ax.set_title("Recombination NEB (FP → perfect)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        png = outdir / f"{TEST_NAME}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {png}")
    except ImportError as e:
        print(f"  Matplotlib not available ({e}); skipping plot.")

    # ── Console summary ──
    print(f"\n  {'shell':>5} {'d0':>7} {'d_rel':>7} {'E_FP':>9} {'status':>11} {'E_b':>8}")
    for r in results:
        ef = f"{r['E_FP']:9.4f}" if r["E_FP"] is not None else f"{'—':>9}"
        eb = f"{r['barrier']:8.4f}" if r["barrier"] is not None else f"{'—':>8}"
        print(f"  {r['shell']:>5} {r['dist']:7.3f} {r['d_relax']:7.3f} {ef} "
              f"{r['status']:>11} {eb}")
    print("  Recombination radius: " + (f"{r_rec:.3f} Å" if r_rec is not None else "none"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Frenkel-pair stability / recombination test "
                                            "for Ge MTP potentials")
    p.add_argument("--pot", required=True, help="Path to potential (.mtp/.almtp)")
    p.add_argument("--a0", type=float, default=5.76,
                   help="Lattice constant Å — use the potential's equilibrium a0 "
                        "(the bulk cell is relaxed anyway; default 5.76)")
    p.add_argument("--interstitial", choices=list(INTERSTITIALS), default="X",
                   help="SIA type: X = ⟨110⟩ split dumbbell (default), T = tetrahedral")
    p.add_argument("--supercell", type=int, nargs=3, default=list(SUPERCELL),
                   metavar=("NX", "NY", "NZ"),
                   help=f"Conventional-cell repeat count (default: {list(SUPERCELL)})")
    p.add_argument("--n-shells", type=int, default=8,
                   help="Number of vacancy shells around the SIA (default: 8)")
    p.add_argument("--neb-shells", type=int, default=2,
                   help="Run recombination NEB for the closest N stable shells (default: 2; 0 = none)")
    p.add_argument("--recomb-tol", type=float, default=0.05,
                   help="E_FP below this (eV) with no empty WS cell = recombined (default: 0.05)")
    p.add_argument("--relax-fmax", type=float, default=1e-3,
                   help="Force tolerance for FP relaxations eV/Å (convergence check; default 1e-3)")
    p.add_argument("--relax-steps", type=int, default=2000,
                   help="Max minimisation steps per FP relaxation (default: 2000)")
    p.add_argument("--n-images", type=int, default=7, help="NEB intermediate images (default: 7)")
    p.add_argument("--fmax", type=float, default=0.05, help="NEB force threshold eV/Å (default: 0.05)")
    p.add_argument("--fmax-relax", type=float, default=0.02,
                   help="NEB endpoint pre-relaxation threshold eV/Å (default: 0.02)")
    p.add_argument("--max-steps", type=int, default=200, help="Max NEB BFGS steps (default: 200)")
    p.add_argument("--outdir", default=None,
                   help=f"Output directory (default: results/tests/<pot's parent dir name>/{TEST_NAME})")
    p.add_argument("--backend", choices=["mlp", "lammps", "lammps-sw"], default="lammps",
                   help="'lammps' (pair_style mtp+nlh, default — required for mtp+nlh pots), "
                        "'lammps-sw' (SW + NLH table, --pot = its directory) or 'mlp'")
    p.add_argument("--config", default=DEFAULT_TRAIN_CONFIG,
                   help="Training YAML (provides mlp_binary; --backend mlp only)")
    p.add_argument("--mlp", default=None, help="Override mlp binary path (--backend mlp only)")
    p.add_argument("--lammps", default=DEFAULT_LAMMPS,
                   help=f"LAMMPS command (--backend lammps only; default: {DEFAULT_LAMMPS})")
    args = p.parse_args()
    if args.backend == "lammps-sw":
        lammps_pair.check_sw(args.pot)
        args.backend = "lammps"

    mlp = args.mlp
    if args.backend == "mlp" and mlp is None:
        with open(args.config) as f:
            mlp = yaml.safe_load(f)["mlp_binary"]

    if args.outdir is None:
        outdir = Path("results/tests") / Path(args.pot).resolve().parent.name / TEST_NAME
    else:
        outdir = Path(args.outdir)

    run(pot=args.pot, outdir=outdir, a0=args.a0, interstitial=args.interstitial,
        n_shells=args.n_shells, neb_shells=args.neb_shells, supercell=tuple(args.supercell),
        relax_fmax=args.relax_fmax, relax_steps=args.relax_steps, recomb_tol=args.recomb_tol,
        n_images=args.n_images, fmax=args.fmax, fmax_relax=args.fmax_relax,
        max_steps=args.max_steps, backend=args.backend, mlp=mlp, lammps_cmd=args.lammps)


if __name__ == "__main__":
    main()
