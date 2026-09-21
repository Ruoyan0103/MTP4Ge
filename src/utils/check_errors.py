"""
Evaluate MTP potential accuracy against a reference dataset: compares each
config's predicted energy/forces/stresses against the values already embedded
in the reference CFG and reports RMSE (and mean/max absolute error).

Report path: results/errors/<potential_name>/<dataset_name>.txt
(<potential_name> from --pot's filename stem, <dataset_name> from --cfg's).

Backends:
    mlp     — `mlp check_errors` (default).
    lammps  — one LAMMPS single-point run per config, via a caller-supplied
              `pair_style hybrid/overlay mtp nlh` input template (e.g.
              config/lammps/phonon_dispersion_mtp.in) and LAMMPS command; for
              potentials trained with a radial basis type `mlp` can't load
              ("Wrong radial basis type"). Energy/forces come straight from
              the potential; the virial stress is derived from LAMMPS's
              pressure tensor (pxx/pyy/pzz/pyz/pxz/pxy) after the same
              single-point run — see `_run_lammps_point`. Both backends
              report the same categories (Energy, Energy per atom, Forces,
              Stresses, Virial stresses) using the same accumulation method
              as `mlp check_errors` itself (per-atom force-difference vector
              norm; per-config Frobenius norm of the stress-difference
              tensor, double-counting off-diagonal Voigt components — see
              mlip-3-prune/src/error_monitor.cpp), so results from the two
              backends are directly comparable. The LAMMPS command and
              template path are not hardcoded here; see
              scripts/submit_check_errors.sh for the cluster-specific values.

Usage:
    python src/utils/check_errors.py --pot results/potentials/pot.almtp --cfg data/cfg/test.cfg
    python src/utils/check_errors.py --pot results/potentials/20.mtp --cfg data/cfg/test.cfg \\
        --backend lammps --lammps "srun /path/to/lmp_mpi" \\
        --lammps-template config/lammps/phonon_dispersion_mtp.in
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = "config/training.yaml"
DEFAULT_OUTDIR = "results/errors"
GE_MASS = 72.630
EV_A3_TO_GPA = 160.2176487        # matches mlip-3-prune/src/error_monitor.cpp's eVA3_to_GPa
BAR_TO_EV_A3 = 1.0 / (EV_A3_TO_GPA * 1e4)   # 1 GPa = 1e4 bar (LAMMPS metal units)


def _mlp_binary(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("mlp_binary", "mlp")


_UNIT_MAP = {
    "Energy:": "Energy (eV):",
    "Energy per atom:": "Energy per atom (eV/atom):",
    "Forces:": "Forces (eV/Å):",
    "Stresses (in energy units):": "Stresses (eV):",
    "Virial stresses (in pressure units):": "Virial stresses (GPa):",
}


def _print_summary(report: Path) -> None:
    if not report.exists():
        return
    print("\n--- Error Summary ---")
    for line in report.read_text().splitlines():
        stripped = line.strip()
        print(_UNIT_MAP.get(stripped, line))
    print("---------------------")


# ---------------------------------------------------------------------------
# mlp backend — wraps `mlp check_errors` directly
# ---------------------------------------------------------------------------

def run_check_errors(mlp: str, pot: str, cfg_path: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg_path).stem
    report = outdir / f"{stem}.txt"
    log_file = outdir / f"{stem}_errors.log"
    cmd = shlex.split(mlp) + [
        "check_errors", pot, cfg_path,
        f"--log={log_file}",
        f"--report_to={report}",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: mlp check_errors exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\nError report written to: {report}")
    _print_summary(report)


# ---------------------------------------------------------------------------
# lammps backend — pair_style hybrid/overlay mtp nlh, one config at a time
# ---------------------------------------------------------------------------

def _parse_cfg_blocks(path: Path) -> list[dict]:
    """Parse every CFG block in *path* into cell/positions/types/energy/forces/stress.

    Missing Energy, forces, or PlusStress (e.g. a bare geometry-only CFG) are
    returned as ``None`` rather than raising, so callers can skip them per-config.
    """
    text = path.read_text()
    blocks = re.findall(r"BEGIN_CFG(.*?)END_CFG", text, re.DOTALL)
    if not blocks:
        raise ValueError(f"No CFG block found in {path}")

    configs = []
    for block in blocks:
        n_atoms, cell, positions, types, forces, energy, stress = 0, [], [], [], [], None, None
        has_forces, mode = False, None

        for line in block.splitlines():
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
                has_forces = "fx" in s.split()
                mode = "atoms"
            elif mode == "atoms" and s and not s.startswith(
                ("Feature", "Energy", "PlusStress", "BEGIN", "END")
            ):
                parts = s.split()
                if len(parts) >= 5:
                    types.append(int(parts[1]))
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    if has_forces and len(parts) >= 8:
                        forces.append([float(parts[5]), float(parts[6]), float(parts[7])])
                    if len(positions) == n_atoms:
                        mode = None
            elif s == "Energy":
                mode = "energy"
            elif mode == "energy":
                energy = float(s)
                mode = None
            elif s.startswith("PlusStress"):
                mode = "stress"
            elif mode == "stress":
                stress = [float(x) for x in s.split()]  # xx yy zz yz xz xy
                mode = None

        configs.append({
            "cell": np.array(cell),
            "positions": np.array(positions),
            "types": types,
            "energy": energy,
            "forces": np.array(forces) if len(forces) == n_atoms else None,
            "stress": np.array(stress) if stress is not None and len(stress) == 6 else None,
        })

    return configs


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


def _run_lammps_point(
    lammps_cmd: str, template_path: Path, pot: str,
    cell: np.ndarray, positions: np.ndarray, workdir: Path,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Single-point energy [eV], forces [eV/Å], and virial stress [eV, Voigt
    xx/yy/zz/yz/xz/xy] of one config via LAMMPS.

    Virial = pressure[bar] * volume[Å³] * BAR_TO_EV_A3. Sign convention: the
    CFG's PlusStress is virial = -stress_ASE * volume (see
    src/utils/convert_format.py's `_parse_virial_from_atoms`), and the
    standard LAMMPS elastic-constant convention Cij = -dP/dε (see
    src/physical_validation/elastic_constant/in.elastic) implies
    stress_ASE = -pressure_LAMMPS; the two minus signs cancel, so no extra
    sign flip is applied here.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    workdir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(symbols=["Ge"] * len(positions), cell=cell, positions=positions, pbc=True)
    ase_write(str(workdir / "structure.lammps"), atoms,
              format="lammps-data", atom_style="atomic")

    text = (
        Path(template_path).read_text()
        .replace("${STRUCT_FILE}", "structure.lammps")
        .replace("${POT_PATH}", str(Path(pot).resolve()))
        .replace("${MASS}", str(GE_MASS))
    )
    # The template's own `run 0` only invokes thermo_press's scalar part (the
    # default "Press" column); the pxx/pyy/... tensor components stay stale
    # ("not current") unless thermo_style explicitly requests them during a
    # run. Re-request them and re-run (free — same timestep, same state) so
    # the subsequent variable/print calls can read them.
    text += (
        "\nthermo_style custom step pxx pyy pzz pyz pxz pxy\n"
        "run 0 post no\n"
        "variable pxx equal pxx\n"
        "variable pyy equal pyy\n"
        "variable pzz equal pzz\n"
        "variable pyz equal pyz\n"
        "variable pxz equal pxz\n"
        "variable pxy equal pxy\n"
        'print "${pxx} ${pyy} ${pzz} ${pyz} ${pxz} ${pxy}" file stress.txt screen no\n'
    )
    (workdir / "in.check_errors").write_text(text)

    cmd = shlex.split(lammps_cmd) + [
        "-in", "in.check_errors", "-log", "log.lammps", "-screen", "none",
    ]
    result = subprocess.run(cmd, cwd=str(workdir))
    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS exited {result.returncode}; see {workdir / 'log.lammps'}")

    energy = float((workdir / "energy.txt").read_text().split()[0])
    forces = _parse_forces_from_dump(workdir / "force.dump", len(positions))

    pressure = np.array([float(x) for x in (workdir / "stress.txt").read_text().split()])
    volume = abs(np.linalg.det(cell))
    stress = pressure * volume * BAR_TO_EV_A3

    return energy, forces, stress


# Squared-Frobenius-norm weights for a symmetric 3x3 tensor stored as a Voigt
# vector [xx, yy, zz, yz, xz, xy]: off-diagonal components appear twice in
# the full matrix (once at [i][j], once at [j][i]), so they're weighted 2x —
# matching mlp's Matrix3::NormFrobeniusSq() (mlip-3-prune/src/common/matrix3.h)
# applied to Configuration::stresses (mlip-3-prune/src/configuration.h).
_STRESS_FROB_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])


def _frob_normsq(voigt: np.ndarray) -> float:
    """Squared Frobenius norm of the symmetric 3x3 tensor a Voigt vector
    [xx, yy, zz, yz, xz, xy] represents."""
    return float(np.sum(_STRESS_FROB_WEIGHTS * np.asarray(voigt) ** 2))


def _new_accum() -> dict:
    """A running accumulator mirroring mlp's ErrorMonitor::Accumulator
    (mlip-3-prune/src/error_monitor.h): one |diff| (delta) and squared
    reference magnitude (valsq) per item (atom, for forces; config, for
    stresses), pooled by summing/maxing across items."""
    return {"count": 0, "sum_delta": 0.0, "sum_dltsq": 0.0, "sum_valsq": 0.0,
            "max_delta": 0.0, "max_value": 0.0}


def _accumulate(acc: dict, delta: float, valsq: float) -> None:
    acc["sum_delta"] += delta
    acc["sum_dltsq"] += delta * delta
    acc["sum_valsq"] += valsq
    acc["max_delta"] = max(acc["max_delta"], delta)
    acc["max_value"] = max(acc["max_value"], float(np.sqrt(valsq)))
    acc["count"] += 1


def _rmsabs(acc: dict) -> float:
    return float(np.sqrt(acc["sum_dltsq"] / acc["count"]))


def _aveabs(acc: dict) -> float:
    return acc["sum_delta"] / acc["count"]


def _rmsrel(acc: dict) -> float:
    return float(np.sqrt(acc["sum_dltsq"] / acc["sum_valsq"])) if acc["sum_valsq"] else 0.0


def _maxratio(acc: dict) -> float:
    return acc["max_delta"] / acc["max_value"] if acc["max_value"] else 0.0


def _write_lammps_report(
    path: Path, e_diffs: list, epa_diffs: list,
    frc_acc: dict, str_acc: dict, vir_acc: dict, n_e: int,
) -> None:
    lines = ["_________________Errors report_________________"]

    if n_e:
        e = np.abs(np.array(e_diffs))
        epa = np.abs(np.array(epa_diffs))
        lines += [
            "Energy:",
            f"\tErrors checked for {n_e} configurations",
            f"\tMaximal absolute difference = {e.max():.6g}",
            f"\tAverage absolute difference = {e.mean():.6g}",
            f"\tRMS     absolute difference = {np.sqrt((e ** 2).mean()):.6g}",
            "",
            "Energy per atom:",
            f"\tErrors checked for {n_e} configurations",
            f"\tMaximal absolute difference = {epa.max():.6g}",
            f"\tAverage absolute difference = {epa.mean():.6g}",
            f"\tRMS     absolute difference = {np.sqrt((epa ** 2).mean()):.6g}",
            "",
        ]

    if frc_acc["count"]:
        lines += [
            "Forces:",
            f"\tErrors checked for {frc_acc['count']} atoms",
            f"\tMaximal absolute difference = {frc_acc['max_delta']:.6g}",
            f"\tAverage absolute difference = {_aveabs(frc_acc):.6g}",
            f"\tRMS     absolute difference = {_rmsabs(frc_acc):.6g}",
            f"\tMax(ForceDiff) / Max(Force) = {_maxratio(frc_acc):.6g}",
            f"\tRMS(ForceDiff) / RMS(Force) = {_rmsrel(frc_acc):.6g}",
            "",
        ]

    if str_acc["count"]:
        lines += [
            "Stresses (in energy units):",
            f"\tErrors checked for {str_acc['count']} configurations",
            f"\tMaximal absolute difference = {str_acc['max_delta']:.6g}",
            f"\tAverage absolute difference = {_aveabs(str_acc):.6g}",
            f"\tRMS     absolute difference = {_rmsabs(str_acc):.6g}",
            f"\tMax(StresDiff) / Max(Stres) = {_maxratio(str_acc):.6g}",
            f"\tRMS(StresDiff) / RMS(Stres) = {_rmsrel(str_acc):.6g}",
            "",
        ]

        lines += [
            "Virial stresses (in pressure units):",
            f"\tErrors checked for {vir_acc['count']} configurations",
            f"\tMaximal absolute difference = {vir_acc['max_delta']:.6g}",
            f"\tAverage absolute difference = {_aveabs(vir_acc):.6g}",
            f"\tRMS     absolute difference = {_rmsabs(vir_acc):.6g}",
            f"\tMax(StresDiff) / Max(Stres) = {_maxratio(vir_acc):.6g}",
            f"\tRMS(StresDiff) / RMS(Stres) = {_rmsrel(vir_acc):.6g}",
            "",
        ]

    if lines and lines[-1] == "":
        lines.pop()
    lines.append("_______________________________________________")
    path.write_text("\n".join(lines) + "\n")


def run_check_errors_lammps(
    lammps_cmd: str, template_path: Path, pot: str, cfg_path: str, outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg_path).stem
    configs = _parse_cfg_blocks(Path(cfg_path))
    lammps_dir = outdir / f"{stem}_lammps"

    e_diffs, epa_diffs = [], []
    frc_acc, str_acc, vir_acc = _new_accum(), _new_accum(), _new_accum()
    n_e = 0

    for i, cfg in enumerate(configs):
        n = len(cfg["positions"])
        print(f"  [{i + 1}/{len(configs)}] {n} atoms")
        e_pred, f_pred, s_pred = _run_lammps_point(
            lammps_cmd, template_path, pot, cfg["cell"], cfg["positions"], lammps_dir / f"cfg_{i:04d}"
        )

        if cfg["energy"] is not None:
            diff = e_pred - cfg["energy"]
            e_diffs.append(diff)
            epa_diffs.append(diff / n)
            n_e += 1

        if cfg["forces"] is not None:
            d = f_pred - cfg["forces"]
            dltsq_atoms = np.sum(d * d, axis=1)
            valsq_atoms = np.sum(cfg["forces"] ** 2, axis=1)
            for dltsq, valsq in zip(dltsq_atoms, valsq_atoms):
                _accumulate(frc_acc, float(np.sqrt(dltsq)), float(valsq))

        if cfg["stress"] is not None:
            dltsq_e = _frob_normsq(s_pred - cfg["stress"])
            valsq_e = _frob_normsq(cfg["stress"])
            _accumulate(str_acc, float(np.sqrt(dltsq_e)), valsq_e)

            vol = abs(np.linalg.det(cfg["cell"]))
            s_pred_gpa = s_pred / vol * EV_A3_TO_GPA
            s_ref_gpa = cfg["stress"] / vol * EV_A3_TO_GPA
            dltsq_v = _frob_normsq(s_pred_gpa - s_ref_gpa)
            valsq_v = _frob_normsq(s_ref_gpa)
            _accumulate(vir_acc, float(np.sqrt(dltsq_v)), valsq_v)

    report = outdir / f"{stem}.txt"
    _write_lammps_report(report, e_diffs, epa_diffs, frc_acc, str_acc, vir_acc, n_e)
    print(f"\nError report written to: {report}")
    _print_summary(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MTP potential accuracy against a reference dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Training YAML config (provides mlp_binary; --backend mlp only)")
    parser.add_argument("--pot", required=True, help="Trained potential (.mtp / .almtp)")
    parser.add_argument("--cfg", required=True, help="Reference dataset (.cfg)")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help=f"Base output directory; report is written to "
                             f"<outdir>/<pot_stem>/<cfg_stem>.txt (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--backend", choices=["mlp", "lammps"], default="mlp",
                        help="Evaluation engine: 'mlp' check_errors (default) "
                             "or 'lammps' (pair_style hybrid/overlay mtp nlh — for potentials "
                             "'mlp' can't load; energy, forces, and stress)")
    parser.add_argument("--mlp", default=None,
                        help="Override mlp binary path (--backend mlp only)")
    parser.add_argument("--lammps", default=None,
                        help="LAMMPS command, e.g. 'srun /path/to/lmp_mpi' "
                             "(--backend lammps only; required, no default — "
                             "see scripts/submit_check_errors.sh)")
    parser.add_argument("--lammps-template", default=None,
                        help="LAMMPS input template with ${STRUCT_FILE}/${POT_PATH}/${MASS} "
                             "placeholders, e.g. config/lammps/phonon_dispersion_mtp.in "
                             "(--backend lammps only; required, no default)")
    args = parser.parse_args()

    if not Path(args.pot).exists():
        print(f"ERROR: --pot file not found: {args.pot}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.cfg).exists():
        print(f"ERROR: --cfg file not found: {args.cfg}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) / Path(args.pot).stem

    if args.backend == "mlp":
        mlp = args.mlp or _mlp_binary(args.config)
        run_check_errors(mlp, args.pot, args.cfg, outdir)
    else:
        if not args.lammps or not args.lammps_template:
            parser.error("--backend lammps requires both --lammps and --lammps-template")
        run_check_errors_lammps(args.lammps, args.lammps_template, args.pot, args.cfg, outdir)


if __name__ == "__main__":
    main()
