"""
Re-evaluate structures with the TurboGAP GAP potential (`turbogap predict`):
single-point energy, forces, virial and stress for every configuration in an
extended-XYZ file. This is the GAP counterpart of the VASP single-point
tagging done with `VASP_settings.py` — same input structures, a different
reference method — so the two (and MTP) can later be compared config-by-
config (matching frame order/filename).

GAP potential: gap_files/Ge.gap (+ its sparse-vector and core-pot
dependencies), trained separately and stored alongside this script.
`turbogap predict` resolves the paths written inside Ge.gap (desc_sparse,
alphas_sparse, core_pot_file) relative to its own working directory, not
relative to Ge.gap's location — so each run happens inside a scratch
directory (results/gap_tagging/<input_stem>/ by default) that gets its own
copy of gap_files/ (a few MB; copied rather than symlinked so the run
directory stays a self-contained record even if gap_files/ later changes or
moves). turbogap's fixed-column reader also only accepts
Properties=species:S:1:pos:R:3 in the atoms file (no extra per-atom/per-frame
fields), so the input structures are stripped down to species+positions+cell
before being handed to it; energy/forces/stress labels already present in
the input (e.g. DFT reference values) are left untouched in the original
file and are not needed here. Afterwards `_merge_labels` reattaches the
config-type metadata (config_type, subconfig_type, sub_type,
sub_config_type) from the original frames by index, drops turbogap-only
fields that have no counterpart in the reference file (local_energy,
volume), and sets free_energy = energy — matching the VASP-tagged
convention, where free_energy (not energy) is the training-label energy.

If the input has DFT labels (free_energy/forces/virial/stress), `main`
also calls `check_errors_gap` to report GAP-vs-DFT RMSE/max/average error,
config-by-config, reusing the same accumulation method and report format as
src/utils/check_errors.py (mlp/LAMMPS backends) so all three are directly
comparable. Report: results/gap_tagging/<input_stem>/errors.txt, alongside
that run's atoms_in.xyz/trajectory_out.xyz/turbogap.log.

Usage:
    python src/tagging_structs/GAP_settings.py --input data/VASP_tagged/extxyz/train-tgap.xyz
    python src/tagging_structs/GAP_settings.py --input data/VASP_tagged/extxyz/train-tgap.xyz \\
        --outdir data/GAP_tagged/extxyz --turbogap "srun /path/to/turbogap"
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
GAP_FILES_DIR = SCRIPT_DIR / "gap_files"

# src/utils/check_errors.py holds the shared RMSE-accumulation/report
# machinery (also used by the mlp/LAMMPS backends), reused here so GAP error
# reports are directly comparable to the MTP ones.
sys.path.insert(0, str(SCRIPT_DIR.parent / "utils"))
from check_errors import (  # noqa: E402
    EV_A3_TO_GPA,
    _accumulate,
    _frob_normsq,
    _new_accum,
    _print_summary,
    _write_lammps_report,
)

# Config-type provenance fields carried over from the reference file; turbogap
# never sees them (stripped by _bare_atoms) so they must be reattached by index.
METADATA_KEYS = ("config_type", "subconfig_type", "sub_type", "sub_config_type")

POT_FILE = "gap_files/Ge.gap"   # relative to the turbogap run directory, see module docstring
N_SPECIES = 1
SPECIES = "Ge"
MASSES = 72.630                 # matches src/utils/check_errors.py's GE_MASS

DEFAULT_TURBOGAP = "/projappl/project_2012355/CODE/TurboGAP_EPH/bin/turbogap"
DEFAULT_OUTDIR = "data/GAP_tagged/extxyz"
DEFAULT_WORKDIR = "results/gap_tagging"

INPUT_TEMPLATE = """\
atoms_file = "atoms_in.xyz"
pot_file = "{pot_file}"
n_species = {n_species}
species = {species}
masses = {masses}
"""


def _bare_atoms(frames):
    """Strip calculator results/info down to species+positions+cell: turbogap's
    fixed-column parser (Properties=species:S:1:pos:R:3) errors out on any
    extra per-atom or per-frame field (e.g. forces, energy, config_type)."""
    bare = []
    for atoms in frames:
        b = atoms.copy()
        b.calc = None
        b.info = {}
        bare.append(b)
    return bare


def _run_turbogap_inline(workdir: Path, turbogap_command: str) -> None:
    """Run `turbogap predict` directly as a subprocess, using the caller's
    current allocation (e.g. turbogap_command already prefixed with srun)."""
    cmd = shlex.split(turbogap_command) + ["predict"]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    (workdir / "turbogap.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(
            f"turbogap predict failed (exit {result.returncode}); see {workdir / 'turbogap.log'}"
        )


def _submit_turbogap_sbatch(workdir: Path, turbogap_command: str, gap_settings: dict) -> None:
    """Submit `turbogap predict` (atoms_in.xyz/input already written in workdir)
    as its own SLURM job via `sbatch --wait`, and block until it finishes.

    Unlike VASP's per-structure array job (_submit_vasp_array_and_wait in
    active_learning.py), turbogap predict runs once over all structures, so
    this is a single job — no array, no squeue polling needed; `sbatch --wait`
    blocks and exits with the job's own exit code.
    """
    account   = gap_settings.get("slurm_account", "")
    partition = gap_settings.get("slurm_partition", "small")
    time_tg   = gap_settings.get("slurm_time_turbogap", "01:00:00")
    ntasks    = int(gap_settings.get("slurm_ntasks_turbogap", 1))
    setup     = gap_settings.get("slurm_setup_turbogap", "")

    account_line = f"#SBATCH --account={account}" if account else ""
    abs_workdir = workdir.resolve()
    script_path = workdir / "turbogap_job.sh"
    script_path.write_text(f"""\
#!/bin/bash -l
#SBATCH --job-name=turbogap_predict
{account_line}
#SBATCH --partition={partition}
#SBATCH --time={time_tg}
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

cd "{abs_workdir}"
{turbogap_command} predict
""")

    cmd = ["sbatch", "--wait", str(abs_workdir / "turbogap_job.sh")]
    print(f"  Submitting turbogap SLURM job: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)

    log_parts = [p.read_text() for p in (workdir / "job.out", workdir / "job.err") if p.exists()]
    (workdir / "turbogap.log").write_text("".join(log_parts))

    if result.returncode != 0:
        # job.out/job.err only exist if the job actually started; a failure
        # here (e.g. transient slurmctld error) happens at the `sbatch`
        # submission step itself, so the real reason is in sbatch's own
        # stdout/stderr, not in turbogap.log.
        raise RuntimeError(
            f"turbogap SLURM job failed (exit {result.returncode}); "
            f"sbatch stdout: {result.stdout.strip()!r}; sbatch stderr: {result.stderr.strip()!r}; "
            f"see {workdir / 'turbogap.log'}"
        )
    print(f"  {result.stdout.strip()}")


def run_predict(frames: list, workdir: Path, gap_settings: dict) -> Path:
    """Run `turbogap predict` on frames (already-loaded ASE Atoms) inside
    workdir; returns the path to the raw trajectory_out.xyz it writes. If
    workdir/trajectory_out.xyz already exists (e.g. from a prior run resumed
    after a crash, or one run manually outside this driver), turbogap is not
    re-run and the existing file is returned as-is.

    gap_settings:
      turbogap_command : binary/launch command, 'predict' is appended here,
                          not part of the command (default: srun {DEFAULT_TURBOGAP})
      turbogap_mode     : 'inline' (default) — run directly as a subprocess
                          using the caller's current allocation; 'sbatch' —
                          submit as its own SLURM job (blocks until done, see
                          slurm_* keys below)
      slurm_account, slurm_partition, slurm_time_turbogap, slurm_ntasks_turbogap,
      slurm_setup_turbogap : SLURM job settings, used only when turbogap_mode=sbatch
    """
    workdir.mkdir(parents=True, exist_ok=True)

    gap_copy = workdir / "gap_files"
    if not gap_copy.exists():
        shutil.copytree(GAP_FILES_DIR, gap_copy)

    write(str(workdir / "atoms_in.xyz"), _bare_atoms(frames), format="extxyz")

    (workdir / "input").write_text(
        INPUT_TEMPLATE.format(
            pot_file=POT_FILE, n_species=N_SPECIES, species=SPECIES, masses=MASSES
        )
    )

    out = workdir / "trajectory_out.xyz"
    if out.exists():
        print(f"  {out} already exists — skipping turbogap predict.")
        return out

    turbogap_command = gap_settings.get("turbogap_command", f"srun {DEFAULT_TURBOGAP}")
    turbogap_mode = gap_settings.get("turbogap_mode", "inline")
    if turbogap_mode == "sbatch":
        _submit_turbogap_sbatch(workdir, turbogap_command, gap_settings)
    elif turbogap_mode == "inline":
        _run_turbogap_inline(workdir, turbogap_command)
    else:
        raise ValueError(f"unknown turbogap_mode {turbogap_mode!r} (expected 'inline' or 'sbatch')")

    if not out.exists():
        raise RuntimeError(f"turbogap predict did not produce {out}")
    return out


def _merge_labels(orig_frames: list, gap_frames: list) -> list:
    """Reattach config-type metadata from orig_frames onto gap_frames (by
    index — turbogap never saw it), drop turbogap-only fields with no
    reference-file counterpart (local_energy, volume), and set
    free_energy = energy (matching the VASP-tagged convention)."""
    merged = []
    for orig, gap in zip(orig_frames, gap_frames):
        energy = gap.get_potential_energy()
        gap.calc = SinglePointCalculator(
            gap,
            energy=energy,
            free_energy=energy,
            forces=gap.get_forces(),
            stress=gap.get_stress(),
        )
        gap.arrays.pop("local_energy", None)
        gap.info.pop("volume", None)

        for key in METADATA_KEYS:
            if key in orig.info:
                gap.info[key] = orig.info[key]

        merged.append(gap)
    return merged


def tag_with_gap(input_xyz: Path, outdir: Path, workdir: Path, turbogap: str) -> Path:
    """Re-evaluate input_xyz with the GAP potential; writes the result (same
    filename, in extended-XYZ) to outdir and returns its path."""
    outdir.mkdir(parents=True, exist_ok=True)

    orig_frames = read(str(input_xyz), index=":")
    print(f"Re-evaluating {input_xyz} with GAP potential ({POT_FILE}): {len(orig_frames)} configuration(s)")

    traj_out = run_predict(orig_frames, workdir, {"turbogap_command": turbogap})
    gap_frames = read(str(traj_out), index=":")
    if len(gap_frames) != len(orig_frames):
        raise RuntimeError(
            f"frame count mismatch: {len(orig_frames)} input vs {len(gap_frames)} turbogap output"
        )

    frames = _merge_labels(orig_frames, gap_frames)
    out_path = outdir / input_xyz.name
    write(str(out_path), frames, format="extxyz")
    print(f"  Wrote {len(frames)} configs -> {out_path}")
    return out_path


def _voigt(tensor_3x3) -> np.ndarray:
    """[xx, yy, zz, yz, xz, xy] from a symmetric 3x3 tensor — matches
    src/utils/convert_format.py's _parse_virial_from_atoms convention."""
    t = np.asarray(tensor_3x3)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])


def check_errors_gap(input_xyz: Path, gap_xyz: Path, outdir: Path) -> Path | None:
    """Compare gap_xyz (GAP-predicted) against input_xyz (DFT reference),
    config-by-config in file order — valid because _merge_labels/tag_with_gap
    keep the two files in lockstep. Reuses check_errors.py's accumulation and
    report format:
      - "Energy"/"Energy per atom" — from free_energy (the training-label
        energy in this repo's convention, see module docstring), not energy.
      - "Forces" — per-atom force-difference vector norm.
      - "Stresses (in energy units)" — our `virial` field (eV); named to
        match check_errors.py's CFG-derived report, where "stress" (CFG's
        PlusStress) is itself energy-unit virial, not pressure.
      - "Virial stresses (in pressure units)" — our `stress` field (eV/Å³,
        converted to GPa), i.e. the actual pressure-unit stress tensor.
    Returns the report path, or None if input_xyz has no DFT labels at all
    (e.g. a bare geometry-only file) — nothing to compare against.
    """
    orig_frames = read(str(input_xyz), index=":")
    gap_frames = read(str(gap_xyz), index=":")
    if len(orig_frames) != len(gap_frames):
        raise RuntimeError(
            f"frame count mismatch: {len(orig_frames)} in {input_xyz} vs {len(gap_frames)} in {gap_xyz}"
        )

    e_diffs, epa_diffs = [], []
    frc_acc, virial_acc, stress_acc = _new_accum(), _new_accum(), _new_accum()
    n_e = 0

    for orig, gap in zip(orig_frames, gap_frames):
        n = len(orig)
        has_energy = orig.calc is not None and orig.calc.results.get("free_energy") is not None
        has_forces = orig.calc is not None and orig.calc.results.get("forces") is not None
        has_virial = "virial" in orig.info
        has_stress = orig.calc is not None and orig.calc.results.get("stress") is not None

        if has_energy:
            diff = gap.calc.results["free_energy"] - orig.calc.results["free_energy"]
            e_diffs.append(diff)
            epa_diffs.append(diff / n)
            n_e += 1

        if has_forces:
            d = gap.get_forces() - orig.get_forces()
            dltsq_atoms = np.sum(d * d, axis=1)
            valsq_atoms = np.sum(orig.get_forces() ** 2, axis=1)
            for dltsq, valsq in zip(dltsq_atoms, valsq_atoms):
                _accumulate(frc_acc, float(np.sqrt(dltsq)), float(valsq))

        if has_virial:
            v_ref = _voigt(orig.info["virial"])
            v_pred = _voigt(gap.info["virial"])
            dltsq = _frob_normsq(v_pred - v_ref)
            valsq = _frob_normsq(v_ref)
            _accumulate(virial_acc, float(np.sqrt(dltsq)), valsq)

        if has_stress:
            s_ref = orig.get_stress() * EV_A3_TO_GPA
            s_pred = gap.get_stress() * EV_A3_TO_GPA
            dltsq = _frob_normsq(s_pred - s_ref)
            valsq = _frob_normsq(s_ref)
            _accumulate(stress_acc, float(np.sqrt(dltsq)), valsq)

    if not (n_e or frc_acc["count"] or virial_acc["count"] or stress_acc["count"]):
        print(f"  No DFT labels (free_energy/forces/virial/stress) found in {input_xyz}; skipping error report.")
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    report = outdir / "errors.txt"
    _write_lammps_report(report, e_diffs, epa_diffs, frc_acc, virial_acc, stress_acc, n_e)
    print(f"\nError report written to: {report}")
    _print_summary(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True,
        help="Extended-XYZ file to re-evaluate (e.g. data/VASP_tagged/extxyz/train-tgap.xyz)",
    )
    parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help=f"Output directory for the re-evaluated extxyz (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--workdir", default=None,
        help=f"Scratch directory for the turbogap run (default: {DEFAULT_WORKDIR}/<input_stem>)",
    )
    parser.add_argument(
        "--turbogap", default=DEFAULT_TURBOGAP,
        help="turbogap binary/launch command, e.g. 'srun /path/to/turbogap'",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    workdir = Path(args.workdir) if args.workdir else Path(DEFAULT_WORKDIR) / input_path.stem

    out_path = tag_with_gap(input_path, Path(args.outdir), workdir, args.turbogap)
    check_errors_gap(input_path, out_path, workdir)


if __name__ == "__main__":
    main()
