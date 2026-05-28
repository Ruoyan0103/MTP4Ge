"""
LAMMPS molecular dynamics testing wrapper.

Generates a run-specific mlip.ini pointing to the trained potential,
then launches LAMMPS with the configured input script.

Usage:
    python src/md_test.py --pot results/potentials/pot.almtp \\
        --template config/lammps/md_nvt.in \\
        --T 1200 --steps 50000 --outdir results/md/run01

MD simulation details (ensemble, box size, thermostat) are controlled
via the LAMMPS template in config/lammps/md_nvt.in.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


LAMMPS_BINARY = "lmp_mpi"


def write_mlip_ini(pot_path: str, run_dir: Path) -> Path:
    ini = run_dir / "mlip.ini"
    ini.write_text(
        f"mtp-filename = {Path(pot_path).resolve()}\n"
        "select = false\n"
    )
    return ini


def patch_lammps_input(template: Path, run_dir: Path, T: float, steps: int) -> Path:
    text = template.read_text()
    text = text.replace("${T_RUN}", str(T))
    text = text.replace("${RUN_STEPS}", str(steps))
    out = run_dir / "md.in"
    out.write_text(text)
    return out


def run_md(
    pot_path: str,
    template: Path,
    T: float,
    steps: int,
    outdir: Path,
    lammps: str = LAMMPS_BINARY,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_mlip_ini(pot_path, outdir)
    md_in = patch_lammps_input(template, outdir, T, steps)

    cmd = [lammps, "-in", str(md_in)]
    print("Running:", " ".join(cmd))
    print(f"  Potential: {pot_path}")
    print(f"  Temperature: {T} K  |  Steps: {steps}")
    result = subprocess.run(cmd, cwd=str(outdir))
    if result.returncode != 0:
        print(f"ERROR: LAMMPS exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"MD complete. Results in: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LAMMPS MD with MTP potential")
    parser.add_argument("--pot", required=True, help="Trained potential (.almtp)")
    parser.add_argument(
        "--template",
        default="config/lammps/md_nvt.in",
        help="LAMMPS input template",
    )
    parser.add_argument("--T", type=float, default=1200.0, help="Temperature (K)")
    parser.add_argument("--steps", type=int, default=50000, help="Number of MD steps")
    parser.add_argument("--outdir", default="results/md/run", help="Output directory")
    parser.add_argument("--lammps", default=LAMMPS_BINARY, help="LAMMPS executable")
    args = parser.parse_args()

    run_md(
        pot_path=args.pot,
        template=Path(args.template),
        T=args.T,
        steps=args.steps,
        outdir=Path(args.outdir),
        lammps=args.lammps,
    )


if __name__ == "__main__":
    main()
