"""
Evaluate MTP potential accuracy against a reference dataset.

Modes:
    errors  — mlp check_errors (RMSE for energy, forces, stresses)
    efs     — mlp calculate_efs (per-configuration E/F/S predictions)

Usage:
    python src/test_errors.py --pot results/potentials/pot.almtp --cfg data/test.cfg
    python src/test_errors.py --pot results/potentials/pot.almtp --cfg data/test.cfg --mode efs
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = "config/training.yaml"


def _mlp_binary(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("mlp_binary", "mlp")


def run_check_errors(mlp: str, pot: str, cfg_path: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report = outdir / "errors.txt"
    cmd = [
        mlp, "check_errors", pot, cfg_path,
        "--log=stdout",
        f"--report_to={report}",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: mlp check_errors exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\nError report written to: {report}")
    _print_summary(report)


def _print_summary(report: Path) -> None:
    if not report.exists():
        return
    text = report.read_text()
    print("\n--- Error Summary ---")
    for line in text.splitlines():
        if any(kw in line.lower() for kw in ("rmse", "mae", "energy", "force", "stress")):
            print(" ", line.strip())
    print("---------------------")


def run_calculate_efs(mlp: str, pot: str, cfg_path: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    out_cfg = outdir / "efs_output.cfg"
    cmd = [
        mlp, "calculate_efs", pot, cfg_path,
        f"--output_cfg={out_cfg}",
        "--log=stdout",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: mlp calculate_efs exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"EFS predictions written to: {out_cfg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MTP potential accuracy")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--pot", required=True, help="Trained potential (.almtp)")
    parser.add_argument("--cfg", required=True, help="Reference dataset (.cfg)")
    parser.add_argument(
        "--mode",
        choices=["errors", "efs"],
        default="errors",
        help="errors: RMSE report; efs: per-config E/F/S output (default: errors)",
    )
    parser.add_argument("--outdir", default="results/errors", help="Output directory")
    args = parser.parse_args()

    mlp = _mlp_binary(args.config)
    outdir = Path(args.outdir)

    if args.mode == "errors":
        run_check_errors(mlp, args.pot, args.cfg, outdir)
    else:
        run_calculate_efs(mlp, args.pot, args.cfg, outdir)


if __name__ == "__main__":
    main()
