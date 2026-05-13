"""
MTP pruning pipeline using NSGA-II multi-objective optimization.

Steps:
  1. extract_problem  — compute XᵀWX / XᵀWy matrices for train and val sets
  2. prune            — run NSGA-II to generate Pareto front of cost vs accuracy
  3. mask_inherited   — apply a chosen mask row to produce a pruned .almtp file

Usage:
    # Full pipeline (all three steps):
    python src/prune.py --pot results/potentials/pot.almtp

    # Apply a specific Pareto-front row after pruning:
    python src/prune.py --step mask --row 5
"""

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = "config/training.yaml"
PRUNE_CONFIG_TEMPLATE = "config/pruning.json"
PRUNE_OUTDIR = Path("results/pruning")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_problem(
    mlp: str,
    pot: str,
    cfg_path: str,
    xtwx_out: Path,
    xtwy_out: Path,
) -> float:
    """Run mlp extract_problem and return ytwy (sum of squared targets)."""
    xtwx_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [mlp, "extract_problem", pot, cfg_path, str(xtwx_out), str(xtwy_out)]
    print("  extract_problem:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"ERROR: extract_problem failed (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    ytwy = _parse_ytwy(result.stdout)
    neigh_count = _parse_neigh(result.stdout)
    return ytwy, neigh_count


def _parse_ytwy(output: str) -> float:
    m = re.search(r"Sum of Squared Ground Truths \(yTWy\):\s*([\d.eE+\-]+)", output)
    if m:
        return float(m.group(1))
    raise ValueError("Could not parse yTWy from extract_problem output")


def _parse_neigh(output: str) -> float:
    m = re.search(r"Average number of neighbors\s*=\s*([\d.eE+\-]+)", output)
    if m:
        return float(m.group(1))
    return 0.0


def build_prune_config(
    template_path: str,
    train_ytwy: float,
    val_ytwy: float,
    neigh_count: float,
    prune_outdir: Path,
    xtwx_train: Path,
    xtwy_train: Path,
    xtwx_val: Path,
    xtwy_val: Path,
    mtp_file: str,
) -> Path:
    with open(template_path) as f:
        cfg = json.load(f)

    cfg["ytwy_train"] = train_ytwy
    cfg["ytwy_val"] = val_ytwy
    cfg["neigh_count"] = neigh_count
    cfg["xtwx_train_file"] = str(xtwx_train.resolve())
    cfg["xtwy_train_file"] = str(xtwy_train.resolve())
    cfg["xtwx_val_file"] = str(xtwx_val.resolve())
    cfg["xtwy_val_file"] = str(xtwy_val.resolve())
    cfg["mtp_file"] = str(Path(mtp_file).resolve())
    cfg["out_dir"] = str((prune_outdir / "pareto").resolve())

    run_config = prune_outdir / "pruning_run.json"
    prune_outdir.mkdir(parents=True, exist_ok=True)
    run_config.write_text(json.dumps(cfg, indent=2))
    print(f"  Pruning config written to: {run_config}")
    return run_config


def run_prune(mlp: str, config_path: Path) -> None:
    cmd = [mlp, "prune", str(config_path)]
    print("  prune:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: prune failed (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def apply_mask(
    mlp: str,
    base_mtp: str,
    config_path: Path,
    population_csv: str,
    row: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        mlp, "mask_inherited",
        base_mtp,
        str(config_path),
        population_csv,
        str(row),
        str(output),
    ]
    print("  mask_inherited:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: mask_inherited failed (code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"  Pruned potential written to: {output}")


def find_pareto_csv(prune_outdir: Path) -> str:
    candidates = sorted(glob.glob(str(prune_outdir / "pareto" / "**" / "pareto_final_*.csv"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(str(prune_outdir / "pareto*" / "pareto_final_*.csv")))
    if not candidates:
        print(
            "ERROR: No pareto_final_*.csv found under results/pruning/.\n"
            "  Run the full pipeline first: python src/prune.py --pot <pot>",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Using Pareto CSV: {candidates[-1]}")
    return candidates[-1]


def pipeline(
    train_cfg_path: str,
    val_cfg_path: str,
    pot_path: str,
    template_mtp: str,
    config_template: str,
    row: int,
    mlp: str,
) -> None:
    print("\n[1/3] Extracting problem matrices...")
    xtwx_train = PRUNE_OUTDIR / "xtwx_train.bin"
    xtwy_train = PRUNE_OUTDIR / "xtwy_train.bin"
    xtwx_val   = PRUNE_OUTDIR / "xtwx_val.bin"
    xtwy_val   = PRUNE_OUTDIR / "xtwy_val.bin"

    train_ytwy, neigh_train = extract_problem(mlp, pot_path, train_cfg_path, xtwx_train, xtwy_train)
    val_ytwy, _ = extract_problem(mlp, pot_path, val_cfg_path, xtwx_val, xtwy_val)
    neigh_count = neigh_train
    print(f"  yTWy train={train_ytwy:.6e}  val={val_ytwy:.6e}  avg_neigh={neigh_count:.3f}")

    print("\n[2/3] Building pruning config and running NSGA-II...")
    run_config = build_prune_config(
        config_template, train_ytwy, val_ytwy, neigh_count,
        PRUNE_OUTDIR, xtwx_train, xtwy_train, xtwx_val, xtwy_val, template_mtp,
    )
    run_prune(mlp, run_config)

    print("\n[3/3] Applying mask to produce pruned potential...")
    pareto_csv = find_pareto_csv(PRUNE_OUTDIR)
    pruned_out = Path("results/potentials/pruned.almtp")
    apply_mask(mlp, template_mtp, run_config, pareto_csv, row, pruned_out)

    print(f"\nPruning complete.")
    print(f"  Full potential : {pot_path}")
    print(f"  Pruned potential: {pruned_out}")
    print(f"  Pareto front CSV: {pareto_csv}")
    print(f"  Selected row: {row}  (use --row N to choose a different Pareto point)")


def mask_only(mlp: str, base_mtp: str, row: int) -> None:
    run_config = PRUNE_OUTDIR / "pruning_run.json"
    if not run_config.exists():
        print(f"ERROR: {run_config} not found. Run the full pipeline first.", file=sys.stderr)
        sys.exit(1)
    pareto_csv = find_pareto_csv(PRUNE_OUTDIR)
    pruned_out = Path("results/potentials/pruned.almtp")
    apply_mask(mlp, base_mtp, run_config, pareto_csv, row, pruned_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="MTP pruning pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training YAML config")
    parser.add_argument("--prune-config", default=PRUNE_CONFIG_TEMPLATE, help="Pruning JSON template")
    parser.add_argument("--pot", help="Trained potential (.almtp) (overrides config)")
    parser.add_argument("--train-cfg", help="Training CFG (overrides config)")
    parser.add_argument("--val-cfg", help="Validation CFG (default: data/val.cfg)")
    parser.add_argument("--template", help="Base MTP template for pruning (overrides config)")
    parser.add_argument("--row", type=int, default=0, help="Pareto front row to select (default: 0)")
    parser.add_argument(
        "--step",
        choices=["all", "extract", "prune", "mask"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    mlp = cfg["mlp_binary"]
    pot = args.pot or cfg["output_potential"]
    train_cfg = args.train_cfg or cfg["train_cfg"]
    val_cfg = args.val_cfg or str(Path(train_cfg).parent / "val.cfg")
    template_mtp = args.template or cfg.get("mtp_template", "mtp_templates/16.almtp")

    if args.step == "mask":
        mask_only(mlp, template_mtp, args.row)
    else:
        pipeline(train_cfg, val_cfg, pot, template_mtp, args.prune_config, args.row, mlp)


if __name__ == "__main__":
    main()
