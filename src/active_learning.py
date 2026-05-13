"""
Active learning loop for MTP potential refinement.

Each iteration:
  1. Grade structures in preselected_cfg with mlp calculate_grade
  2. Select extrapolative configurations with mlp select_add
  3. Pause for user to add DFT labels to selected.cfg
  4. Merge labelled configs into train.cfg
  5. Retrain the potential via train.py

Usage:
    python src/active_learning.py [--config config/training.yaml]
                                   [--al-config config/active_learning.yaml]
                                   [--pot results/potentials/pot.almtp]
                                   [--max-iter 10]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_TRAIN_CONFIG = "config/training.yaml"
DEFAULT_AL_CONFIG = "config/active_learning.yaml"


def _load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _count_cfg(path: Path) -> int:
    if not path.exists():
        return 0
    return path.read_text().count("BEGIN_CFG")


def grade(mlp: str, pot: str, train_cfg: str, graded_cfg: Path) -> None:
    cmd = [mlp, "calculate_grade", pot, train_cfg, str(graded_cfg)]
    print("  Grade:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def select_add(mlp: str, pot: str, train_cfg: str, preselected: str, selected: Path) -> int:
    cmd = [mlp, "select_add", pot, train_cfg, preselected, str(selected)]
    print("  Select:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _count_cfg(selected)


def merge_cfg(source: Path, destination: Path) -> None:
    text = source.read_text()
    with open(destination, "a") as f:
        f.write("\n" + text)
    print(f"  Merged {_count_cfg(source)} configs from {source} → {destination}")


def retrain(train_cfg_path: str, pot_path: str, train_config: str, iteration: int) -> None:
    from src.train import load_config, run_training
    cfg = load_config(train_config)
    overrides = {
        "train_cfg": train_cfg_path,
        "output_potential": pot_path,
        "seed": iteration,
    }
    run_training(cfg, overrides)


def run_active_learning(train_cfg: dict, al_cfg: dict, pot_path: str, max_iter: int) -> None:
    mlp = train_cfg["mlp_binary"]
    train_data = Path(train_cfg["train_cfg"])
    preselected = Path(al_cfg["preselected_cfg"])
    grade_threshold = al_cfg.get("grade_threshold", 2.0)
    al_outdir = Path("results/active_learning")
    al_outdir.mkdir(parents=True, exist_ok=True)

    if not preselected.exists():
        print(f"ERROR: preselected_cfg not found: {preselected}", file=sys.stderr)
        print("  Create this file with new (unlabelled) candidate structures in CFG format.")
        sys.exit(1)

    for iteration in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration}/{max_iter}")
        print(f"{'='*60}")

        iter_dir = al_outdir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        graded_cfg = iter_dir / "graded.cfg"
        selected_cfg = iter_dir / "selected.cfg"

        grade(mlp, pot_path, str(train_data), graded_cfg)
        n_selected = select_add(mlp, pot_path, str(train_data), str(preselected), selected_cfg)

        if n_selected == 0:
            print(f"\nNo extrapolative configurations found (threshold={grade_threshold}).")
            print("Active learning converged.")
            break

        print(f"\n  {n_selected} configurations selected for labelling.")
        print(f"  Selected structures: {selected_cfg}")
        print("\n  *** ACTION REQUIRED ***")
        print(f"  Run DFT on the structures in:\n    {selected_cfg}")
        print("  Add energies, forces, and stresses (in CFG format) to:")
        print(f"    {iter_dir}/labelled.cfg")
        print("  Then press ENTER to continue (or Ctrl-C to stop).")
        input()

        labelled = iter_dir / "labelled.cfg"
        if not labelled.exists():
            print(f"ERROR: labelled.cfg not found at {labelled}", file=sys.stderr)
            print("  Stopping active learning. Re-run from iteration", iteration)
            sys.exit(1)

        merge_cfg(labelled, train_data)

        print(f"\n  Retraining potential...")
        retrain(str(train_data), pot_path, DEFAULT_TRAIN_CONFIG, iteration)

    print("\nActive learning loop complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop for MTP refinement")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG, help="Training YAML config")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG, help="Active learning YAML config")
    parser.add_argument("--pot", help="Current potential path (overrides config)")
    parser.add_argument("--max-iter", type=int, help="Maximum AL iterations (overrides al-config)")
    args = parser.parse_args()

    train_cfg = _load(args.config)
    al_cfg = _load(args.al_config)

    pot_path = args.pot or train_cfg["output_potential"]
    max_iter = args.max_iter or al_cfg.get("max_iterations", 10)

    run_active_learning(train_cfg, al_cfg, pot_path, max_iter)


if __name__ == "__main__":
    main()
