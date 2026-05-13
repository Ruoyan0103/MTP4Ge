"""
Active learning loop for MTP potential refinement.

Two modes:

  --auto (default when auto_labeled: true in al-config)
    The candidate pool (candidate_pool.cfg) is already DFT-labeled.
    Each iteration: select_add → merge directly → retrain. No human pause.
    Selected configs are removed from the pool after each iteration.

  Interactive (default when auto_labeled: false)
    Requires user to run DFT and provide labelled.cfg before each retrain.

Usage:
    python src/active_learning.py                     # auto mode (default)
    python src/active_learning.py --no-auto           # interactive/DFT mode
    python src/active_learning.py --pot results/potentials/pot.almtp --max-iter 20
"""

import argparse
import re
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


def remove_from_pool(selected_cfg: Path, pool_cfg: Path) -> int:
    """Remove configs that appear in selected_cfg from pool_cfg (matched by content block).

    Returns the number of configs remaining in the pool after removal.
    """
    def _parse_blocks(text: str) -> list[str]:
        # Split on BEGIN_CFG boundaries, keep the delimiter
        parts = re.split(r"(?=BEGIN_CFG\b)", text)
        return [p.strip() for p in parts if p.strip().startswith("BEGIN_CFG")]

    selected_blocks = set(_parse_blocks(selected_cfg.read_text()))
    pool_text = pool_cfg.read_text()
    pool_blocks = _parse_blocks(pool_text)

    remaining = [b for b in pool_blocks if b not in selected_blocks]
    removed = len(pool_blocks) - len(remaining)

    pool_cfg.write_text("\n\n".join(remaining) + ("\n" if remaining else ""))
    print(f"  Pool updated: removed {removed}, {len(remaining)} configs remain.")
    return len(remaining)


def retrain(train_cfg_path: str, pot_path: str, train_config: str, iteration: int) -> None:
    from src.train import load_config, run_training
    cfg = load_config(train_config)
    overrides = {
        "train_cfg": train_cfg_path,
        "output_potential": pot_path,
        "seed": iteration,
    }
    run_training(cfg, overrides)


def run_active_learning(
    train_cfg: dict,
    al_cfg: dict,
    pot_path: str,
    max_iter: int,
    auto: bool,
) -> None:
    mlp = train_cfg["mlp_binary"]
    train_data = Path(train_cfg["train_cfg"])
    pool_path = Path(al_cfg["preselected_cfg"])
    grade_threshold = al_cfg.get("grade_threshold", 2.0)
    al_outdir = Path("results/active_learning")
    al_outdir.mkdir(parents=True, exist_ok=True)

    if not pool_path.exists():
        print(f"ERROR: candidate pool not found: {pool_path}", file=sys.stderr)
        if auto:
            print("  Run: python src/convert.py --input <xyz> --outdir data/ --mode pool")
        else:
            print("  Provide unlabelled candidate structures in CFG format.")
        sys.exit(1)

    mode_label = "auto (self-labeled pool)" if auto else "interactive (DFT required)"
    print(f"Active learning mode: {mode_label}")
    print(f"  Training set  : {train_data}  ({_count_cfg(train_data)} configs)")
    print(f"  Candidate pool: {pool_path}  ({_count_cfg(pool_path)} configs)")
    print(f"  Grade threshold: {grade_threshold}  |  Max iterations: {max_iter}")

    for iteration in range(1, max_iter + 1):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration}/{max_iter}")
        print(f"{'='*60}")

        iter_dir = al_outdir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)

        selected_cfg = iter_dir / "selected.cfg"

        n_selected = select_add(
            mlp, pot_path, str(train_data), str(pool_path), selected_cfg
        )

        if n_selected == 0:
            print(f"\nNo extrapolative configurations found (grade threshold={grade_threshold}).")
            print("Active learning converged.")
            break

        print(f"\n  {n_selected} configurations selected.")

        if auto:
            # Candidates are already labeled — merge directly and update pool
            merge_cfg(selected_cfg, train_data)
            remaining = remove_from_pool(selected_cfg, pool_path)
            if remaining == 0:
                print("  Candidate pool exhausted.")

        else:
            # Interactive: pause for DFT labeling
            print(f"  Selected structures: {selected_cfg}")
            print("\n  *** ACTION REQUIRED ***")
            print(f"  Run DFT on the structures in:\n    {selected_cfg}")
            print(f"  Add labeled CFG data to:\n    {iter_dir}/labelled.cfg")
            print("  Then press ENTER to continue (or Ctrl-C to abort).")
            input()

            labelled = iter_dir / "labelled.cfg"
            if not labelled.exists():
                print(f"ERROR: labelled.cfg not found at {labelled}", file=sys.stderr)
                print("  Stopping. Re-run from iteration", iteration)
                sys.exit(1)
            merge_cfg(labelled, train_data)

        print(f"\n  Retraining potential  ({_count_cfg(train_data)} configs total)...")
        retrain(str(train_data), pot_path, DEFAULT_TRAIN_CONFIG, iteration)

    print(f"\nActive learning complete. Final training set: {_count_cfg(train_data)} configs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning loop for MTP refinement")
    parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG, help="Training YAML config")
    parser.add_argument("--al-config", default=DEFAULT_AL_CONFIG, help="Active learning YAML config")
    parser.add_argument("--pot", help="Current potential path (overrides config)")
    parser.add_argument("--max-iter", type=int, help="Maximum AL iterations (overrides al-config)")
    parser.add_argument(
        "--auto",
        action="store_true",
        default=None,
        help="Auto mode: candidate pool is already labeled (no DFT pause)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Interactive mode: pause each iteration for DFT labeling",
    )
    args = parser.parse_args()

    train_cfg = _load(args.config)
    al_cfg = _load(args.al_config)

    pot_path = args.pot or train_cfg["output_potential"]
    max_iter = args.max_iter or al_cfg.get("max_iterations", 20)

    # Resolve auto flag: CLI takes priority, then al-config, then default True
    if args.no_auto:
        auto = False
    elif args.auto:
        auto = True
    else:
        auto = al_cfg.get("auto_labeled", True)

    run_active_learning(train_cfg, al_cfg, pot_path, max_iter, auto=auto)


if __name__ == "__main__":
    main()
