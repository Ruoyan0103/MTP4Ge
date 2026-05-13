"""
Train an MTP potential using the mlp binary.

Usage:
    python src/train.py [--config config/training.yaml] [--template mtp_templates/16.almtp]
                        [--train-cfg data/train.cfg] [--output results/potentials/pot.almtp]
                        [--seed 1]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = "config/training.yaml"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_training(cfg: dict, overrides: dict) -> None:
    c = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}

    mlp = c["mlp_binary"]
    template = c["mtp_template"]
    train_cfg = c["train_cfg"]
    output = c["output_potential"]
    iteration_limit = c.get("iteration_limit", 500)
    al_mode = c.get("al_mode", "nbh")
    energy_weight = c.get("energy_weight", 1.0)
    force_weight = c.get("force_weight", 0.01)
    stress_weight = c.get("stress_weight", 0.001)
    seed = c.get("seed", None)

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        mlp, "train", template, train_cfg,
        f"--save_to={output}",
        f"--iteration_limit={iteration_limit}",
        f"--al_mode={al_mode}",
        f"--energy-weight={energy_weight}",
        f"--force-weight={force_weight}",
        f"--stress-weight={stress_weight}",
    ]
    if seed is not None:
        cmd.append(f"--init_random={seed}")

    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: mlp train exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Training complete. Potential saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTP potential via mlp binary")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--template", help="Override MTP template (.almtp)")
    parser.add_argument("--train-cfg", help="Override training CFG path")
    parser.add_argument("--output", help="Override output potential path")
    parser.add_argument("--iteration-limit", type=int, help="Override iteration limit")
    parser.add_argument("--seed", type=int, help="Random seed for initialization")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {
        "mtp_template": args.template,
        "train_cfg": args.train_cfg,
        "output_potential": args.output,
        "iteration_limit": args.iteration_limit,
        "seed": args.seed,
    }
    run_training(cfg, overrides)


if __name__ == "__main__":
    main()
