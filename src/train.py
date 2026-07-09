"""
Train an MTP potential using the mlp binary.

Usage:
    python src/train.py [--config config/training.yaml] [--template mtp_templates/16.almtp]
                        [--train-cfg data/train.cfg] [--output results/potentials/pot.almtp]
"""

import argparse
import datetime
import shlex
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

    mlp = c.get("mlp_mpi_binary") or c["mlp_binary"]
    template = c["mtp_template"]
    train_cfg = c["train_cfg"]
    output = c["output_potential"].replace("{date}", datetime.date.today().isoformat())
    iteration_limit = c.get("iteration_limit", 500)
    tolerance = c.get("tolerance", 1e-8)
    al_mode = c.get("al_mode", "cfg")
    energy_weight = c.get("energy_weight", 1.0)
    force_weight = c.get("force_weight", 0.01)
    stress_weight = c.get("stress_weight", 0.001)
    weight_scaling = c.get("weight_scaling", 1)
    weight_scaling_forces = c.get("weight_scaling_forces", 0)
    init_random = c.get("init_random", False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    cmd = shlex.split(mlp) + [
        "train", template, train_cfg,
        f"--save_to={output}",
        f"--iteration_limit={iteration_limit}",
        f"--tolerance={tolerance}",
        f"--al_mode={al_mode}",
        f"--energy_weight={energy_weight}",
        f"--force_weight={force_weight}",
        f"--stress_weight={stress_weight}",
        f"--weight_scaling={weight_scaling}",
        f"--weight_scaling_forces={weight_scaling_forces}",
    ]
    cmd.append(f"--init_random={'true' if init_random else 'false'}")
    log_path = c.get("log_path")

    print("Running:", " ".join(str(x) for x in cmd))
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_fh:
            result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(open(log_path).read(), end="", file=sys.stderr)
            print(f"ERROR: mlp train exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
    else:
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {
        "mtp_template": args.template,
        "train_cfg": args.train_cfg,
        "output_potential": args.output,
        "iteration_limit": args.iteration_limit,
    }
    run_training(cfg, overrides)


if __name__ == "__main__":
    main()
