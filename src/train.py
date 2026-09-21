"""
Train an MTP potential using the mlp binary.

Each run creates its own results/potentials/pot_<random_id>/ folder containing:
  - pot.almtp                the trained potential
  - pot_training_info.txt    path to the training CFG used, plus the full mlp train log

(active_learning.py's retrain() instead passes an explicit output_potential/
log_path override, to overwrite a fixed AL potential path in place each
iteration rather than spawning a new pot_<id>/ folder.)

mlp_binary in config/training.yaml is a plain (serial) launch command. To run
training under MPI (e.g. `srun -n 20 ...`), pass --mlp-binary — this is set by
scripts/submit_train.sh, not hardcoded in the YAML, since it's cluster/job
specific.

Usage:
    python src/train.py --train-cfg data/train.cfg [--config config/training.yaml]
                        [--template mtp_templates/16.almtp] [--potentials-dir results/potentials]
                        [--mlp-binary "srun -n 20 /path/to/mlp"]

Example:
    python src/train.py --train-cfg data/train.cfg --template mtp_templates/20.almtp
"""

import argparse
import random
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = "config/training.yaml"
DEFAULT_POTENTIALS_DIR = "results/potentials"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _new_pot_dir(potentials_dir: Path) -> Path:
    potentials_dir.mkdir(parents=True, exist_ok=True)
    while True:
        pot_dir = potentials_dir / f"pot_{random.randint(0, 999999):06d}"
        try:
            pot_dir.mkdir(parents=True)
            return pot_dir
        except FileExistsError:
            continue


def run_training(cfg: dict, overrides: dict) -> None:
    c = {**cfg, **{k: v for k, v in overrides.items() if v is not None}}

    mlp = c["mlp_binary"]
    template = c["mtp_template"]
    train_cfg = c["train_cfg"]
    iteration_limit = c.get("iteration_limit", 500)
    tolerance = c.get("tolerance", 1e-8)
    al_mode = c.get("al_mode", "cfg")
    energy_weight = c.get("energy_weight", 1.0)
    force_weight = c.get("force_weight", 0.01)
    stress_weight = c.get("stress_weight", 0.001)
    weight_scaling = c.get("weight_scaling", 1)
    weight_scaling_forces = c.get("weight_scaling_forces", 0)
    init_random = c.get("init_random", False)

    explicit_output = c.get("output_potential")
    if explicit_output:
        # Used by active_learning.py's retrain(): overwrite a fixed path each
        # AL iteration instead of spawning a new pot_<id>/ folder.
        output = Path(explicit_output)
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output = _new_pot_dir(Path(c.get("potentials_dir", DEFAULT_POTENTIALS_DIR))) / "pot.almtp"

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

    print("Running:", " ".join(str(x) for x in cmd))
    log_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="")
            log_lines.append(line)
    returncode = proc.returncode

    log_path_override = c.get("log_path")
    info_path = Path(log_path_override) if log_path_override else output.with_name(f"{output.stem}_training_info.txt")
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        f.write(f"train_cfg: {train_cfg}\n\n")
        f.write("=== Training log ===\n")
        f.writelines(log_lines)

    if returncode != 0:
        print(f"ERROR: mlp train exited with code {returncode}", file=sys.stderr)
        sys.exit(returncode)

    print(f"Training complete. Potential saved to: {output}")
    print(f"Training info saved to: {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTP potential via mlp binary")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--mlp-binary", help="Override mlp launch command, e.g. 'srun -n 20 /path/to/mlp'")
    parser.add_argument("--template", help="Override MTP template (.almtp)")
    parser.add_argument("--train-cfg", required=True, help="Training CFG path (required)")
    parser.add_argument("--potentials-dir", help="Override root directory for pot_<id>/ output folders")
    parser.add_argument("--iteration-limit", type=int, help="Override iteration limit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {
        "mlp_binary": args.mlp_binary,
        "mtp_template": args.template,
        "train_cfg": args.train_cfg,
        "potentials_dir": args.potentials_dir,
        "iteration_limit": args.iteration_limit,
    }
    run_training(cfg, overrides)


if __name__ == "__main__":
    main()
