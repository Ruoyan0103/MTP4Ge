"""LAMMPS pair_style block for the LAMMPS backends of src/physical_validation/.

  lammps     --pot is an mtp+nlh .mtp/.almtp file: the `pair_style hybrid/overlay mtp nlh` block already
             written in config/lammps/*.in, elastic_constant/potential.mod and melting_point.py.
  lammps-sw  --pot is a directory holding nlh-SW.table and Ge_3body.sw
             (results/potentials/ref/sw_nordlund1998_nlh): the SW cascade setup of
             ml4cascades/potentials/sw.py, three-body-only SW (Nordlund 1998, A = 0) + the NLH pair table.

The potential kind is read from --pot (file = MTP, directory = SW), so the input-rendering code only has
to pass its rendered text through apply(); --backend lammps-sw in the scripts just checks --pot (check_sw)
and then takes the ordinary LAMMPS code path.
"""

import re
from pathlib import Path

SW_TABLE = "nlh-SW.table"
SW_FILE = "Ge_3body.sw"
# same table resolution as the SW cascades (memory: several GB per MPI rank)
SW_TABLE_N = 100000000

_MTP_BLOCK = re.compile(r"^pair_style\s+hybrid/overlay mtp nlh\n(?:pair_coeff[^\n]*\n)+", re.M)


def is_sw(pot) -> bool:
    return pot is not None and Path(pot).is_dir()


def check_sw(pot) -> None:
    """Exit unless `pot` is a directory with both SW files (for --backend lammps-sw)."""
    d = Path(pot)
    missing = [f for f in (SW_TABLE, SW_FILE) if not (d / f).is_file()]
    if not d.is_dir() or missing:
        raise SystemExit(f"--backend lammps-sw needs --pot <dir with {SW_TABLE} and {SW_FILE}>, "
                         f"e.g. results/potentials/ref/sw_nordlund1998_nlh; got {pot}")


def sw_block(pot) -> str:
    d = Path(pot).absolute()
    return (f"pair_style  hybrid/overlay table linear {SW_TABLE_N} sw\n"
            f"pair_coeff  * * table {d / SW_TABLE} NLH_GE\n"
            f"pair_coeff  * * sw {d / SW_FILE} Ge\n")


def apply(text: str, pot) -> str:
    """Return a rendered LAMMPS input with its mtp+nlh block replaced by the SW block if `pot` is SW."""
    if not is_sw(pot):
        return text
    new, n = _MTP_BLOCK.subn(lambda m: sw_block(pot), text)
    if n != 1:
        raise RuntimeError(f"expected one 'pair_style hybrid/overlay mtp nlh' block, found {n}")
    return new


def label(pot) -> str:
    return "SW 1998 + NLH table" if is_sw(pot) else "mtp+nlh"
