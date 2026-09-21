"""
Split data/test_set.cfg into val.cfg and test.cfg, 50/50, via a seeded
random shuffle. data/test_set.txt (one source OUTCAR path per line, in the
same order as the CFG blocks) is split the same way into val.txt/test.txt,
so each split's structures can still be traced back to their DFT source.

CFG blocks are sliced out of the raw file text (BEGIN_CFG .. up to the next
BEGIN_CFG) rather than re-parsed and rewritten, so Energy/PlusStress/Feature
fields are preserved byte-for-byte.

Usage:
    python src/utils/split_test.py
    python src/utils/split_test.py --cfg data/test_set.cfg --paths data/test_set.txt --seed 42
"""

import argparse
import random
import re
from pathlib import Path


def _split_cfg_blocks(text: str) -> list[str]:
    starts = [m.start() for m in re.finditer(r"^BEGIN_CFG", text, re.MULTILINE)]
    if not starts:
        raise ValueError("No BEGIN_CFG blocks found")
    ends = starts[1:] + [len(text)]
    return [text[s:e] for s, e in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default="data/test_set.cfg",
                        help="CFG file to split (default: data/test_set.cfg)")
    parser.add_argument("--paths", default="data/test_set.txt",
                        help="Source-path listing matching --cfg, one line per "
                             "structure (default: data/test_set.txt)")
    parser.add_argument("--outdir", default="data",
                        help="Directory to write val.cfg/val.txt and test.cfg/test.txt "
                             "into (default: data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    blocks = _split_cfg_blocks(Path(args.cfg).read_text())
    paths = Path(args.paths).read_text().splitlines()
    if len(blocks) != len(paths):
        raise ValueError(
            f"{args.cfg} has {len(blocks)} structures but {args.paths} has "
            f"{len(paths)} lines"
        )

    order = list(range(len(blocks)))
    random.seed(args.seed)
    random.shuffle(order)

    mid = len(order) // 2
    splits = {"val": order[:mid], "test": order[mid:]}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, idxs in splits.items():
        (outdir / f"{name}.cfg").write_text("".join(blocks[i] for i in idxs))
        (outdir / f"{name}.txt").write_text("\n".join(paths[i] for i in idxs) + "\n")

    print(f"Total: {len(blocks)}  ->  val: {len(splits['val'])}, test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
