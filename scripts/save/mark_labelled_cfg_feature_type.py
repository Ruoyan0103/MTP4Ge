"""Label each structure in a DFT labelled.cfg with its defect-type feature_type.

labelled.cfg (produced by the VASP-labelling step of the thermal active-learning
loop) preserves the same structure order as the selected.cfg it was labelled
from. This script re-derives, for each structure in selected.cfg, which
md_NNN/preselected.cfg MD run it came from (same content-matching approach as
mark_selected_cfg_source.py), maps that run number to a defect-type label, and
writes labelled_marked.cfg with a `Feature feature_type <label>` line added to
each structure, applying the labels positionally (index i in labelled.cfg <->
index i in selected.cfg).

The run-number -> label mapping is either:
  --categories "V,1NN,2NN,H,B,T,X"   (default) -- run number mod len(categories)
    e.g. pot26_v3: md_000/007 -> V, md_001/008 -> 1NN, ..., md_006/013 -> X
  --run-map "0-2:1NN,3-5:2NN,6-7:B"  -- explicit ranges, for uneven group sizes
    e.g. pot26_v5: md_000-002 -> 1NN, md_003-005 -> 2NN, md_006-007 -> B

Usage:
    python scripts/save/mark_labelled_cfg_feature_type.py <al_iter_dir> <labelled_cfg> [--out labelled_marked.cfg] [--categories ...] [--run-map ...]

<al_iter_dir> is the LAMMPS-selection iteration directory containing
selected.cfg and the md_*/preselected.cfg runs, e.g.
results/active_learning/pot26_v3/thermal_parallel/iter_001

<labelled_cfg> is the DFT-labelled CFG file with the same structure order,
e.g. data/pot26_v3/thermal_dft/iter_001/labelled.cfg
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_selected_cfg_source import parse_cfg_blocks, normalize

DEFAULT_CATEGORIES = ["V", "1NN", "2NN", "H", "B", "T", "X"]


def parse_run_map(spec: str) -> dict[int, str]:
    """Parse "0-2:1NN,3-5:2NN,6-7:B" into {0:'1NN', 1:'1NN', 2:'1NN', 3:'2NN', ...}."""
    mapping: dict[int, str] = {}
    for part in spec.split(","):
        rng, label = part.split(":")
        if "-" in rng:
            lo, hi = rng.split("-")
            for r in range(int(lo), int(hi) + 1):
                mapping[r] = label
        else:
            mapping[int(rng)] = label
    return mapping


def build_index_to_category(
    al_iter_dir: Path,
    category_for: Callable[[int], str],
    selected_name: str,
    md_glob: str,
    preselected_name: str,
) -> list[str]:
    source_index: dict[str, list[str]] = {}
    md_dirs = sorted(d for d in al_iter_dir.glob(md_glob) if d.is_dir())
    for md_dir in md_dirs:
        pre_path = md_dir / preselected_name
        if not pre_path.exists():
            continue
        for block in parse_cfg_blocks(pre_path):
            source_index.setdefault(normalize(block), []).append(md_dir.name)

    selected_path = al_iter_dir / selected_name
    if not selected_path.exists():
        sys.exit(f"selected.cfg not found: {selected_path}")

    index_to_category = []
    for i, block in enumerate(parse_cfg_blocks(selected_path)):
        candidates = source_index.get(normalize(block), [])
        if not candidates:
            sys.exit(f"selected.cfg structure [{i}]: no matching md_*/{preselected_name} found")

        run_numbers = sorted(int(name.split("_")[-1]) for name in candidates)
        cats = {category_for(num) for num in run_numbers}
        if len(cats) > 1:
            sys.exit(
                f"selected.cfg structure [{i}]: sources {candidates} map to "
                f"conflicting categories {cats}"
            )
        category = cats.pop()
        index_to_category.append(category)
        print(f"  [{i}] {','.join(candidates)} -> {category}")

    return index_to_category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("al_iter_dir", type=Path)
    parser.add_argument("labelled_cfg", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Default: <labelled_cfg's dir>/labelled_marked.cfg")
    parser.add_argument("--selected", default="selected.cfg")
    parser.add_argument("--md-glob", default="md_*")
    parser.add_argument("--preselected-name", default="preselected.cfg")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                         help="Comma-separated labels, indexed by run-number mod len(categories). "
                              "Ignored if --run-map is given.")
    parser.add_argument("--run-map", default=None,
                         help="Explicit run-number ranges, e.g. '0-2:1NN,3-5:2NN,6-7:B'. "
                              "Overrides --categories for uneven group sizes.")
    args = parser.parse_args()

    if args.run_map:
        run_map = parse_run_map(args.run_map)
        def category_for(run_number: int) -> str:
            if run_number not in run_map:
                sys.exit(f"run number {run_number} not covered by --run-map {args.run_map!r}")
            return run_map[run_number]
    else:
        categories = args.categories.split(",")
        def category_for(run_number: int) -> str:
            return categories[run_number % len(categories)]

    print(f"Resolving source md_* run for each selected.cfg structure in {args.al_iter_dir} ...")
    index_to_category = build_index_to_category(
        args.al_iter_dir, category_for, args.selected, args.md_glob, args.preselected_name
    )

    labelled_blocks = parse_cfg_blocks(args.labelled_cfg)
    if len(labelled_blocks) != len(index_to_category):
        sys.exit(
            f"labelled.cfg has {len(labelled_blocks)} structures but "
            f"selected.cfg has {len(index_to_category)} -- orders don't match"
        )

    out_blocks = []
    for block, category in zip(labelled_blocks, index_to_category):
        lines = block.splitlines()
        assert lines[-1].strip() == "END_CFG"
        lines.insert(-1, f" Feature   feature_type\t{category}")
        out_blocks.append("\n".join(lines))

    out_path = args.out or (args.labelled_cfg.parent / "labelled_marked.cfg")
    out_path.write_text("\n".join(out_blocks) + "\n")
    print(f"\nWrote {len(out_blocks)} structures to {out_path}")


if __name__ == "__main__":
    main()
