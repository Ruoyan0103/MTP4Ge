"""Combine labelled_marked.cfg structures across iterations/potentials by feature_type.

Reads every labelled_marked.cfg matching the given glob(s) (each structure
carries a `Feature feature_type <label>` line added by
mark_labelled_cfg_feature_type.py), groups structures by that label, and
writes one concatenated CFG file per label into --out-dir (e.g. B.cfg, H.cfg,
X.cfg, T.cfg, 1NN.cfg, 2NN.cfg).

Usage:
    python scripts/combine_labelled_by_feature_type.py <glob> [<glob> ...] --out-dir <dir> [--labels B,H,X,T,1NN,2NN]
"""
import argparse
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_selected_cfg_source import parse_cfg_blocks


def extract_feature_type(block: str) -> str | None:
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "Feature" and parts[1] == "feature_type":
            return parts[2]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Glob pattern(s) matching labelled_marked.cfg files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--labels", default="B,H,X,T,1NN,2NN",
                         help="Comma-separated labels to write out; other labels found are reported but skipped")
    args = parser.parse_args()

    wanted = set(args.labels.split(","))

    paths = sorted({Path(p) for pattern in args.patterns for p in glob(pattern, recursive=True)})
    if not paths:
        sys.exit(f"No files matched: {args.patterns}")

    by_label: dict[str, list[str]] = {}
    skipped: dict[str, int] = {}
    for path in paths:
        blocks = parse_cfg_blocks(path)
        for block in blocks:
            label = extract_feature_type(block)
            if label is None:
                sys.exit(f"{path}: structure missing Feature feature_type")
            if label in wanted:
                by_label.setdefault(label, []).append(block)
            else:
                skipped[label] = skipped.get(label, 0) + 1
        print(f"  {path}: {len(blocks)} structures")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(wanted):
        blocks = by_label.get(label, [])
        if not blocks:
            print(f"WARNING: no structures found for label {label!r}, skipping {label}.cfg")
            continue
        out_path = args.out_dir / f"{label}.cfg"
        out_path.write_text("\n".join(blocks) + "\n")
        print(f"Wrote {len(blocks)} structures -> {out_path}")

    if skipped:
        print("\nSkipped labels not in --labels:")
        for label, count in sorted(skipped.items()):
            print(f"  {label}: {count} structures")


if __name__ == "__main__":
    main()
