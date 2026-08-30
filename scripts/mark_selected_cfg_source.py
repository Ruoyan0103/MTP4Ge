"""Trace which md_NNN/ MD run each structure in selected.cfg came from.

Matches each structure in an active-learning iteration's selected.cfg against
the structures in every md_*/preselected.cfg by exact structure content
(atom data + MV_grade), then writes selected_marked.cfg with a
`Feature source_md <folder>` line recording the originating md_NNN folder(s).

MV_grade alone is not always a unique key: separate md_* runs can capture a
byte-identical structure (e.g. an early snapshot recorded before the
trajectories diverge), so matching is done on the full structure block, and
any structure with more than one matching source folder is reported as
ambiguous rather than guessed at.

Usage:
    python scripts/mark_selected_cfg_source.py <iter_dir> [--selected selected.cfg] [--out selected_marked.cfg]
"""
import argparse
import sys
from pathlib import Path


def parse_cfg_blocks(path: Path) -> list[str]:
    """Split a CFG file into a list of raw BEGIN_CFG..END_CFG block strings."""
    blocks = []
    current: list[str] = []
    in_block = False
    for line in path.read_text().splitlines():
        if line.strip() == "BEGIN_CFG":
            in_block = True
            current = [line]
        elif line.strip() == "END_CFG":
            current.append(line)
            blocks.append("\n".join(current))
            in_block = False
        elif in_block:
            current.append(line)
    return blocks


def normalize(block: str) -> str:
    """Normalize a block for structural comparison: drop trailing per-line whitespace
    and any `Feature` lines, since select_add appends extra Feature lines
    (selected_eqn_inds, strain, type, ...) to selected.cfg that are absent
    from the source md_*/preselected.cfg blocks."""
    return "\n".join(
        line.rstrip()
        for line in block.splitlines()
        if not line.strip().startswith("Feature")
    )


def extract_grade(block: str) -> float | None:
    for line in block.splitlines():
        if "MV_grade" in line:
            return float(line.split()[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iter_dir", type=Path)
    parser.add_argument("--selected", default="selected.cfg")
    parser.add_argument("--out", default="selected_mark.cfg")
    parser.add_argument("--md-glob", default="md_*")
    parser.add_argument("--preselected-name", default="preselected.cfg")
    args = parser.parse_args()

    iter_dir: Path = args.iter_dir
    selected_path = iter_dir / args.selected
    if not selected_path.exists():
        sys.exit(f"selected.cfg not found: {selected_path}")

    source_index: dict[str, list[str]] = {}
    md_dirs = sorted(d for d in iter_dir.glob(args.md_glob) if d.is_dir())
    for md_dir in md_dirs:
        pre_path = md_dir / args.preselected_name
        if not pre_path.exists():
            continue
        for block in parse_cfg_blocks(pre_path):
            key = normalize(block)
            source_index.setdefault(key, []).append(md_dir.name)

    selected_blocks = parse_cfg_blocks(selected_path)

    out_blocks = []
    unmatched = 0
    ambiguous = 0
    for i, block in enumerate(selected_blocks):
        key = normalize(block)
        candidates = source_index.get(key, [])
        grade = extract_grade(block)

        if not candidates:
            unmatched += 1
            source_label = "UNKNOWN"
            print(f"  [{i}] MV_grade={grade}: NO MATCH found in any {args.md_glob}/{args.preselected_name}")
        else:
            source_label = ",".join(candidates)
            note = ""
            if len(candidates) > 1:
                ambiguous += 1
                note = "  (ambiguous: identical structure present in multiple md_* runs)"
            print(f"  [{i}] MV_grade={grade}: {source_label}{note}")

        lines = block.splitlines()
        assert lines[-1].strip() == "END_CFG"
        lines.insert(-1, f" Feature   source_md\t{source_label}")
        out_blocks.append("\n".join(lines))

    out_path = iter_dir / args.out
    out_path.write_text("\n".join(out_blocks) + "\n")

    print(f"\nWrote {len(out_blocks)} structures to {out_path}")
    print(f"  matched: {len(out_blocks) - unmatched}, unmatched: {unmatched}, ambiguous: {ambiguous}")


if __name__ == "__main__":
    main()
