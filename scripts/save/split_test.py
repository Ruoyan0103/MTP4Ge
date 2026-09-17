#!/usr/bin/env python3
"""Shuffle test.xyz and split 50/50 into test_new.xyz and val.xyz."""

import argparse
import random
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="data/test.xyz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = read(args.input, index=":")
    random.seed(args.seed)
    random.shuffle(frames)

    mid = len(frames) // 2
    stem = args.input.replace(".xyz", "")

    write(f"{stem}_new.xyz", frames[:mid])
    write("data/val.xyz", frames[mid:])

    print(f"Total: {len(frames)}  →  test_new: {mid},  val: {len(frames) - mid}")

if __name__ == "__main__":
    main()
