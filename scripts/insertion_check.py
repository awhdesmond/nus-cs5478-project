"""Run the insertion-module grid: python scripts/insertion_check.py [--seeds 2] [--errors 0 1 2 4]

Both controllers × clearance {1, 3} mm × friction {0.3, 0.8} × port error × both pegs × seeds.
Writes every trial to results/insertion_check/trials.csv and prints success, time and force per condition.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from harness.insertion_check import insertion_grid, summarise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--errors", type=float, nargs="+", default=[0, 1, 2, 4], help="port error [mm]")
    parser.add_argument("--out", type=Path, default=Path("results/insertion_check"))
    args = parser.parse_args()

    start = time.perf_counter()
    def progress(done, total):
        print(f"\r{done}/{total} trials, {time.perf_counter() - start:.0f} s", end="", file=sys.stderr)

    results = insertion_grid(range(args.seeds), [e / 1000 for e in args.errors], progress=progress)
    print(file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "trials.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200, "display.max_columns", 20):
        print(summarise(results))
    print(f"\nall trials: {args.out / 'trials.csv'}")


if __name__ == "__main__":
    main()
