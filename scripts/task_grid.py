"""Run the full-task grid: python scripts/task_grid.py [--seeds 20] [--workers 8] [--pose-error 0]

clearance {3, 1} mm × friction {0.3, 0.8} × camera noise {off, 2 mm} × seeds, compliant vs stiff on the same
seeds. Writes every trial to <out>/trials.csv and prints the success, comparison, failure and perception tables.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from harness.grid import comparison_table, failure_table, perception_table, success_table, task_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 2.0], help="camera depth noise  [mm]")
    parser.add_argument("--pose-error", type=float, nargs="+", default=[0.0], help="per-object error [mm]")
    parser.add_argument("--out", type=Path, default=Path("results/task_grid"))
    args = parser.parse_args()

    start = time.perf_counter()
    def progress(done, total):
        print(f"\r{done}/{total} trials, {time.perf_counter() - start:.0f} s", end="", file=sys.stderr)

    results = task_grid(range(args.seeds), noises=[n / 1000 for n in args.noise],
                        pose_errors=[e / 1000 for e in args.pose_error],
                        workers=args.workers, progress=progress)
    print(file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "trials.csv", index=False)

    with pd.option_context("display.width", 160, "display.max_rows", 200, "display.max_columns", 20):
        for title, table in (("Success (Wilson 95 % interval), time [s], force [N]", success_table(results)),
                             ("Compliant vs stiff on the same seeds (exact McNemar)", comparison_table(results)),
                             ("Failures by stage", failure_table(results)),
                             ("Perception error against ground truth", perception_table(results))):
            print(f"\n{title}\n{table}")
    print(f"\nall trials: {args.out / 'trials.csv'}")


if __name__ == "__main__":
    main()
