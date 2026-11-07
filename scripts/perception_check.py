"""Log perception error against ground truth: python scripts/perception_check.py [--seeds 50] [--noise 0 0.002]

Writes every trial's errors to results/perception_check/errors.csv and prints a summary.
"""

import argparse
from pathlib import Path

import pandas as pd

from harness.perception_check import perception_check, summarise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.002], help="depth noise σ [m]")
    parser.add_argument("--out", type=Path, default=Path("results/perception_check"))
    args = parser.parse_args()

    errors = perception_check(range(args.seeds), args.noise)
    args.out.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.out / "errors.csv", index=False)

    print(f"Perception error over {args.seeds} layouts (xy, z in mm; yaw in degrees, square objects only)")
    with pd.option_context("display.width", 140):
        print(summarise(errors))
    print(f"\nall trials: {args.out / 'errors.csv'}")
    failed = errors[~errors.found]
    print("PASS: everything found" if failed.empty else f"FAIL: perception failed on seeds {sorted(set(failed.seed))}")


if __name__ == "__main__":
    main()
