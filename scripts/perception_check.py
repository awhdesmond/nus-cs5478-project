"""M2 perception check: identification and pose error vs ground truth over random layouts.

    uv run python scripts/perception_check.py                          # 40 layouts × depth noise σ {0, 1, 2} mm
                                                                       # (RGB noise 3 levels whenever depth noise is on)
    uv run python scripts/perception_check.py --seeds 100 --sigmas 0 0.001

Writes results/perception_check/rows.csv (one row per part per layout) and summary.csv.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

from insertion.camera import DepthNoise
from insertion.config import RESULTS_DIR
from insertion.perception_check import check_layout


RGB_NOISE = 3.0  # intensity levels, applied whenever depth noise is on


def _job(args):
    seed, sigma = args
    noise = DepthNoise(sigma=sigma, rgb_sigma=RGB_NOISE if sigma > 0 else 0.0)
    return [asdict(row) for row in check_layout(seed, noise)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["noise_sigma", "part", "kind"]).agg(
        layouts=("seed", "size"),
        identified=("detected", "mean"),
        wrong_count_layouts=("n_estimates", lambda n: int((n != 2).sum())),
        xy_p50_mm=("xy_error", lambda e: e.median() * 1e3),
        xy_p95_mm=("xy_error", lambda e: e.quantile(0.95) * 1e3),
        xy_max_mm=("xy_error", lambda e: e.max() * 1e3),
        z_p95_mm=("z_error", lambda e: e.abs().quantile(0.95) * 1e3),
        yaw_p95_deg=("yaw_error", lambda e: np.degrees(e.quantile(0.95))),
        yaw_max_deg=("yaw_error", lambda e: np.degrees(e.max())),
        squareness_min=("squareness", "min"),
        squareness_max=("squareness", "max"),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.001, 0.002], help="depth noise std [m]")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    jobs = [(seed, sigma) for sigma in args.sigmas for seed in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame([row for rows in pool.map(_job, jobs) for row in rows])
    out = RESULTS_DIR / "perception_check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
