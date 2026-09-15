"""M3/M4 benchmark: perception-driven pick and compliant insertion of both connectors over random layouts.

    uv run python scripts/assembly_benchmark.py                               # default grid, 20 layouts per cell
    uv run python scripts/assembly_benchmark.py --layouts 5 --clearances 0.003 --frictions 0.3 --conditions ideal
    uv run python scripts/assembly_benchmark.py --fixed-sockets                # sockets at their default poses (M3/M4)

Conditions: `ideal` (perfect camera) and `realistic` (depth σ 1 mm, RGB noise 3 levels, extrinsic calibration
error up to ±2 mm per axis and 0.5°). Writes results/assembly_benchmark/connectors.csv (one row per connector per
run) and summary.csv.
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from insertion.assembly_scene import sample_layout
from insertion.assembly_skills import AssemblyRun
from insertion.camera import DepthNoise
from insertion.config import RESULTS_DIR, AssemblyConfig

CONDITIONS = {
    "ideal": DepthNoise(),
    "realistic": DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5),
}


def _job(args):
    seed, clearance, friction, condition, sockets = args
    cfg = replace(sample_layout(AssemblyConfig(), seed, sockets=sockets), clearance=clearance, friction=friction)
    result = AssemblyRun(cfg, seed=seed, noise=CONDITIONS[condition]).run()
    rows = []
    for log in result.connectors:
        row = {k: v for k, v in asdict(log).items() if k not in ("grasp", "search_info")}
        row.update({f"search.{k}": v for k, v in log.search_info.items()})
        row.update(seed=seed, clearance=clearance, friction=friction, condition=condition,
                   both_seated_at_end=result.both_seated_at_end, run_sim_time=result.sim_time,
                   grasp_margin_deg=log.grasp.friction_margin_deg if log.grasp else np.nan)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--clearances", type=float, nargs="+", default=[0.003, 0.001])
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.3, 0.8])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--fixed-sockets", action="store_true", help="don't randomise socket poses")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    jobs = [(seed, c, f, cond, not args.fixed_sockets) for cond, c, f in itertools.product(args.conditions, args.clearances, args.frictions) for seed in range(args.layouts)]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame([row for rows in pool.map(_job, jobs) for row in rows])
    out = RESULTS_DIR / "assembly_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "connectors.csv", index=False)

    keys = ["condition", "clearance", "friction", "kind"]
    summary = df.groupby(keys).agg(
        runs=("seed", "size"),
        identified=("identified", "mean"),
        grasped=("grasped", "mean"),
        lifted=("lifted", "mean"),
        other_disturbed_max_mm=("other_disturbed_mm", "max"),
        in_hand_err_p95_mm=("in_hand_tip_error_mm", lambda e: e.quantile(0.95)),
        grasp_face_err_max_deg=("grasp_face_error_deg", "max"),
        seated=("seated", "mean"),
        stayed_seated=("stayed_seated", "mean"),
        ft_belief=("inserted_belief", "mean"),
        seated_visual=("seated_visual", "mean"),
        verified=("verified", "mean"),
        visual_height_mm_mean=("visual_height_mm", "mean"),
        visual_offset_mm_max=("visual_offset_mm", "max"),
        wrong_socket=("wrong_socket", "sum"),
        searched=("search.searched", "mean"),
        insert_time_mean=("insert_time", "mean"),
        peak_force_10ms_max=("peak_force_10ms", "max"),
    ).reset_index()
    per_run = df.groupby(["condition", "clearance", "friction", "seed"]).both_seated_at_end.first().groupby(["condition", "clearance", "friction"]).mean()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nBoth connectors seated at end (per run):")
    print(per_run.to_string(float_format=lambda v: f"{v:.2f}"))
    # How well each completion check predicts the ground truth (stayed seated after release).
    checks = {}
    for check in ("inserted_belief", "seated_visual", "verified"):
        predicted, truth = df[check].astype(bool), df.stayed_seated.astype(bool)
        checks[check] = {"agree": (predicted == truth).mean(), "false_accept": int((predicted & ~truth).sum()),
                         "false_reject": int((~predicted & truth).sum())}
    print("\nCompletion checks vs ground truth (per connector):")
    print(pd.DataFrame(checks).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nFailures:", df[df.failure != ""].groupby(keys + ["failure"]).size().to_dict())
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
