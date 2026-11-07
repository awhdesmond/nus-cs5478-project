"""Assembly benchmark (M3-M5): perception-driven pick and compliant insertion of both connectors over random layouts.

    uv run python scripts/assembly_benchmark.py                               # default grid, 20 layouts per cell
    uv run python scripts/assembly_benchmark.py --layouts 5 --clearances 0.003 --frictions 0.3 --conditions ideal
    uv run python scripts/assembly_benchmark.py --orientations upright lying mixed --out m4b
    uv run python scripts/assembly_benchmark.py --orientations upright mixed --object-errors 0 0.002 --out m5   # M5 grid
    uv run python scripts/assembly_benchmark.py --orientations upright mixed --conditions realistic \\
        --clearances 0.001 --frictions 0.3 --debris 0.004 --out m5_debris                                        # failure injection
    uv run python scripts/assembly_benchmark.py --fixed-sockets                # sockets at their default poses (M3/M4)

Axes: camera condition (`ideal`; `realistic`: depth σ 1 mm, RGB noise 3 levels, extrinsic calibration error up to
±2 mm per axis and 0.5°), clearance, friction, connector orientation, independent per-object perception error
(uniform ±e per horizontal axis for each connector and socket), and debris height on one socket's floor (round for even
seeds, square for odd; 0 = none).

Each finished run is printed and appended to results/<out>/connectors.csv (one row per connector), so progress and
partial results survive an interrupted job. summary.csv and the printed tables come at the end.
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
CELL = ["orientation", "condition", "clearance", "friction", "object_error", "debris"]


def _job(job: dict) -> list[dict]:
    debris_kind = "round" if job["seed"] % 2 == 0 else "square"
    cfg = replace(
        sample_layout(AssemblyConfig(), job["seed"], sockets=job["sockets"], orientation=job["orientation"]),
        clearance=job["clearance"], friction=job["friction"],
        socket_debris=(debris_kind, job["debris"]) if job["debris"] > 0 else None,
    )
    result = AssemblyRun(cfg, seed=job["seed"], noise=CONDITIONS[job["condition"]], object_error=job["object_error"]).run()
    rows = []
    for log in result.connectors:
        row = {k: v for k, v in asdict(log).items() if k not in ("grasp", "search_info")}
        row.update({f"search.{k}": v for k, v in log.search_info.items()})
        row.update({k: job[k] for k in CELL + ["seed"]})
        row.update(failure_stage=log.failure_stage, debris_here=job["debris"] > 0 and debris_kind == log.kind,
                   both_seated_at_end=result.both_seated_at_end, run_sim_time=result.sim_time,
                   grasp_margin_deg=log.grasp.friction_margin_deg if log.grasp else np.nan)
        rows.append(row)
    return rows


def completion_checks(df: pd.DataFrame) -> pd.DataFrame:
    """How well each completion check predicts the ground truth (stayed seated after release), per connector."""
    table = {}
    for check in ("inserted_belief", "seated_visual", "verified"):
        predicted, truth = df[check].astype(bool), df.stayed_seated.astype(bool)
        table[check] = {"agree": (predicted == truth).mean(), "false_accept": int((predicted & ~truth).sum()),
                        "false_reject": int((~predicted & truth).sum()), "connectors": len(df)}
    return pd.DataFrame(table).T


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--clearances", type=float, nargs="+", default=[0.003, 0.001])
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.3, 0.8])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--orientations", nargs="+", default=["upright"], choices=["upright", "lying", "mixed"])
    parser.add_argument("--object-errors", type=float, nargs="+", default=[0.0], help="per-object perception error [m]")
    parser.add_argument("--debris", type=float, nargs="+", default=[0.0], help="debris height on one socket's floor [m]")
    parser.add_argument("--fixed-sockets", action="store_true", help="don't randomise socket poses")
    parser.add_argument("--out", default="assembly_benchmark", help="results subdirectory")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    grid = itertools.product(args.orientations, args.conditions, args.clearances, args.frictions, args.object_errors, args.debris)
    jobs = [dict(zip(CELL, cell), seed=seed, sockets=not args.fixed_sockets) for cell in grid for seed in range(args.layouts)]
    out = RESULTS_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "connectors.csv"
    csv.unlink(missing_ok=True)
    print(f"{len(jobs)} runs, {args.workers} workers → {out}", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result_rows = future.result()
            except Exception as err:  # keep going; the run shows up as missing
                print(f"[{done}/{len(jobs)}] CRASH {job}: {err!r}", flush=True)
                continue
            rows += result_rows
            pd.DataFrame(result_rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
            status = "ok" if result_rows[0]["both_seated_at_end"] else "FAIL " + " ".join(
                f"{r['kind']}:{r['failure'] or r['failure_stage'] or 'not seated'}" for r in result_rows if not (r["stayed_seated"] and r["verified"]))
            cell = " ".join(f"{k}={job[k]}" for k in CELL)
            print(f"[{done}/{len(jobs)}] seed {job['seed']:3d} {cell}: {status}", flush=True)

    df = pd.DataFrame(rows)
    keys = CELL + ["kind", "lying"]
    summary = df.groupby(keys).agg(
        runs=("seed", "size"),
        identified_correctly=("identified_correctly", "mean"),
        grasped=("grasped", "mean"),
        lifted=("lifted", "mean"),
        seated=("seated", "mean"),
        stayed_seated=("stayed_seated", "mean"),
        ft_belief=("inserted_belief", "mean"),
        seated_visual=("seated_visual", "mean"),
        verified=("verified", "mean"),
        wrong_socket=("wrong_socket", "sum"),
        wrong_end=("wrong_end", "sum"),
        searched=("search.searched", "mean"),
        relative_err_p95_mm=("relative_xy_error_mm", lambda e: e.quantile(0.95)),
        in_hand_err_p95_mm=("in_hand_tip_error_mm", lambda e: e.quantile(0.95)),
        grasp_face_err_max_deg=("grasp_face_error_deg", "max"),
        other_disturbed_max_mm=("other_disturbed_mm", "max"),
        sideways_clearance_min_mm=("sideways_clearance_mm", "min"),
        insert_time_mean=("insert_time", "mean"),
        cycle_time_mean=("cycle_time", "mean"),
        cycle_time_max=("cycle_time", "max"),
        peak_force_10ms_max=("peak_force_10ms", "max"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    runs = df.groupby(CELL + ["seed"]).agg(both=("both_seated_at_end", "first"), time=("run_sim_time", "first"))
    per_cell = runs.groupby(CELL).agg(runs=("both", "size"), both_seated=("both", "mean"), run_time_mean=("time", "mean"))
    print("\nBoth connectors seated at end (per run):")
    print(per_cell.to_string(float_format=lambda v: f"{v:.3g}"))
    print("\nCompletion checks vs ground truth (per connector):")
    print(completion_checks(df).to_string(float_format=lambda v: f"{v:.3f}"))
    if (df.debris > 0).any():
        print("\nCompletion checks on connectors whose socket had debris:")
        print(completion_checks(df[df.debris_here]).to_string(float_format=lambda v: f"{v:.3f}"))
    stages = df[df.failure_stage != ""].groupby(CELL + ["kind", "failure_stage"]).size()
    print("\nFailures by stage:", stages.to_dict() if len(stages) else "none")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
