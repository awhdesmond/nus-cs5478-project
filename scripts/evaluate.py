"""Evaluate a controller over a clearance × uncertainty grid through the harness.

    uv run python scripts/evaluate.py --controller guarded --clearances 0.003 0.001 --uncertainties 0.002 0.005 --trials 20
    uv run python scripts/evaluate.py --controller straight --clearances 0.001 --uncertainties 0.005 --trials 5 --record

Writes results/<name>/trials.csv (one row per trial) and summary.csv (one row per grid cell).
Seeds are shared across controllers, so the same seed means the same socket offset and start pose.
"""

import argparse
import itertools
import os

import pandas as pd

from insertion.config import RESULTS_DIR, SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.controllers import CONTROLLERS
from insertion.harness import run_trials


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["cfg.scene.friction", "cfg.scene.clearance", "cfg.uncertainty.translation_max"]
    groups = df.groupby(keys)
    summary = groups.agg(
        trials=("success", "size"),
        success_rate=("success", "mean"),
        seated_rate=("seated", "mean"),
        peak_force_mean=("peak_force", "mean"),
        peak_force_10ms_mean=("peak_force_10ms", "mean"),
        over_cap=("failure_reason", lambda r: int((r == "force_cap").sum())),
        raw_over_cap=("raw_over_cap", "sum"),
        timeouts=("failure_reason", lambda r: int((r == "timeout").sum())),
    )
    successes = df[df.success].groupby(keys)
    summary["cycle_time_mean"] = successes.cycle_time.mean()
    summary["cycle_time_std"] = successes.cycle_time.std()
    if "info.attempts" in df:
        summary["attempts_mean"] = groups["info.attempts"].mean()
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=CONTROLLERS, default="guarded")
    parser.add_argument("--clearances", type=float, nargs="+", default=[0.003, 0.001])
    parser.add_argument("--uncertainties", type=float, nargs="+", default=[0.002, 0.005])
    parser.add_argument("--frictions", type=float, nargs="+", default=[SceneConfig().friction])
    parser.add_argument("--trials", type=int, default=20, help="trials (seeds) per grid cell")
    parser.add_argument("--joint-noise", type=float, default=0.25, help="start-pose perturbation [rad]")
    parser.add_argument("--budget", type=float, default=60.0, help="cycle time budget [s sim time]")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--name", default=None, help="output folder under results/ (default: <controller>_grid)")
    parser.add_argument("--record", action="store_true", help="save per-step trajectories for replay")
    args = parser.parse_args()

    configs = [
        TrialConfig(
            seed=seed,
            scene=SceneConfig(clearance=clearance, friction=friction),
            uncertainty=UncertaintyConfig(translation_max=uncertainty),
            start=StartConfig(joint_noise=args.joint_noise),
            task=TaskConfig(cycle_time_budget=args.budget),
        )
        for friction, clearance, uncertainty in itertools.product(args.frictions, args.clearances, args.uncertainties)
        for seed in range(args.trials)
    ]
    out_dir = RESULTS_DIR / (args.name or f"{args.controller}_grid")
    df = run_trials(
        configs,
        CONTROLLERS[args.controller],
        out_dir / "trials.csv",
        record_dir=out_dir / "trajectories" if args.record else None,
        workers=args.workers,
    )
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n{len(df)} trials → {out_dir}")


if __name__ == "__main__":
    main()
