"""Week 1 finish line: hardcoded straight-down insert, run through the evaluation harness.

Default is easy mode (10 mm clearance, zero uncertainty). Writes one CSV row per trial and
per-step trajectories to results/<name>/. Replay with scripts/replay.py.

    uv run python scripts/easy_insert.py
    uv run python scripts/easy_insert.py --clearance 0.001 --uncertainty 0.005 --seeds 5
"""

import argparse

from insertion.config import RESULTS_DIR, SceneConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.control.scripted import StraightDownInsert
from insertion.harness import run_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clearance", type=float, default=0.010, help="total clearance [m]")
    parser.add_argument("--uncertainty", type=float, default=0.0, help="max xy offset of believed socket pose [m]")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budget", type=float, default=15.0, help="cycle time budget [s sim time]")
    parser.add_argument("--name", default="easy_insert", help="output folder under results/")
    args = parser.parse_args()

    configs = [
        TrialConfig(
            seed=seed,
            scene=SceneConfig(clearance=args.clearance),
            uncertainty=UncertaintyConfig(translation_max=args.uncertainty),
            task=TaskConfig(cycle_time_budget=args.budget),
        )
        for seed in range(args.seeds)
    ]
    out_dir = RESULTS_DIR / args.name
    df = run_trials(configs, StraightDownInsert, out_dir / "trials.csv", record_dir=out_dir / "trajectories")

    cols = ["seed", "success", "seated", "reached_contact", "cycle_time", "peak_force", "max_penetration", "final_depth", "failure_reason"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSuccess {df.success.sum()}/{len(df)} → {out_dir}")


if __name__ == "__main__":
    main()
