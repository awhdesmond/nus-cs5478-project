"""Diagnose spiral search failures from the benchmark: where the peg went and what the controller did.

    uv run python scripts/diagnose_spiral.py            # all failing trials in results/benchmark_spiral
    uv run python scripts/diagnose_spiral.py --all      # every trial (slower)

Writes results/benchmark_spiral/failure_diagnosis.csv.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import pandas as pd

from insertion.config import RESULTS_DIR, SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.diagnose import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="diagnose every trial, not just failures")
    args = parser.parse_args()

    trials = pd.read_csv(RESULTS_DIR / "benchmark_spiral" / "trials.csv")
    selected = trials if args.all else trials[~trials.success]
    if selected.empty:
        print("No failing trials to diagnose.")
        return
    configs = [
        TrialConfig(
            seed=int(row.seed),
            scene=SceneConfig(clearance=row["cfg.scene.clearance"], friction=row["cfg.scene.friction"]),
            uncertainty=UncertaintyConfig(translation_max=row["cfg.uncertainty.translation_max"]),
            start=StartConfig(joint_noise=row["cfg.start.joint_noise"]),
            task=TaskConfig(cycle_time_budget=row["cfg.task.cycle_time_budget"]),
        )
        for _, row in selected.iterrows()
    ]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        df = pd.DataFrame([asdict(d) for d in pool.map(diagnose, configs)])
    out = RESULTS_DIR / "benchmark_spiral" / "failure_diagnosis.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
