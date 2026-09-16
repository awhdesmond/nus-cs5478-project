"""Compare controllers on the benchmark grid: success with 95% CIs, paired tests, efficiency.

    uv run python scripts/compare.py                        # spiral vs guarded (the plan's comparison)
    uv run python scripts/compare.py --a spiral --b impedance

Reads results/benchmark_<controller>/trials.csv. Trials are paired by (friction, clearance,
uncertainty, seed): the same seed means the same socket offset and start pose for both controllers,
so outcomes are compared with an exact McNemar test on discordant pairs.
Writes results/comparison_<a>_vs_<b>.csv.
"""

import argparse

import pandas as pd

from insertion.config import RESULTS_DIR
from insertion.stats import mcnemar_exact, wilson

CELL = ["cfg.scene.friction", "cfg.scene.clearance", "cfg.uncertainty.translation_max"]


def compare(a: str, b: str) -> pd.DataFrame:
    ta = pd.read_csv(RESULTS_DIR / f"benchmark_{a}" / "trials.csv")
    tb = pd.read_csv(RESULTS_DIR / f"benchmark_{b}" / "trials.csv")
    paired = ta.merge(tb, on=CELL + ["seed"], suffixes=(f"_{a}", f"_{b}"))
    rows = []
    for cell, g in list(paired.groupby(CELL)) + [(("all", "all", "all"), paired)]:
        n = len(g)
        sa, sb = int(g[f"success_{a}"].sum()), int(g[f"success_{b}"].sum())
        a_only = int((g[f"success_{a}"] & ~g[f"success_{b}"]).sum())
        b_only = int((~g[f"success_{a}"] & g[f"success_{b}"]).sum())
        both = g[g[f"success_{a}"] & g[f"success_{b}"]]
        lo_a, hi_a = wilson(sa, n)
        lo_b, hi_b = wilson(sb, n)
        rows.append({
            "friction": cell[0],
            "clearance_mm": cell[1] * 1e3 if cell[1] != "all" else "all",
            "uncertainty_mm": cell[2] * 1e3 if cell[2] != "all" else "all",
            "n": n,
            f"{a}_success": sa / n,
            f"{a}_ci95": f"[{lo_a:.2f}, {hi_a:.2f}]",
            f"{b}_success": sb / n,
            f"{b}_ci95": f"[{lo_b:.2f}, {hi_b:.2f}]",
            f"only_{a}": a_only,
            f"only_{b}": b_only,
            "mcnemar_p": mcnemar_exact(a_only, b_only),
            "both_succeeded": len(both),
            # Efficiency on trials both controllers solved, so the comparison isn't biased by difficulty.
            f"{a}_cycle_s": both[f"cycle_time_{a}"].mean() if len(both) else float("nan"),
            f"{b}_cycle_s": both[f"cycle_time_{b}"].mean() if len(both) else float("nan"),
            f"cycle_ratio_{a}/{b}": (both[f"cycle_time_{a}"] / both[f"cycle_time_{b}"]).median() if len(both) else float("nan"),
            f"{a}_peak10ms_mean": g[f"peak_force_10ms_{a}"].mean(),
            f"{b}_peak10ms_mean": g[f"peak_force_10ms_{b}"].mean(),
            f"{a}_raw_over_cap": int(g[f"raw_over_cap_{a}"].sum()),
            f"{b}_raw_over_cap": int(g[f"raw_over_cap_{b}"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="spiral")
    parser.add_argument("--b", default="guarded")
    args = parser.parse_args()
    df = compare(args.a, args.b)
    out = RESULTS_DIR / f"comparison_{args.a}_vs_{args.b}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
