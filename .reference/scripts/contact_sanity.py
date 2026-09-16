"""Contact sanity check at the tightest target clearance; exits non-zero if any scenario fails.

    uv run python scripts/contact_sanity.py
"""

import sys
from dataclasses import asdict

import pandas as pd

from insertion.config import RESULTS_DIR
from insertion.contact_check import MAX_FORCE_JITTER, MAX_PENETRATION, MAX_TIP_SPEED, run_all


def main():
    results = run_all(clearance=0.001)
    df = pd.DataFrame([{**asdict(r), "passed": r.passed} for r in results])
    out = RESULTS_DIR / "contact_sanity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    view = df.assign(
        timestep_ms=df.timestep * 1e3,
        jitter_pct=df.force_jitter * 100,
        penetration_mm=df.max_penetration * 1e3,
        tip_speed_mm_s=df.max_tip_speed * 1e3,
    )[["timestep_ms", "scenario", "mean_force", "jitter_pct", "penetration_mm", "tip_speed_mm_s", "warnings", "passed"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(
        f"\nLimits: penetration < {MAX_PENETRATION * 1e3} mm, jitter < {MAX_FORCE_JITTER:.0%}, "
        f"tip speed < {MAX_TIP_SPEED * 1e3} mm/s, no warnings → {out}"
    )
    sys.exit(0 if df.passed.all() else 1)


if __name__ == "__main__":
    main()
