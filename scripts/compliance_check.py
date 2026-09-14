"""Isolation checks for the impedance controller: surface press, external push, tilted press.

    uv run python scripts/compliance_check.py
"""

from dataclasses import asdict

import pandas as pd

from insertion.compliance_check import external_push, surface_press, tilted_press
from insertion.config import RESULTS_DIR


def show(title, rows):
    df = pd.DataFrame([asdict(r) for r in rows])
    print(f"\n{title}\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return df


def main():
    out = RESULTS_DIR / "compliance_check"
    out.mkdir(parents=True, exist_ok=True)
    press = [surface_press("impedance", d) for d in (0.005, 0.010, 0.030)] + [surface_press("pd", 0.008)]
    show("Surface press (1 cm/s touchdown)", press).to_csv(out / "surface_press.csv", index=False)
    speed = [surface_press("impedance", 0.005, speed=v, start_height=0.002) for v in (0.02, 0.01, 0.005, 0.0025, 0.001)]
    show("Impact spike vs touchdown speed (5 mm setpoint depth)", speed).assign(
        touchdown_speed=[0.02, 0.01, 0.005, 0.0025, 0.001]
    ).to_csv(out / "touchdown_speed.csv", index=False)
    show("External push at the tip (3 N along +x)", [external_push("impedance"), external_push("pd")]).to_csv(out / "external_push.csv", index=False)
    show("Tilted press (4° commanded)", [tilted_press(k) for k in (0.25, 0.5, 1.0, 2.0, 5.0, 20.0)]).to_csv(out / "tilted_press.csv", index=False)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
