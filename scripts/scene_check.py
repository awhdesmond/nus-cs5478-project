"""Week-1 deliverable: the scene rests stably, and each peg fits only its own port. Exits 1 on any failure.

    uv run python scripts/scene_check.py                  # 10 random layouts at rest; keying at 1, 2, 3 mm
    uv run python scripts/scene_check.py --layouts 30 --clearances 0.001
"""

import argparse
import sys
from dataclasses import replace

from harness.scene_check import keying_check, rest_check
from sim.config import SceneConfig
from sim.layout import sample_layout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=10)
    parser.add_argument("--clearances", type=float, nargs="+", default=[0.001, 0.002, 0.003])
    args = parser.parse_args()
    ok = True

    print(f"Rest: {args.layouts} random layouts, 2 s each, arm holding the ready pose")
    print(f"{'seed':>4}  {'max drift':>10}  contacts other than peg–table")
    for seed in range(args.layouts):
        r = rest_check(sample_layout(SceneConfig(), seed))
        ok &= r.passed
        print(f"{seed:>4}  {r.max_drift * 1e6:>7.1f} µm  {r.unexpected_contacts or 'none'}" + ("" if r.passed else "  FAIL"))

    for clearance in args.clearances:
        print(f"\nKeying at {clearance * 1000:.0f} mm clearance: pressed in with up to 30 N")
        print(f"{'peg':>6} → {'port':<6} {'turned':>7}  {'depth':>8}  {'result':<22} {'penetration':>11}")
        for r in keying_check(replace(SceneConfig(), clearance=clearance)):
            ok &= r.passed
            if r.peg == r.port:
                result = "seats" if r.seated else "DOESN'T SEAT"
            elif r.entered:
                result = "enters (fits: expected)" if r.fits else "ENTERS"
            else:
                result = "blocked at the rim"
            print(f"{r.peg:>6} → {r.port:<6} {r.yaw_offset_deg:>6.2f}°  {r.depth * 1000:>5.2f} mm  {result:<22} "
                  f"{r.max_penetration * 1000:>8.3f} mm" + ("" if r.passed else "  FAIL"))

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
