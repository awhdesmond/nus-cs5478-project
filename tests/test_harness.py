"""Step 6: the full-task harness and the statistics behind the grids."""

from dataclasses import replace

import numpy as np
import pytest

from harness.task import STAGES, add_pose_error, run_task
from perception.estimate import SceneEstimate
from perception.pegs import PegEstimate
from perception.ports import PortEstimate
from sim.config import SceneConfig


# --- 6a: one full-task trial -----------------------------------------------------------------------

def test_pose_error_moves_every_object_by_exactly_that_much():
    peg = lambda: PegEstimate("round", np.array([0.1, 0.5, 0.22]), 0.0, 0.01, 0.8, np.zeros((4, 2)) + [0.1, 0.5])
    port = lambda: PortEstimate("round", np.array([-0.1, 0.5, 0.2]), 0.0, 0.015, 0.8)
    scene = SceneEstimate(0.17, {"round": peg(), "square": peg()}, {"round": port(), "square": port()})
    add_pose_error(scene, 0.002, np.random.default_rng(0))
    for est, before in ((scene.pegs["round"], peg()), (scene.ports["square"], port())):
        moved = (est.top if hasattr(est, "top") else est.entrance) - (before.top if hasattr(before, "top") else before.entrance)
        assert np.linalg.norm(moved) == pytest.approx(0.002) and moved[2] == 0     # sideways, 2 mm
    assert np.allclose(scene.pegs["round"].outline - [0.1, 0.5], (scene.pegs["round"].top - peg().top)[:2])  # outline too
    assert not np.allclose(scene.pegs["round"].top, scene.pegs["square"].top)     # independent directions


def test_full_task_places_both_pegs():
    row = run_task(replace(SceneConfig(), depth_noise=0.002), seed=4, controller="compliant")
    assert row["success"] and row["failed_stage"] is None
    for kind in ("round", "square"):
        assert row[f"{kind}_peg_xy"] < 0.0005 and row[f"{kind}_port_xy"] < 0.0005   # perception logged
        assert row[f"{kind}_insert_time"] < 60 and row[f"{kind}_sustained_force"] < 30
    assert row["task_time"] < 120


def test_failure_is_blamed_on_the_first_stage_that_went_wrong():
    # 2 mm per-object error: the stiff arm jams in an insertion (each peg and port is off in its own random
    # direction, so which peg jams depends on how the errors add up).
    row = run_task(replace(SceneConfig(), depth_noise=0.002), seed=4, controller="stiff", pose_error=0.002)
    assert not row["success"] and row["failed_stage"] == "insertion" and row["failed_stage"] in STAGES
    assert max(row.get(f"{k}_sustained_force", 0) for k in ("round", "square")) > 29   # stopped by the cap
    assert row["round_peg_xy"] == pytest.approx(0.002, abs=0.0005)   # the injected error shows up as perception error

# --- 6b: statistics and the grid tables -------------------------------------------------------------

import pandas as pd

from harness.grid import comparison_table, failure_table, success_table
from harness.stats import mcnemar_exact, wilson_interval


@pytest.mark.parametrize("k, n, low, high", [(20, 20, 0.839, 1.0), (0, 10, 0.0, 0.278), (5, 10, 0.237, 0.763),
                                             (18, 20, 0.699, 0.972)])
def test_wilson_interval_known_values(k, n, low, high):
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=0.001)


def test_wilson_interval_is_never_zero_width():
    low, high = wilson_interval(20, 20)
    assert high == 1.0 and low < 0.9               # 20/20 doesn't prove 100 %


@pytest.mark.parametrize("a_only, b_only, p", [(0, 0, 1.0), (6, 0, 2 / 64), (0, 6, 2 / 64), (5, 5, 1.0),
                                               (10, 2, 2 * (1 + 12 + 66) / 4096)])
def test_mcnemar_exact(a_only, b_only, p):
    assert mcnemar_exact(a_only, b_only) == pytest.approx(p)


def synthetic_results():
    """Hand-made grid rows: 6 seeds, 1 condition. Compliant solves all; stiff fails seeds 0-3 at insertion."""
    rows = []
    for seed in range(6):
        for controller, ok in (("compliant", True), ("stiff", seed >= 4)):
            rows.append(dict(seed=seed, controller=controller, clearance=0.001, friction=0.3, noise=0.0,
                             pose_error=0.002, success=ok, failed_stage=None if ok else "insertion",
                             task_time=40.0 + (1.0 if controller == "compliant" else 0.0),
                             round_sustained_force=5.0 if ok else 30.2, round_peak_force=8.0))
    return pd.DataFrame(rows)


def test_grid_tables_count_and_compare_correctly():
    results = synthetic_results()
    success = success_table(results)
    assert success.loc[(0.001, 0.3, 0.0, 0.002, "stiff"), "success"] == pytest.approx(2 / 6, abs=0.001)  # tables are rounded
    compare = comparison_table(results).iloc[0]
    assert (compare.both, compare.compliant_only, compare.stiff_only) == (2, 4, 0)
    assert compare.p_value == pytest.approx(2 / 16)      # 4 discordant pairs, all one way: 2 × ½⁴
    assert compare.time_diff_median == pytest.approx(1.0)  # compared only on the 2 seeds both solved
