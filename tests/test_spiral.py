import numpy as np
import pytest

from insertion.config import SceneConfig, SpiralSearchConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.control.compliant import ImpedanceInsert, SpiralSearchInsert
from insertion.harness import run_trial


def trial(seed, uncertainty, clearance=0.001, friction=0.3, budget=40.0):
    return TrialConfig(
        seed=seed,
        scene=SceneConfig(clearance=clearance, friction=friction),
        uncertainty=UncertaintyConfig(translation_max=uncertainty),
        start=StartConfig(joint_noise=0.25),
        task=TaskConfig(cycle_time_budget=budget),
    )


def test_spiral_finds_hole_beyond_chamfer_capture_where_plain_impedance_fails():
    cfg = trial(seed=1, uncertainty=0.005)  # ~4.5 mm offset: outside the chamfer's reach
    plain, _ = run_trial(cfg, ImpedanceInsert())
    spiral, _ = run_trial(cfg, SpiralSearchInsert())
    assert not plain.seated
    assert spiral.success
    assert spiral.controller_info["searched"] and spiral.controller_info["found"]
    assert spiral.peak_force_10ms < 10.0


def test_no_search_when_peg_goes_straight_in():
    result, _ = run_trial(trial(seed=0, uncertainty=0.0), SpiralSearchInsert())
    assert result.success
    assert not result.controller_info["searched"]


def test_spiral_setpoint_pitch_and_centre():
    ctrl = SpiralSearchInsert(search=SpiralSearchConfig(pitch=0.002))
    ctrl.entrance = np.array([0.1, 0.2, 0.3])
    for turns in (1, 2, 3):
        ctrl.theta = 2 * np.pi * turns
        np.testing.assert_allclose(np.linalg.norm(ctrl._spiral_xy() - ctrl.entrance[:2]), 0.002 * turns)


@pytest.mark.parametrize("seed", [3, 5])
def test_spiral_succeeds_at_high_friction(seed):
    # μ=0.8 benchmark: search finds the hole and keeps sustained force well under the cap.
    result, _ = run_trial(trial(seed=seed, uncertainty=0.005, friction=0.8, budget=60.0), SpiralSearchInsert())
    assert result.success
    assert result.peak_force_10ms < 12.0
