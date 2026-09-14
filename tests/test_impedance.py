import numpy as np
import pytest

from insertion.compliance_check import external_push, surface_press, tilted_press
from insertion.config import ImpedanceConfig, SceneConfig
from insertion.control.impedance import CartesianImpedance
from insertion.scene import build_model
from insertion.sim import Sim

CFG = ImpedanceConfig()


def test_task_inertia_is_symmetric_positive_definite():
    sim = Sim(build_model(SceneConfig()))
    sim.reset()
    lam = CartesianImpedance(sim.model, CFG).task_inertia(sim.q())
    np.testing.assert_allclose(lam, lam.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(lam) > 0)


def test_free_space_setpoint_step_settles_without_overshoot():
    sim = Sim(build_model(SceneConfig()))
    sim.reset()
    imp = CartesianImpedance(sim.model, CFG)
    imp.set_damping_at(sim.q())
    start, rot = imp.tip_pose(sim.q())
    target = start + [0.005, 0.0, -0.005]
    for _ in range(2500):
        sim.step(imp.torque(sim.q(), sim.qd(), target, rot)[0])
    tip, _ = sim.site_pose("peg_tip")
    np.testing.assert_allclose(tip, target, atol=1e-4)


@pytest.mark.parametrize("depth", [0.005, 0.010])
def test_surface_press_force_matches_stiffness_and_stays_normal(depth):
    result = surface_press("impedance", depth, speed=0.005, start_height=0.003)
    assert result.mean_force == pytest.approx(CFG.stiffness_pos[2] * depth, rel=0.03)
    assert result.lateral_ratio < 0.02
    assert result.max_tip_speed < 1e-4


def test_surface_press_force_saturates():
    result = surface_press("impedance", 0.050, speed=0.005, start_height=0.003)
    assert result.mean_force == pytest.approx(CFG.max_spring_force, rel=0.03)


def test_joint_pd_press_pushes_sideways_unlike_impedance():
    # High friction so the sideways push shows up as force; at μ=0.3 it saturates the friction cone and slides.
    result = surface_press("pd", 0.008, scene=SceneConfig(friction=0.8), speed=0.005, start_height=0.003)
    assert result.lateral_ratio > 0.3


def test_external_push_displacement_matches_lateral_stiffness_and_recovers():
    result = external_push("impedance", push=3.0)
    assert result.displacement == pytest.approx(result.expected_displacement, rel=0.05)
    assert result.residual_after_release < 5e-4


def test_low_rotational_stiffness_flattens_tilted_peg():
    assert tilted_press(CFG.stiffness_rot[0]).final_tilt_deg < 0.5


def test_high_rotational_stiffness_keeps_commanded_tilt():
    assert tilted_press(20.0).final_tilt_deg > 3.0
