import numpy as np
import pytest

from insertion.config import JointPDConfig, SceneConfig
from insertion.control.joint_pd import joint_pd_torque
from insertion.scene import UR5E_TORQUE_LIMITS, build_model
from insertion.sim import Sim

CFG = SceneConfig()


@pytest.fixture
def sim():
    s = Sim(build_model(CFG))
    s.reset()
    return s


def run_pd(sim, target, seconds, gravity_comp=True):
    max_err = 0.0
    for _ in range(int(seconds / sim.model.opt.timestep)):
        tau = joint_pd_torque(sim, target, JointPDConfig())
        if not gravity_comp:
            tau -= sim.gravity_comp()
        sim.step(tau)
        max_err = max(max_err, np.abs(sim.q() - target).max())
    return max_err


def test_actuators_are_torque_motors(sim):
    m = sim.model
    assert m.nu == 6
    np.testing.assert_allclose(m.actuator_gainprm[:, 0], 1.0)
    assert np.all(m.actuator_biastype == 0)
    np.testing.assert_allclose(m.actuator_ctrlrange[:, 1], UR5E_TORQUE_LIMITS)
    assert np.all(m.actuator_ctrllimited)
    np.testing.assert_allclose(m.key_ctrl, 0.0)


def test_peg_welded_to_flange_pointing_down_at_home(sim):
    flange, flange_rot = sim.site_pose("attachment_site")
    tip, _ = sim.site_pose("peg_tip")
    np.testing.assert_allclose(tip - flange, CFG.peg_length * flange_rot[:, 2], atol=1e-9)
    assert flange_rot[2, 2] == pytest.approx(-1.0, abs=1e-4)


def test_pd_with_gravity_comp_holds_home(sim):
    q_home = sim.q()
    assert run_pd(sim, q_home, seconds=3.0) < 1e-4


def test_arm_sags_without_gravity_comp(sim):
    q_home = sim.q()
    assert run_pd(sim, q_home, seconds=1.0, gravity_comp=False) > 1e-2


def test_pd_tracks_small_joint_step(sim):
    target = sim.q() + 0.05
    run_pd(sim, target, seconds=1.0)
    np.testing.assert_allclose(sim.q(), target, atol=1e-3)
