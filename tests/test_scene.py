"""Step 1: the scene."""

import mujoco
import numpy as np
import pytest

from sim.config import SceneConfig
from sim.scene import build_model
from sim.ur5e import UR5E_JOINTS, UR5E_TORQUE_LIMITS


@pytest.fixture
def model():
    return build_model(SceneConfig())


def start_at_home(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    return data


def test_arm_has_six_torque_motors_at_the_real_limits(model):
    assert model.nu == 6
    for i, limit in enumerate(UR5E_TORQUE_LIMITS):
        np.testing.assert_allclose(model.actuator_ctrlrange[i], [-limit, limit])
        # A pure motor: gain 1 on ctrl, and no position/velocity terms (a servo would have them).
        assert model.actuator_gainprm[i, 0] == 1.0
        assert not model.actuator_biasprm[i].any()
    assert model.opt.timestep == 0.001


def test_arm_falls_without_torque(model):
    """Zero torque → nothing holds the arm up. Proves the servos are really gone."""
    data = start_at_home(model)
    home = data.qpos.copy()
    for _ in range(500):                  # 0.5 s
        mujoco.mj_step(model, data)
    assert np.abs(data.qpos - home).max() > 0.1   # sagged by more than ~6°


def test_gravity_compensation_plus_pd_holds_home(model):
    """τ = qfrc_bias + Kp·(q* − q) − Kd·q̇ holds the pose: the controller pattern used from now on."""
    data = start_at_home(model)
    q_home = data.qpos[:6].copy()         # only arm joints exist so far; later we look up by name
    kp = np.array([400, 400, 400, 100, 100, 50.0])   # stiffer on the big joints
    kd = np.array([40, 40, 40, 10, 10, 5.0])         # damping ≈ kp/10 stops oscillation
    for _ in range(1000):                 # 1 s
        q, qd = data.qpos[:6], data.qvel[:6]
        # qfrc_bias = torque needed to cancel gravity (+ Coriolis) at the current state.
        # MuJoCo recomputes it on every step; we just add it.
        data.ctrl[:] = data.qfrc_bias[:6] + kp * (q_home - q) - kd * qd
        mujoco.mj_step(model, data)
    assert np.abs(data.qpos[:6] - q_home).max() < 1e-3   # within 1 mrad: effectively still
