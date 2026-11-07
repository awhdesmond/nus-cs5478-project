"""Step 1: the scene."""

import math
import mujoco
import pytest
import numpy as np

from sim.config import SceneConfig
from sim.scene import build_model
from sim.ur5e import UR5E_JOINTS, UR5E_TORQUE_LIMITS
from sim.scene import port_hole_half, port_top_height


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


def height_below(model, data, x, y):
    """Height of the first surface straight down from (x, y, 0.5 m): a laser rangefinder.

    geomgroup keeps only group-0 geoms (table, ports, floor), so the arm's visual meshes (group 2)
    can't block the ray when the arm happens to be overhead.
    """
    geomid = np.zeros(1, dtype=np.int32)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    distance = mujoco.mj_ray(model, data, np.array([x, y, 0.5]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
    return 0.5 - distance


@pytest.mark.parametrize("clearance", [0.001, 0.003])
def test_each_hole_is_open_to_its_floor_and_its_rim_is_flat(clearance):
    cfg = SceneConfig(clearance=clearance)
    model = build_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    top = port_top_height(cfg)
    floor = cfg.table_height + cfg.port_base
    for kind, pose in cfg.port_poses.items():
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)

        def world(u, v):
            """Port-frame (u, v) → world (x, y)."""
            return pose.x + c * u - s * v, pose.y + s * u + c * v

        h = port_hole_half(cfg, kind)
        # Inside the hole: the ray falls to the floor (centre, and 90% of the way to the wall).
        assert height_below(model, data, *world(0, 0)) == pytest.approx(floor, abs=1e-6)
        assert height_below(model, data, *world(0.9 * h, 0)) == pytest.approx(floor, abs=1e-6)
        assert height_below(model, data, *world(0, -0.9 * h)) == pytest.approx(floor, abs=1e-6)
        # On the rim, halfway between the chamfer and the block's outer edge, all the way round: flat top.
        # Square hole: the 8 points around it (sides and corners). Round hole: one point per polygon face.
        m = h + cfg.port_chamfer + (cfg.port_wall - cfg.port_chamfer) / 2
        if kind == "square":
            rim = [(u, v) for u in (-m, 0, m) for v in (-m, 0, m) if (u, v) != (0, 0)]
        else:
            phis = 2 * math.pi * np.arange(cfg.round_sides) / cfg.round_sides
            rim = [(m * math.cos(phi), m * math.sin(phi)) for phi in phis]
        for u, v in rim:
            assert height_below(model, data, *world(u, v)) == pytest.approx(top, abs=1e-6)


def test_ports_sit_at_their_configured_poses():
    cfg = SceneConfig()
    model = build_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for kind, pose in cfg.port_poses.items():
        np.testing.assert_allclose(data.site(f"{kind}_port_entrance").xpos, [pose.x, pose.y, port_top_height(cfg)],
                                   atol=1e-12)
        rot = data.site(f"{kind}_port_bottom").xmat.reshape(3, 3)
        assert math.atan2(rot[1, 0], rot[0, 0]) == pytest.approx(pose.yaw)
