"""Step 3: IK and trajectories."""

import math

import numpy as np
import pytest

from motion.kinematics import Kinematics, rotation_error, tool_down
from sim.config import SceneConfig
from sim.scene import build_model


@pytest.fixture(scope="module")
def kin():
    return Kinematics(build_model(SceneConfig()))


# --- 3a: kinematics and IK -------------------------------------------------------------------------

def test_tool_down_is_a_rotation_pointing_down():
    for yaw in (0.0, 0.7, -2.0):
        R = tool_down(yaw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(R[:, 2], [0, 0, -1])


def test_rotation_error_is_the_turn_between_two_orientations():
    assert np.allclose(rotation_error(tool_down(0.3), tool_down(0.3)), 0)
    # tool_down(a) -> tool_down(b) is a turn by b - a about world z.
    np.testing.assert_allclose(rotation_error(tool_down(0.5), tool_down(0.2)), [0, 0, 0.3], atol=1e-12)


def test_fk_matches_the_ready_keyframe(kin):
    cfg = SceneConfig()
    pos, R = kin.fk(np.array(cfg.ready_arm))
    np.testing.assert_allclose(pos, cfg.ready_tcp, atol=1e-4)
    np.testing.assert_allclose(R, tool_down(0.0), atol=1e-3)


def test_jacobian_matches_finite_differences(kin):
    q = np.array(SceneConfig().ready_arm) + 0.1
    J, h = kin.jacobian(q), 1e-6
    pos, R = kin.fk(q)
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = h
        pos2, R2 = kin.fk(q + dq)
        np.testing.assert_allclose((pos2 - pos) / h, J[:3, j], atol=1e-4)                  # linear
        np.testing.assert_allclose(rotation_error(R2, R) / h, J[3:, j], atol=1e-4)         # angular


def test_ik_recovers_the_ready_pose_from_home(kin):
    cfg = SceneConfig()
    home = kin.model.key("home").qpos[kin.qpos]
    result = kin.ik(cfg.ready_tcp, tool_down(0.0), home)
    assert result.converged and result.iterations < 20, result
    # This is how the stored ready_arm (sim/config.py) was solved: same joint angles, to rounding.
    np.testing.assert_allclose(result.q, cfg.ready_arm, atol=1e-4)


def test_ik_reaches_poses_all_over_the_pick_and_port_areas(kin):
    """Tool down at random positions, heights and yaws, starting from ready each time."""
    cfg = SceneConfig()
    rng = np.random.default_rng(0)
    q_ready = np.array(cfg.ready_arm)
    for center, half in ((cfg.pick_area_center, cfg.pick_area_half), (cfg.port_area_center, cfg.port_area_half)):
        for _ in range(25):
            x, y = rng.uniform(np.array(center) - half, np.array(center) + half)
            z = cfg.table_height + rng.uniform(0.02, 0.20)        # 2-20 cm above the table
            yaw = rng.uniform(-math.pi / 2, math.pi / 2)          # the gripper is symmetric under 180°
            result = kin.ik([x, y, z], tool_down(yaw), q_ready)
            assert result.converged and result.iterations < 50, (x, y, z, yaw, result)
            pos, R = kin.fk(result.q)
            assert np.linalg.norm(pos - [x, y, z]) < 1e-5
            # Same branch as ready: the shoulder, elbow and wrist-1/2 joints stay within ~1 rad of ready.
            assert np.abs(result.q - q_ready)[:5].max() < 1.2, result.q - q_ready
