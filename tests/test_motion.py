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

# --- 3b: trajectories and tracking -----------------------------------------------------------------

from motion.control import JointTracker, follow
from motion.trajectory import CartesianMove, JointMove, quintic
from sim.layout import sample_layout
from sim.sim import Sim


def test_quintic_starts_and_ends_at_rest():
    T = 2.0
    assert quintic(0, T) == (0, 0, 0) and quintic(T, T) == pytest.approx((1, 0, 0))
    assert quintic(T / 2, T)[1] == pytest.approx(1.875 / T)                     # peak speed, at the middle
    h = 1e-6                                                                   # derivatives are consistent
    for t in (0.3, 1.1, 1.7):
        assert (quintic(t + h, T)[0] - quintic(t, T)[0]) / h == pytest.approx(quintic(t, T)[1], rel=1e-4)
        assert (quintic(t + h, T)[1] - quintic(t, T)[1]) / h == pytest.approx(quintic(t, T)[2], rel=1e-4)


def test_joint_move_goes_from_start_to_end():
    q0, q1 = np.zeros(6), np.array([0.5, -0.2, 0.1, 0, 0, 1.0])
    move = JointMove(q0, q1)
    assert move.duration == pytest.approx(1.875 * 1.0 / 1.0)                  # 1 rad at ≤ 1 rad/s peak
    np.testing.assert_allclose(move.sample(0)[0], q0)
    q, qd, _ = move.sample(move.duration)
    np.testing.assert_allclose(q, q1)
    np.testing.assert_allclose(qd, 0, atol=1e-12)


def test_cartesian_move_keeps_the_tcp_on_a_straight_line(kin):
    q0 = np.array(SceneConfig().ready_arm)
    p0, _ = kin.fk(q0)
    p1 = p0 + [0.10, 0.15, -0.12]
    move = CartesianMove(kin, q0, p1, tool_down(0.8))                          # also turns 0.8 rad
    for t in np.linspace(0, move.duration, 37):
        p, _ = kin.fk(move.sample(t)[0])
        off_line = np.linalg.norm(np.cross(p - p0, p1 - p0)) / np.linalg.norm(p1 - p0)
        assert off_line < 1e-5                                                 # within 0.01 mm of the line
    p, R = kin.fk(move.sample(move.duration)[0])
    np.testing.assert_allclose(p, p1, atol=1e-5)
    assert np.linalg.norm(rotation_error(tool_down(0.8), R)) < 1e-4


def test_arm_reaches_random_poses_in_the_pick_and_port_areas():
    """In the simulation, under torque control: move above a random point, descend in a straight line, lift."""
    rng = np.random.default_rng(1)
    for seed in range(4):
        cfg = sample_layout(SceneConfig(), seed)
        sim = Sim(build_model(cfg), cfg)
        sim.reset("ready")
        kin, tracker = Kinematics(sim.model), JointTracker(sim)
        for center, half in ((cfg.pick_area_center, cfg.pick_area_half), (cfg.port_area_center, cfg.port_area_half)):
            x, y = rng.uniform(np.array(center) - half, np.array(center) + half)
            R = tool_down(rng.uniform(-math.pi / 2, math.pi / 2))
            above, low = [x, y, cfg.table_height + 0.15], [x, y, cfg.table_height + 0.07]  # 7 cm: above the pegs
            # Each move is planned from where the arm is when it starts, hence the lambdas.
            for plan in (lambda: JointMove(sim.q(), kin.ik(above, R, sim.q()).q),
                         lambda: CartesianMove(kin, sim.q(), low, R),
                         lambda: CartesianMove(kin, sim.q(), above, R)):
                errors = follow(sim, tracker, plan())
                assert np.abs(errors).max() < 0.002, seed                     # tracked within 2 mrad throughout
            tcp = sim.data.site("tcp")
            assert np.linalg.norm(tcp.xpos - above) < 0.0005, seed            # ended within 0.5 mm
            assert np.linalg.norm(rotation_error(R, tcp.xmat.reshape(3, 3))) < math.radians(0.1)
            # The arm touched nothing: the only contacts are the pegs resting on the table.
            pairs = {tuple(sorted((sim.model.geom(c.geom1).name, sim.model.geom(c.geom2).name)))
                     for c in sim.data.contact[:sim.data.ncon]}
            assert pairs <= {("round_peg", "table"), ("square_peg", "table")}, pairs
