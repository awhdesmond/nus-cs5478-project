import numpy as np
import pytest

from insertion.config import SceneConfig
from insertion.control.ik import JacobianIK, orientation_error
from insertion.scene import build_model
from insertion.sim import Sim


@pytest.fixture(scope="module")
def setup():
    model = build_model(SceneConfig())
    sim = Sim(model)
    sim.reset()
    return JacobianIK(model), sim.q()


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_orientation_error_is_zero_for_same_rotation():
    np.testing.assert_allclose(orientation_error(np.eye(3), np.eye(3)), 0.0)


def test_orientation_error_points_along_rotation_axis():
    err = orientation_error(np.eye(3), rot_z(0.01))
    np.testing.assert_allclose(err, [0, 0, 0.01], atol=1e-6)


def test_single_step_reduces_position_error(setup):
    ik, q0 = setup
    pos, _ = ik.fk(q0)
    target = pos + [0.01, -0.01, -0.02]
    dq, err0 = ik.jacobian_step(q0, target)
    _, err1 = ik.jacobian_step(q0 + dq, target)
    assert np.linalg.norm(err1) < 0.05 * np.linalg.norm(err0)


def test_solve_reaches_pose_and_holds_orientation(setup):
    ik, q0 = setup
    pos, rot = ik.fk(q0)
    target = pos + [-0.02, 0.01, -0.25]
    q, converged = ik.solve(q0, target, rot)
    assert converged
    pos_q, rot_q = ik.fk(q)
    np.testing.assert_allclose(pos_q, target, atol=1e-6)
    np.testing.assert_allclose(rot_q, rot, atol=1e-6)


def test_solve_does_not_touch_simulation_state():
    model = build_model(SceneConfig())
    sim = Sim(model)
    sim.reset()
    q_before = sim.q()
    ik = JacobianIK(model)
    pos, rot = ik.fk(q_before)
    ik.solve(q_before, pos - [0, 0, 0.1], rot)
    np.testing.assert_array_equal(sim.q(), q_before)


def test_solution_respects_joint_limits(setup):
    ik, q0 = setup
    pos, rot = ik.fk(q0)
    q, _ = ik.solve(q0, pos + [0.0, 0.0, 2.0], rot, iters=30)  # unreachable: pulls joints hard
    assert np.all(q >= ik.lower - 1e-12) and np.all(q <= ik.upper + 1e-12)


def test_damped_step_stays_bounded_at_singularity(setup):
    ik, _ = setup
    q_straight = np.zeros(6)  # arm fully stretched: elbow singularity
    pos, rot = ik.fk(q_straight)
    dq, _ = ik.jacobian_step(q_straight, pos + [0.0, 0.0, 0.05], rot)
    assert np.all(np.isfinite(dq)) and np.abs(dq).max() < 10.0


def test_restarts_recover_from_a_bad_initial_guess(setup):
    ik, q0 = setup
    pos, rot = ik.fk(q0)
    target = pos + [0.0, 0.0, -0.25]
    q_bad = q0.copy()
    q_bad[2] = 0.0  # start at the elbow singularity
    q, converged = ik.solve_with_restarts(q_bad, target, rot, rng=np.random.default_rng(0), iters=15)
    assert converged
    np.testing.assert_allclose(ik.fk(q)[0], target, atol=1e-5)
