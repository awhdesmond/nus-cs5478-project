import numpy as np
import pytest

from insertion.control.trajectory import LineTrajectory, QuinticTrajectory, TrapezoidalProfile


def test_quintic_boundary_conditions():
    q0, q1 = np.zeros(6), np.array([1.0, -0.5, 0.2, 0.0, 0.0, 0.3])
    traj = QuinticTrajectory(q0, q1, 2.0)
    for t, q_expected in ((0.0, q0), (2.0, q1)):
        q, qd, qdd = traj.sample(t)
        np.testing.assert_allclose(q, q_expected)
        np.testing.assert_allclose(qd, 0.0)
        np.testing.assert_allclose(qdd, 0.0)


def test_quintic_duration_respects_velocity_and_acceleration_limits():
    q0, q1 = np.zeros(6), np.array([1.0, -0.5, 0.2, 0.0, 0.0, 0.3])
    traj = QuinticTrajectory.with_limits(q0, q1, vmax=1.0, amax=2.0)
    samples = [traj.sample(t) for t in np.linspace(0, traj.duration, 2001)]
    assert max(np.abs(s[1]).max() for s in samples) <= 1.0 + 1e-9
    assert max(np.abs(s[2]).max() for s in samples) <= 2.0 + 1e-9


def test_quintic_velocity_matches_numerical_derivative():
    traj = QuinticTrajectory(np.zeros(1), np.ones(1), 1.5)
    ts = np.linspace(0, 1.5, 3001)
    q = np.array([traj.sample(t)[0][0] for t in ts])
    qd = np.array([traj.sample(t)[1][0] for t in ts])
    np.testing.assert_allclose(np.gradient(q, ts)[1:-1], qd[1:-1], atol=1e-5)


@pytest.mark.parametrize("distance, cruises", [(0.3, True), (0.001, False)])
def test_trapezoid_reaches_distance_within_limits(distance, cruises):
    profile = TrapezoidalProfile(distance, vmax=0.05, amax=0.5)
    ts = np.linspace(0, profile.duration, 5001)
    samples = np.array([profile.sample(t) for t in ts])
    assert samples[-1, 0] == pytest.approx(distance)
    assert samples[:, 1].max() <= 0.05 + 1e-12
    assert (profile.t_cruise > 0) == cruises
    np.testing.assert_allclose(np.gradient(samples[:, 0], ts)[1:-1], samples[1:-1, 1], atol=1e-3)


def test_line_moves_along_segment():
    line = LineTrajectory([0, 0, 0.1], [0.01, 0, 0.0], vmax=0.05, amax=0.5)
    mid, vel = line.sample(line.duration / 2)
    direction = np.array([0.01, 0, -0.1]) / np.linalg.norm([0.01, 0, -0.1])
    np.testing.assert_allclose(np.cross(mid - line.start, direction), 0.0, atol=1e-12)
    np.testing.assert_allclose(vel / np.linalg.norm(vel), direction)
    np.testing.assert_allclose(line.sample(line.duration)[0], line.end)
