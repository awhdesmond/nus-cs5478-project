"""Smooth motions for the arm: quintic (minimum-jerk) joint moves and straight-line Cartesian moves.

A trajectory is anything with a `duration` and `sample(t) -> (q, qd, qdd)`: joint angles, velocities and
accelerations at time t. The tracking controller follows whichever it is given.
"""

import math

import numpy as np

from motion.kinematics import Kinematics, rotation_error


def quintic(t: float, T: float) -> tuple[float, float, float]:
    """Minimum-jerk timing: progress s from 0 to 1 over T seconds, and its 1st and 2nd time derivatives.

    s(τ) = 10τ³ - 15τ⁴ + 6τ⁵ with τ = t/T starts and ends with zero velocity and zero acceleration, so the
    arm never gets a sudden kick. Its peak speed is 15/8 = 1.875 times the average, at the midpoint.
    """
    tau = min(max(t / T, 0.0), 1.0)
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    sd = 30 * tau**2 * (1 - tau) ** 2 / T
    sdd = 60 * tau * (1 - tau) * (1 - 2 * tau) / T**2
    return s, sd, sdd


def move_duration(distance: float, max_speed: float, min_duration: float = 0.5) -> float:
    """Shortest quintic duration that keeps the peak speed (1.875 × average) at or under max_speed."""
    return max(min_duration, 1.875 * distance / max_speed)


class JointMove:
    """Straight line in joint space from q0 to q1, minimum-jerk timed. The tcp's path is curved."""

    def __init__(self, q0, q1, duration: float | None = None, max_joint_speed: float = 1.0):
        self.q0, self.q1 = np.asarray(q0, float), np.asarray(q1, float)
        delta = self.q1 - self.q0
        self.duration = duration or move_duration(np.abs(delta).max(), max_joint_speed)  # [rad/s] fastest joint

    def sample(self, t: float):
        s, sd, sdd = quintic(t, self.duration)
        delta = self.q1 - self.q0
        return self.q0 + s * delta, sd * delta, sdd * delta


class CartesianMove:
    """The tcp along a straight line, turning evenly to the new orientation, minimum-jerk timed.

    Solved with IK once, up front, at `dt` intervals: each solve starts from the previous one, so the arm
    stays on the same branch and each solve takes one or two iterations. Joint velocities and accelerations
    come from differencing those samples.
    """

    def __init__(self, kin: Kinematics, q0, p1, R1: np.ndarray, duration: float | None = None,
                 max_speed: float = 0.1, dt: float = 0.002):
        q0 = np.asarray(q0, float)
        p0, R0 = kin.fk(q0)
        p1 = np.asarray(p1, float)
        turn = rotation_error(R1, R0)  # axis × angle from the start orientation to the end one
        # Rotation counts as distance too: 1 rad of turn is treated like 0.1 m of travel.
        self.duration = duration or move_duration(max(np.linalg.norm(p1 - p0), 0.1 * np.linalg.norm(turn)), max_speed)
        self.dt = dt
        self.times = np.arange(0.0, self.duration + dt / 2, dt)
        qs, q = [], q0
        for t in self.times:
            s, _, _ = quintic(t, self.duration)
            R = rotate(R0, s * turn)  # a fraction s of the way round, about the same fixed axis
            result = kin.ik(p0 + s * (p1 - p0), R, q)
            if not result.converged:
                raise ValueError(f"IK failed {t:.3f} s into the move toward {p1}: {result}")
            q = result.q
            qs.append(q)
        self.qs = np.array(qs)
        self.qds = np.gradient(self.qs, dt, axis=0)
        self.qdds = np.gradient(self.qds, dt, axis=0)

    def sample(self, t: float):
        # Linear interpolation between the stored samples (2 ms apart: much finer than the motion).
        x = min(max(t, 0.0), self.duration) / self.dt
        i = min(int(x), len(self.times) - 2)
        f = x - i
        lerp = lambda a: (1 - f) * a[i] + f * a[i + 1]
        return lerp(self.qs), lerp(self.qds), lerp(self.qdds)


def rotate(R: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """R turned by the world-frame rotation `rotvec` (axis × angle)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return R.copy()
    k = rotvec / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    turn = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K  # Rodrigues' formula
    return turn @ R
