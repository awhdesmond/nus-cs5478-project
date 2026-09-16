"""Time-parameterised trajectories.

- `QuinticTrajectory`: joint-space point-to-point with zero velocity and acceleration at both
  ends. Duration is the shortest that respects per-joint velocity and acceleration limits.
- `TrapezoidalProfile`: scalar path parameter with bounded speed and acceleration, used to
  move along a Cartesian straight line (triangular if the distance is too short to cruise).
"""

import math
from dataclasses import dataclass

import numpy as np

# Peak |ṡ| and |s̈| of the quintic blend s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ on τ ∈ [0, 1].
QUINTIC_PEAK_VEL = 1.875
QUINTIC_PEAK_ACC = 10 / math.sqrt(3)


@dataclass
class QuinticTrajectory:
    q0: np.ndarray
    q1: np.ndarray
    duration: float

    @classmethod
    def with_limits(cls, q0, q1, vmax: float, amax: float, min_duration: float = 0.1) -> "QuinticTrajectory":
        """Shortest duration keeping every joint within `vmax` [rad/s] and `amax` [rad/s²]."""
        dq = float(np.abs(np.asarray(q1) - np.asarray(q0)).max())
        duration = max(QUINTIC_PEAK_VEL * dq / vmax, math.sqrt(QUINTIC_PEAK_ACC * dq / amax), min_duration)
        return cls(np.asarray(q0, dtype=float), np.asarray(q1, dtype=float), duration)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, q̇, q̈) at time t, held at the endpoints outside [0, duration]."""
        tau = min(max(t / self.duration, 0.0), 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / self.duration**2
        if t <= 0.0 or t >= self.duration:
            ds = dds = 0.0
        delta = self.q1 - self.q0
        return self.q0 + s * delta, ds * delta, dds * delta


@dataclass
class TrapezoidalProfile:
    distance: float
    vmax: float
    amax: float

    def __post_init__(self):
        # Shorten the cruise speed if there's no room to reach vmax (triangular profile).
        self.v_peak = min(self.vmax, math.sqrt(self.distance * self.amax))
        self.t_acc = self.v_peak / self.amax if self.amax > 0 else 0.0
        d_acc = 0.5 * self.amax * self.t_acc**2
        self.t_cruise = (self.distance - 2 * d_acc) / self.v_peak if self.v_peak > 0 else 0.0
        self.duration = 2 * self.t_acc + self.t_cruise

    def sample(self, t: float) -> tuple[float, float]:
        """(distance travelled, speed) at time t."""
        if self.distance <= 0 or t <= 0:
            return 0.0, 0.0
        if t >= self.duration:
            return self.distance, 0.0
        if t < self.t_acc:
            return 0.5 * self.amax * t**2, self.amax * t
        d_acc = 0.5 * self.amax * self.t_acc**2
        if t < self.t_acc + self.t_cruise:
            return d_acc + self.v_peak * (t - self.t_acc), self.v_peak
        t_dec = self.duration - t
        return self.distance - 0.5 * self.amax * t_dec**2, self.amax * t_dec


@dataclass
class LineTrajectory:
    """Straight Cartesian line from `start` to `end` with a trapezoidal speed profile."""

    start: np.ndarray
    end: np.ndarray
    vmax: float
    amax: float

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.end = np.asarray(self.end, dtype=float)
        length = float(np.linalg.norm(self.end - self.start))
        self.direction = (self.end - self.start) / length if length > 0 else np.zeros(3)
        self.profile = TrapezoidalProfile(length, self.vmax, self.amax)
        self.duration = self.profile.duration

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) at time t."""
        d, v = self.profile.sample(t)
        return self.start + d * self.direction, v * self.direction
