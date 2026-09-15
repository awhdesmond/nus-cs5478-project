"""Motion segments tracked by joint PD + gravity compensation.

A segment maps segment-local time to a joint target (q*, q̇*):
- `JointSegment`: quintic joint-space move (free-space approach).
- `LineSegment`: straight Cartesian line with a trapezoidal speed profile at fixed orientation,
  converted to joint targets by warm-started DLS IK each tick and q̇* = J⁻¹·[v; 0].
"""

import mujoco
import numpy as np

from insertion.config import JointPDConfig
from insertion.control.ik import JacobianIK
from insertion.control.joint_pd import bias_torque, pd_torque
from insertion.control.trajectory import LineTrajectory, QuinticTrajectory
from insertion.scene import arm_ctrlrange


class JointSegment:
    def __init__(self, trajectory: QuinticTrajectory, guarded: bool = False):
        self.trajectory = trajectory
        self.duration = trajectory.duration
        self.guarded = guarded

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q, qd, _ = self.trajectory.sample(t)
        return q, qd


class LineSegment:
    def __init__(self, ik: JacobianIK, q_start: np.ndarray, line: LineTrajectory, rot: np.ndarray, guarded: bool = False):
        self.ik = ik
        self.line = line
        self.rot = rot
        self.duration = line.duration
        self.guarded = guarded
        self._q = np.array(q_start, dtype=float)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        pos, vel = self.line.sample(t)
        self._q, _ = self.ik.solve(self._q, pos, self.rot, iters=3)
        jac = self.ik.jacobian(self._q)
        damped = jac @ jac.T + self.ik.damping**2 * np.eye(6)
        qd = jac.T @ np.linalg.solve(damped, np.concatenate([vel, np.zeros(3)]))
        return self._q.copy(), qd


class SegmentRunner:
    """Plays a queue of segments; the owner appends segments and can interrupt the current one."""

    def __init__(self, model: mujoco.MjModel, ik: JacobianIK, pd: JointPDConfig):
        self.model = model
        self.ik = ik
        self.pd = pd
        self.queue: list = []
        self.current = None
        self.start_time = 0.0
        self.hold_q: np.ndarray | None = None

    def start(self, segments: list, now: float) -> None:
        """Replace everything (current + queued) with `segments`, starting now."""
        self.queue = list(segments)
        self.current = None
        self._advance(now)

    def _advance(self, now: float) -> None:
        if self.current is not None:
            self.hold_q, _ = self.current.sample(self.current.duration)
        self.current = self.queue.pop(0) if self.queue else None
        self.start_time = now

    @property
    def idle(self) -> bool:
        return self.current is None

    def in_guarded(self) -> bool:
        return self.current is not None and self.current.guarded

    def target(self, now: float) -> tuple[np.ndarray, np.ndarray]:
        while self.current is not None and now - self.start_time >= self.current.duration:
            self._advance(now)
        if self.current is None:
            return self.hold_q, np.zeros(6)
        return self.current.sample(now - self.start_time)

    def torque(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray) -> np.ndarray:
        bias = bias_torque(self.model, self.ik.data, self.ik.qpos_adr, self.ik.dof_adr, q, qd)
        return pd_torque(q, qd, q_des, qd_des, bias, self.pd, arm_ctrlrange(self.model))
