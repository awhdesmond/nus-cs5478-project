"""Guarded-move + random-retry baseline (no compliance).

After the collision-avoiding approach, repeat:
  1. Hover `hover_height` above a candidate xy (first candidate: the believed entrance).
  2. Guarded descent toward the nominal hole bottom; abort as soon as the wrist F/T force
     exceeds `contact_force`.
  3. If aborted (surface, chamfer or a jam), retract straight up to hover height, sample a new
     candidate uniformly in ±`search_radius` around the believed entrance, move there, retry.

If a descent completes without tripping the force threshold the peg is in the hole; the
harness detects seating. Joint PD makes the arm stiff laterally, so a candidate only goes in
when it lands within roughly the clearance of the true hole — the expected success rate is the
point of the baseline, not a bug.
"""

import mujoco
import numpy as np

from insertion.config import GuardedRetryConfig, JointPDConfig, MotionConfig
from insertion.control.ik import JacobianIK
from insertion.control.motion import LineSegment, SegmentRunner
from insertion.control.scripted import plan_approach, pointing_down
from insertion.control.trajectory import LineTrajectory
from insertion.harness import InsertionTarget, Observation


class GuardedRetry:
    def __init__(self, cfg: GuardedRetryConfig | None = None, motion: MotionConfig | None = None, pd: JointPDConfig | None = None):
        self.cfg = cfg or GuardedRetryConfig()
        self.motion = motion or MotionConfig()
        self.pd = pd or JointPDConfig()

    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget, seed: int = 0) -> None:
        c, m = self.cfg, self.motion
        self.rng = np.random.default_rng([seed, 2])
        self.ik = JacobianIK(model, "peg_tip")
        self.rot = pointing_down()
        self.entrance = np.asarray(target.believed_entrance, dtype=float)
        self.goal_z = self.entrance[2] - target.depth + c.stop_above_bottom
        self.attempts = 0
        self.contacts = 0

        pre = self.entrance + [0.0, 0.0, m.approach_height]
        hover = self.entrance + [0.0, 0.0, c.hover_height]
        q_pre, approach = plan_approach(self.ik, np.asarray(q0), pre, self.rot, m, seed)
        q_hover, _ = self.ik.solve(q_pre, hover, self.rot)
        to_hover = LineSegment(self.ik, q_pre, LineTrajectory(pre, hover, m.lift_vmax, m.cart_amax), self.rot)
        self.runner = SegmentRunner(model, self.ik, self.pd)
        self.runner.start([*approach, to_hover, self._descent(q_hover, hover)], now=0.0)

    def _descent(self, q_start: np.ndarray, start: np.ndarray) -> LineSegment:
        self.attempts += 1
        goal = np.array([start[0], start[1], self.goal_z])
        return LineSegment(self.ik, q_start, LineTrajectory(start, goal, self.cfg.descend_speed, self.motion.cart_amax), self.rot, guarded=True)

    def _retry(self, obs: Observation) -> None:
        """Blocked: retract from where the peg actually is, move to a new random candidate, descend."""
        c, m = self.cfg, self.motion
        self.contacts += 1
        tip, _ = self.ik.fk(obs.q)
        hover_z = self.entrance[2] + c.hover_height
        up = np.array([tip[0], tip[1], hover_z])
        candidate = self.entrance[:2] + self.rng.uniform(-c.search_radius, c.search_radius, size=2)
        over = np.array([*candidate, hover_z])
        q_up, _ = self.ik.solve(obs.q, up, self.rot)
        q_over, _ = self.ik.solve(q_up, over, self.rot)
        segments = [
            LineSegment(self.ik, obs.q, LineTrajectory(tip, up, m.cart_vmax, m.cart_amax), self.rot),
            LineSegment(self.ik, q_up, LineTrajectory(up, over, m.cart_vmax, m.cart_amax), self.rot),
        ]
        if self.attempts < c.max_attempts:
            segments.append(self._descent(q_over, over))
        self.runner.start(segments, now=obs.time)

    def act(self, obs: Observation) -> np.ndarray:
        if self.runner.in_guarded() and np.linalg.norm(obs.ft[:3]) > self.cfg.contact_force:
            self._retry(obs)
        q_des, qd_des = self.runner.target(obs.time)
        return self.runner.torque(obs.q, obs.qd, q_des, qd_des)

    def info(self) -> dict:
        return {"attempts": self.attempts, "contacts": self.contacts}
