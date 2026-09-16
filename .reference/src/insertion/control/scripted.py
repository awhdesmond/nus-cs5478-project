"""Scripted, non-compliant controllers.

`StraightDownInsert` is the week-1 easy-mode insert: quintic joint move to a pre-insertion
pose above the believed entrance, then a straight Cartesian descent tracked by joint PD.
No force feedback — it only works when clearance exceeds pose error.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.config import JointPDConfig, MotionConfig
from insertion.control.ik import JacobianIK
from insertion.control.motion import JointSegment, LineSegment, SegmentRunner
from insertion.control.trajectory import LineTrajectory, QuinticTrajectory
from insertion.harness import InsertionTarget, Observation


@dataclass
class StraightDownParams:
    settle_time: float = 0.3  # hold at pre-insertion before descending [s]
    descend_speed: float = 0.02  # [m/s]
    stop_above_bottom: float = 0.0005  # aim just above the hole bottom rather than pressing into it [m]


def pointing_down() -> np.ndarray:
    """Peg-tip frame with the peg axis (+z) along world −z (the home-pose orientation)."""
    return np.diag([1.0, -1.0, -1.0])


def plan_approach(ik: JacobianIK, q0: np.ndarray, pre: np.ndarray, rot: np.ndarray, motion: MotionConfig, seed: int):
    """Collision-avoiding approach from an arbitrary start to the pre-insertion pose.

    1. Lift: if the tip starts below the via height, go straight up (keeping start orientation).
    2. Joint move (quintic) to a via pose `via_height` above the pre-insertion pose.
    3. Straight Cartesian line down to the pre-insertion pose, peg pointing down.

    If the arm is already at the pre-insertion pose (within 2 mm and 2°), there is nothing to plan.

    IK is seeded from the home keyframe, not the start, so every trial lands in the same
    elbow-up / wrist-down configuration family (seeding from a perturbed start can converge
    to a flipped solution whose long joint sweep hits the socket). Returns (q_pre, segments).
    """
    tip0, rot0 = ik.fk(np.asarray(q0, dtype=float))
    rot_error = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rot0.T @ rot) - 1) / 2))))
    if np.linalg.norm(tip0 - pre) < 0.002 and rot_error < 2.0:
        return np.asarray(q0, dtype=float), []  # already at the pre-insertion pose (caller chose the arm configuration)
    rng = np.random.default_rng(seed)
    q_home = ik.model.key("home").qpos[ik.qpos_adr]
    via = pre + [0.0, 0.0, motion.via_height]
    q_via, ok_via = ik.solve_with_restarts(q_home, via, rot, rng=rng)
    q_pre, ok_pre = ik.solve_with_restarts(q_via, pre, rot, rng=rng)
    if not (ok_via and ok_pre):
        raise RuntimeError(f"IK failed to reach pre-insertion pose {pre}")

    segments = []
    q_lifted = np.asarray(q0, dtype=float)
    tip0, rot0 = ik.fk(q_lifted)
    if tip0[2] < via[2]:
        lift_to = np.array([tip0[0], tip0[1], via[2]])
        q_lifted, ok_lift = ik.solve(q_lifted, lift_to, rot0)
        if ok_lift:
            segments.append(LineSegment(ik, q0, LineTrajectory(tip0, lift_to, motion.lift_vmax, motion.cart_amax), rot0))
        else:
            q_lifted = np.asarray(q0, dtype=float)  # can't lift straight up from here; go direct
    segments.append(JointSegment(QuinticTrajectory.with_limits(q_lifted, q_via, motion.joint_vmax, motion.joint_amax)))
    segments.append(LineSegment(ik, q_via, LineTrajectory(via, pre, motion.lift_vmax, motion.cart_amax), rot))
    return q_pre, segments


class StraightDownInsert:
    def __init__(self, params: StraightDownParams | None = None, motion: MotionConfig | None = None, pd: JointPDConfig | None = None):
        self.params = params or StraightDownParams()
        self.motion = motion or MotionConfig()
        self.pd = pd or JointPDConfig()

    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget, seed: int = 0) -> None:
        p, m = self.params, self.motion
        ik = JacobianIK(model, "peg_tip")
        rot = pointing_down()
        pre = target.believed_entrance + [0.0, 0.0, m.approach_height]
        goal = target.believed_entrance - [0.0, 0.0, target.depth - p.stop_above_bottom]
        q_pre, approach = plan_approach(ik, np.asarray(q0), pre, rot, m, seed)
        settle = JointSegment(QuinticTrajectory(q_pre, q_pre, p.settle_time))
        descend = LineSegment(ik, q_pre, LineTrajectory(pre, goal, p.descend_speed, m.cart_amax), rot)
        self.runner = SegmentRunner(model, ik, self.pd)
        self.runner.start([*approach, settle, descend], now=0.0)

    def act(self, obs: Observation) -> np.ndarray:
        q_des, qd_des = self.runner.target(obs.time)
        return self.runner.torque(obs.q, obs.qd, q_des, qd_des)
