"""Scripted, non-compliant controllers.

`StraightDownInsert` is the week-1 easy-mode insert: joint-space move to a pre-insertion
pose above the believed entrance, then a straight Cartesian descent tracked by joint PD.
No force feedback — it only works when clearance exceeds pose error.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.config import JointPDConfig
from insertion.control.jacobian_step import JacobianIK
from insertion.control.joint_pd import bias_torque, pd_torque
from insertion.harness import InsertionTarget, Observation


@dataclass
class StraightDownParams:
    approach_height: float = 0.05  # pre-insertion tip height above the entrance [m]
    move_time: float = 2.0  # joint-space move to pre-insertion [s]
    settle_time: float = 0.3  # hold before descending [s]
    descend_speed: float = 0.02  # [m/s]
    stop_above_bottom: float = 0.0005  # aim just above the hole bottom rather than pressing into it [m]


class StraightDownInsert:
    def __init__(self, params: StraightDownParams | None = None, pd: JointPDConfig | None = None):
        self.params = params or StraightDownParams()
        self.pd = pd or JointPDConfig()

    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget) -> None:
        p = self.params
        self.model = model
        self.ik = JacobianIK(model, "peg_tip")
        self.q0 = np.array(q0)
        # Peg pointing straight down: keep the tip orientation of the start (home) pose.
        _, self.rot_down = self.ik.fk(self.q0)
        self.pre = target.believed_entrance + [0.0, 0.0, p.approach_height]
        self.goal = target.believed_entrance - [0.0, 0.0, target.depth - p.stop_above_bottom]
        self.q_pre, converged = self.ik.solve(self.q0, self.pre, self.rot_down)
        if not converged:
            raise RuntimeError(f"IK failed to reach pre-insertion pose {self.pre}")
        self.q_des = self.q_pre.copy()
        self.dt = model.opt.timestep

    def act(self, obs: Observation) -> np.ndarray:
        p = self.params
        t = obs.time
        if t < p.move_time:
            # Cubic smoothstep in joint space (zero velocity at both ends).
            s = t / p.move_time
            blend, blend_rate = 3 * s**2 - 2 * s**3, (6 * s - 6 * s**2) / p.move_time
            q_des = self.q0 + blend * (self.q_pre - self.q0)
            qd_des = blend_rate * (self.q_pre - self.q0)
        elif t < p.move_time + p.settle_time:
            q_des, qd_des = self.q_pre, np.zeros(6)
        else:
            travel = np.linalg.norm(self.pre - self.goal)
            dist = min(p.descend_speed * (t - p.move_time - p.settle_time), travel)
            waypoint = self.pre + (self.goal - self.pre) * (dist / travel)
            q_prev = self.q_des
            self.q_des, _ = self.ik.solve(q_prev, waypoint, self.rot_down, iters=3)
            q_des, qd_des = self.q_des, (self.q_des - q_prev) / self.dt
        bias = bias_torque(self.model, self.ik.data, self.ik.qpos_adr, self.ik.dof_adr, obs.q, obs.qd)
        return pd_torque(obs.q, obs.qd, q_des, qd_des, bias, self.pd, self.model.actuator_ctrlrange)
