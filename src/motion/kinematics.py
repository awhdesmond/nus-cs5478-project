"""The arm's kinematics at the tool centre point (tcp): forward kinematics, Jacobian, and inverse kinematics.

The robot's own kinematic model is something a real controller knows (it ships with the arm), so motion/ may
use the MuJoCo model for this. It computes on its own MjData, never touching the simulation's state.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim.ur5e import UR5E_JOINTS

TCP = "tcp"


def tool_down(yaw: float) -> np.ndarray:
    """The tcp orientation with the tool pointing straight down, turned by `yaw` about world z.

    Columns are the tcp's axes in world coordinates: x = (cos, sin, 0), y = (sin, -cos, 0), z = (0, 0, -1).
    yaw 0 is the ready pose's orientation. The fingers close along the tcp's y axis.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def rotation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The rotation (axis x angle, world frame, [rad]) that turns R into R_target."""
    q_target, q, q_err = np.empty(4), np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(q_target, R_target.ravel())
    mujoco.mju_mat2Quat(q, R.ravel())
    mujoco.mju_negQuat(q, q)  # inverse of a unit quaternion
    mujoco.mju_mulQuat(q_err, q_target, q)  # target · current⁻¹: the remaining turn, in the world frame
    if q_err[0] < 0:
        q_err = -q_err  # same rotation; this sign takes the short way round (angle ≤ 180°)
    rotvec = np.empty(3)
    mujoco.mju_quat2Vel(rotvec, q_err, 1.0)  # quaternion -> axis x angle
    return rotvec


@dataclass
class IkResult:
    q: np.ndarray  # [rad] arm joint angles
    converged: bool
    position_error: float  # [m]
    rotation_error: float  # [rad]
    iterations: int


class Kinematics:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)  # private scratch state
        self.qpos = np.array([model.joint(j).qposadr[0] for j in UR5E_JOINTS])
        self.dof = np.array([model.joint(j).dofadr[0] for j in UR5E_JOINTS])
        self.limits = np.array([model.joint(j).range for j in UR5E_JOINTS])  # (6, 2) [rad]
        self.site = model.site(TCP).id

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.data)  # body and site poses from joint angles
        mujoco.mj_comPos(self.model, self.data)  # quantities mj_jacSite needs

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: tcp position [m] and orientation (3x3) for arm joint angles q."""
        self._set(q)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x6: rows (tcp linear velocity; angular velocity) in the world frame, per arm joint velocity."""
        self._set(q)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
        return np.vstack([jp[:, self.dof], jr[:, self.dof]])

    def ik(self, position, R_target: np.ndarray, q0: np.ndarray, *, tol_position: float = 1e-5,
           tol_rotation: float = 1e-4, damping: float = 0.05, max_step: float = 0.2,
           max_iterations: int = 200) -> IkResult:
        """Joint angles putting the tcp at `position` with orientation `R_target`, starting from q0.

        Damped least squares: each step solves J·Δq ≈ e (e = remaining position and rotation error) as
            Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e.
        The λ² term keeps steps small near singularities (where J loses rank and plain J⁻¹ blows up), at the
        cost of slower convergence there. Starting from q0 and taking small steps keeps the solution on the
        same branch as q0 (elbow up, wrist the same way), so a move doesn't flip the arm over.
        """
        q = np.array(q0, dtype=float)
        target = np.asarray(position, dtype=float)
        for i in range(max_iterations + 1):
            pos, R = self.fk(q)
            e = np.concatenate([target - pos, rotation_error(R_target, R)])
            ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if ep < tol_position and er < tol_rotation:
                return IkResult(q, True, ep, er, i)
            if i == max_iterations:
                break
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            dq *= min(1.0, max_step / np.abs(dq).max())  # at most max_step rad per joint per iteration
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])
        return IkResult(q, False, ep, er, max_iterations)
