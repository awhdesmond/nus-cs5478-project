"""Week 1 Cartesian motion: iterate Δq = J⁺·Δx toward a target site pose.

Grows into full damped-least-squares IK in week 2 (damping, joint limits, null-space).
Orientation hold is already included: a position-only step on a 6-DOF arm lets the peg
tilt during a straight-down move.
"""

import mujoco
import numpy as np

from insertion.scene import UR5E_JOINTS


def orientation_error(rot: np.ndarray, rot_des: np.ndarray) -> np.ndarray:
    """World-frame rotation vector (small-angle) taking `rot` toward `rot_des`."""
    return 0.5 * sum(np.cross(rot[:, i], rot_des[:, i]) for i in range(3))


class JacobianIK:
    """Kinematics on a private MjData, so solving never disturbs the simulation state."""

    def __init__(self, model: mujoco.MjModel, site: str = "peg_tip"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.qpos_adr = np.array([model.joint(j).qposadr[0] for j in UR5E_JOINTS])
        self.dof_adr = np.array([model.joint(j).dofadr[0] for j in UR5E_JOINTS])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site world position and rotation at joint config `q`."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def jacobian_step(
        self, q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Δq = J⁺·e at config `q`. Returns (Δq, task-space error e)."""
        pos, rot = self.fk(q)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        err = target_pos - pos
        jac = self._jacp[:, self.dof_adr]
        if target_rot is not None:
            err = np.concatenate([err, orientation_error(rot, target_rot)])
            jac = np.vstack([jac, self._jacr[:, self.dof_adr]])
        return np.linalg.pinv(jac) @ err, err

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        iters: int = 100,
        tol: float = 1e-6,
        max_step: float = 0.2,
    ) -> tuple[np.ndarray, bool]:
        """Iterate the Jacobian step from `q_init`. Returns (q, converged).

        Steps are clamped to `max_step` rad (max joint) so near-singular pinv can't jump.
        """
        q = np.array(q_init, dtype=float)
        for _ in range(iters):
            dq, err = self.jacobian_step(q, target_pos, target_rot)
            if np.linalg.norm(err) < tol:
                return q, True
            scale = max(1.0, np.abs(dq).max() / max_step)
            q = q + dq / scale
        _, err = self.jacobian_step(q, target_pos, target_rot)
        return q, bool(np.linalg.norm(err) < tol)
