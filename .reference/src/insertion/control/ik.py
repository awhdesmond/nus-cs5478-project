"""Numerical inverse kinematics for the peg tip: damped least squares with joint limits.

Each iteration takes a damped least-squares step on the stacked position + orientation error

    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e,     e = [p* - p,  ½ Σᵢ rᵢ x r*ᵢ]

then clamps q to the joint limits. Damping keeps steps bounded near singularities (plain
J⁺ blows up as σ_min → 0) at the cost of slightly slower convergence; iterating still drives
e → 0 wherever the target is reachable. Solving runs on a private MjData, so it never
disturbs the simulation.
"""

import mujoco
import numpy as np

from insertion.scene import UR5E_JOINTS


def orientation_error(rot: np.ndarray, rot_des: np.ndarray) -> np.ndarray:
    """World-frame rotation vector (small-angle) taking `rot` toward `rot_des`."""
    return 0.5 * sum(np.cross(rot[:, i], rot_des[:, i]) for i in range(3))


class JacobianIK:
    def __init__(self, model: mujoco.MjModel, site: str = "peg_tip", damping: float = 0.01):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.damping = damping
        joint_ids = [model.joint(j).id for j in UR5E_JOINTS]
        self.qpos_adr = model.jnt_qposadr[joint_ids]
        self.dof_adr = model.jnt_dofadr[joint_ids]
        self.lower, self.upper = model.jnt_range[joint_ids].T
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site world position and rotation at joint config `q`."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x6 geometric Jacobian [linear; angular] of the site at `q`."""
        self.fk(q)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])

    def jacobian_step(
        self, q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Damped least-squares step at `q`. Returns (Δq, task-space error e)."""
        jac_full = self.jacobian(q)
        pos = self.data.site_xpos[self.site_id]
        err = target_pos - pos
        jac = jac_full[:3]
        if target_rot is not None:
            rot = self.data.site_xmat[self.site_id].reshape(3, 3)
            err = np.concatenate([err, orientation_error(rot, target_rot)])
            jac = jac_full
        damped = jac @ jac.T + self.damping**2 * np.eye(len(err))
        return jac.T @ np.linalg.solve(damped, err), err

    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        iters: int = 100,
        tol: float = 1e-6,
        max_step: float = 0.2,
    ) -> tuple[np.ndarray, bool]:
        """Iterate DLS steps from `q_init`, clamped to joint limits. Returns (q, converged).

        Steps are also scaled so no joint moves more than `max_step` rad per iteration.
        """
        q = np.clip(np.array(q_init, dtype=float), self.lower, self.upper)
        for _ in range(iters):
            dq, err = self.jacobian_step(q, target_pos, target_rot)
            if np.linalg.norm(err) < tol:
                return q, True
            q = np.clip(q + dq / max(1.0, np.abs(dq).max() / max_step), self.lower, self.upper)
        _, err = self.jacobian_step(q, target_pos, target_rot)
        return q, bool(np.linalg.norm(err) < tol)

    def solve_with_restarts(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        restarts: int = 8,
        spread: float = 0.5,
        **solve_kwargs,
    ) -> tuple[np.ndarray, bool]:
        """`solve` from `q_init`; if that fails, retry from random perturbations of it.

        Restarts stay near `q_init` (±`spread` rad) so a solution keeps the same arm
        configuration family (elbow up, wrist orientation) rather than flipping.
        """
        q, converged = self.solve(q_init, target_pos, target_rot, **solve_kwargs)
        rng = rng or np.random.default_rng(0)
        for _ in range(restarts):
            if converged:
                break
            q_try = q_init + rng.uniform(-spread, spread, size=len(q_init))
            q, converged = self.solve(q_try, target_pos, target_rot, **solve_kwargs)
        return q, converged
