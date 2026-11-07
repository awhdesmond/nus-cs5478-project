"""Cartesian impedance control at the joint-torque level.

    τ = Jᵀ F + τ_bias(q, q̇) - D_q q̇,     F = [sat(K_p e_p) + D_p (ṗ* - ṗ);  K_r e_r + D_r (ω* - ω)]

e_p = p* - p is the tip position error, e_r the orientation error (rotation vector toward R*),
and [ṗ; ω] = J q̇. The arm behaves like a spring-damper about the setpoint: in free space it
tracks it; in contact it yields and the contact force settles at ≈ K·(setpoint penetration),
bounded by the spring saturation. No F/T measurement is needed for the control law itself.

Damping is set per axis from the task-space inertia Λ = (J M⁻¹ Jᵀ)⁻¹ at the pose where the
impedance phase starts (Λ varies little over the few-cm insertion workspace).
"""

import mujoco
import numpy as np

from insertion.config import ImpedanceConfig
from insertion.control.ik import orientation_error
from insertion.scene import UR5E_JOINTS, arm_ctrlrange


class CartesianImpedance:
    def __init__(self, model: mujoco.MjModel, cfg: ImpedanceConfig, site: str = "peg_tip"):
        self.model = model
        self.cfg = cfg
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        joint_ids = [model.joint(j).id for j in UR5E_JOINTS]
        self.qpos_adr = model.jnt_qposadr[joint_ids]
        self.dof_adr = model.jnt_dofadr[joint_ids]
        self.stiffness = np.array([*cfg.stiffness_pos, *cfg.stiffness_rot])
        self.damping = np.zeros(6)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._bias = np.zeros(model.nv)
        self._mass = np.zeros((model.nv, model.nv))

    def _update(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Kinematics, Jacobian and bias torques at (q, q̇) on the private MjData. Returns J (6×6)."""
        d = self.data
        d.qpos[self.qpos_adr] = q
        d.qvel[self.dof_adr] = qd
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_comVel(self.model, d)
        mujoco.mj_rne(self.model, d, 0, self._bias)
        mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])

    def task_inertia(self, q: np.ndarray) -> np.ndarray:
        jac = self._update(q, np.zeros(6))
        mujoco.mj_crb(self.model, self.data)
        mujoco.mj_fullM(self.model, self.data, self._mass)
        mass = self._mass[np.ix_(self.dof_adr, self.dof_adr)]
        return np.linalg.inv(jac @ np.linalg.solve(mass, jac.T))

    def set_damping_at(self, q: np.ndarray) -> None:
        """Critical-style damping D_ii = 2ζ·sqrt(K_ii·Λ_ii) from the task-space inertia at `q`."""
        self._lam_diag = np.diag(self.task_inertia(q))
        self.damping = 2 * self.cfg.damping_ratio * np.sqrt(self.stiffness * self._lam_diag)

    def set_stiffness(self, stiffness: np.ndarray) -> None:
        """Change the 6-axis stiffness online; damping is rescaled from the stored task-space inertia."""
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.damping = 2 * self.cfg.damping_ratio * np.sqrt(self.stiffness * self._lam_diag)

    def tip_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._update(q, np.zeros(6))
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        pos_des: np.ndarray,
        rot_des: np.ndarray,
        vel_des: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Joint torques and the commanded task wrench F (for logging)."""
        jac = self._update(q, qd)
        pos = self.data.site_xpos[self.site_id]
        rot = self.data.site_xmat[self.site_id].reshape(3, 3)
        twist = jac @ qd
        twist_des = np.zeros(6) if vel_des is None else np.concatenate([vel_des, np.zeros(3)])

        spring = self.stiffness * np.concatenate([pos_des - pos, orientation_error(rot, rot_des)])
        magnitude = np.linalg.norm(spring[:3])
        if magnitude > self.cfg.max_spring_force:
            spring[:3] *= self.cfg.max_spring_force / magnitude
        wrench = spring + self.damping * (twist_des - twist)

        tau = jac.T @ wrench + self._bias[self.dof_adr] - self.cfg.joint_damping * qd
        lo, hi = arm_ctrlrange(self.model).T
        return np.clip(tau, lo, hi), wrench
