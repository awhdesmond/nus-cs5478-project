"""Cartesian impedance control at a point on the tool: the arm behaves like a spring and damper there.

    F = K (x_d - x) - D ẋ          (force at the control point, world frame; and the same for rotation)
    τ = Jᵀ F + bias(q, q̇)          (joint torques; bias = gravity, Coriolis, centrifugal)

Unlike the joint tracker, this doesn't fight contact: pushed sideways by a port's chamfer, the peg gives way
by F / K. Low sideways stiffness lets the chamfer centre the peg; higher stiffness down lets it press. The
control point is the held peg's tip (from the in-hand estimate), so the spring acts where contact happens
and the peg pivots about its tip, not about the tool.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from motion.control import TORQUE_LIMITS
from motion.kinematics import Kinematics, rotation_error
from sim.sim import Sim


@dataclass(frozen=True)
class ImpedanceConfig:
    stiffness: tuple[float, float, float] = (300.0, 300.0, 1000.0)  # [N/m] world x, y, z
    rotational_stiffness: float = 10.0  # [N·m/rad] about every axis
    damping_ratio: float = 1.0  # 1 = critically damped: settles without overshoot


class CartesianImpedance:
    def __init__(self, sim: Sim, kin: Kinematics, offset=(0.0, 0.0, 0.0), cfg: ImpedanceConfig = ImpedanceConfig()):
        """offset: the control point in the tcp frame [m] (e.g. a held peg's tip); 0 = the tcp itself."""
        self.sim, self.kin, self.cfg = sim, kin, cfg
        self.offset = np.asarray(offset, float)
        self.K = np.diag([*cfg.stiffness, *[cfg.rotational_stiffness] * 3])
        self._M = np.zeros((sim.model.nv, sim.model.nv))

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Control point position and tool orientation, from the joint encoders."""
        p, R = self.kin.fk(self.sim.q())
        return p + R @ self.offset, R

    def jacobian(self) -> np.ndarray:
        """6x6 Jacobian of the control point (linear) and the tool (angular).

        A point r away from the tcp moves with v_tcp + ω x r = v_tcp - [r]x ω.
        """
        J = self.kin.jacobian(self.sim.q())
        _, R = self.kin.fk(self.sim.q())
        r = R @ self.offset
        r_cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        J[:3] -= r_cross @ J[3:]
        return J

    def torque(self, x_d, R_d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint torques pulling the control point toward x_d (and the tool toward R_d).

        Returns (torques, the spring force on the control point [N, world]) so callers can see what they ask.
        """
        sim = self.sim
        J = self.jacobian()
        x, R = self.pose()
        error = np.concatenate([np.asarray(x_d) - x, rotation_error(R_d, R)])
        velocity = J @ sim.qd()

        # Damping per axis, 2ζ·√(k·m), with m the arm's effective mass along that axis at this posture: the
        # diagonal of Λ = (J M⁻¹ Jᵀ)⁻¹, the operational-space inertia.
        mujoco.mj_fullM(sim.model, sim.data, self._M)
        M = self._M[np.ix_(sim.arm_dof, sim.arm_dof)]
        Lambda = np.linalg.inv(J @ np.linalg.solve(M, J.T))
        D = 2 * self.cfg.damping_ratio * np.sqrt(np.diag(self.K) * np.clip(np.diag(Lambda), 1e-6, None))

        wrench = self.K @ error - D * velocity
        tau = J.T @ wrench + sim.gravity_comp()
        return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS), wrench[:3]
