"""Joint-space tracking: computed-torque control that makes the arm follow a trajectory.

    τ = M(q) · (q̈_ref + Kp (q_ref - q) + Kd (q̇_ref - q̇)) + bias(q, q̇)

M is the arm's mass matrix and `bias` the gravity, Coriolis and centrifugal torques. Multiplying the
correction by M makes every joint behave like the same unit-mass spring-damper, whatever the arm's posture,
so one pair of gains (ωn, ζ) suits all joints. Step 5 adds a Cartesian impedance controller beside this one.
"""

import mujoco
import numpy as np

from sim.sim import Sim
from sim.ur5e import UR5E_TORQUE_LIMITS

TORQUE_LIMITS = np.array(UR5E_TORQUE_LIMITS)  # [N·m] per joint


class JointTracker:
    def __init__(self, sim: Sim, natural_frequency: float = 20.0, damping_ratio: float = 1.0):
        self.sim = sim
        self.kp = natural_frequency**2  # [1/s²] stiffness per unit mass
        self.kd = 2 * damping_ratio * natural_frequency  # [1/s] critical damping at ζ = 1
        self._M = np.zeros((sim.model.nv, sim.model.nv))

    def mass_matrix(self) -> np.ndarray:
        """The arm's 6x6 joint-space mass matrix at the current state (MuJoCo computes it every step)."""
        mujoco.mj_fullM(self.sim.model, self.sim.data, self._M)  # MuJoCo stores it sparse; this expands it
        dof = self.sim.arm_dof
        return self._M[np.ix_(dof, dof)]

    def torque(self, q_ref, qd_ref, qdd_ref) -> np.ndarray:
        sim = self.sim
        accel = qdd_ref + self.kp * (q_ref - sim.q()) + self.kd * (qd_ref - sim.qd())
        tau = self.mass_matrix() @ accel + sim.gravity_comp()
        return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)  # what the motors can give


def follow(sim: Sim, tracker: JointTracker, trajectory, settle: float = 0.3) -> np.ndarray:
    """Run the simulation along `trajectory`, then hold its end for `settle` s. Returns the tracking error
    (q_ref - q) at every step, (steps, 6) [rad], for checking."""
    dt = sim.model.opt.timestep
    steps = round((trajectory.duration + settle) / dt)
    errors = np.empty((steps, 6))
    start = sim.time
    for i in range(steps):
        q_ref, qd_ref, qdd_ref = trajectory.sample(sim.time - start)
        errors[i] = q_ref - sim.q()
        sim.step(tracker.torque(q_ref, qd_ref, qdd_ref))
    return errors
