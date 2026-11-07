"""Joint-space PD with gravity compensation: τ = Kp·(q* - q) + Kd·(q̇* - q̇) + qfrc_bias.

Used to hold poses and track joint trajectories in free space; the impedance controller
takes over for the contact phase.
"""

import mujoco
import numpy as np

from insertion.config import JointPDConfig
from insertion.scene import arm_ctrlrange
from insertion.sim import Sim


def pd_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    bias: np.ndarray,
    cfg: JointPDConfig,
    ctrlrange: np.ndarray,
) -> np.ndarray:
    tau = np.asarray(cfg.kp) * (q_des - q) + np.asarray(cfg.kd) * (qd_des - qd) + bias
    return np.clip(tau, ctrlrange[:, 0], ctrlrange[:, 1])


def joint_pd_torque(
    sim: Sim,
    q_des: np.ndarray,
    cfg: JointPDConfig,
    qd_des: np.ndarray | None = None,
) -> np.ndarray:
    """PD hold/track using the simulator's own state (harness-side use)."""
    if qd_des is None:
        qd_des = np.zeros_like(q_des)
    return pd_torque(sim.q(), sim.qd(), q_des, qd_des, sim.gravity_comp(), cfg, arm_ctrlrange(sim.model))


def bias_torque(model: mujoco.MjModel, data: mujoco.MjData, qpos_adr, dof_adr, q, qd) -> np.ndarray:
    """Gravity + Coriolis joint torques at (q, q̇), computed on a scratch MjData.

    For controllers that only have observations: runs kinematics and recursive Newton-Euler
    with zero acceleration, skipping collision detection.
    """
    data.qpos[qpos_adr] = q
    data.qvel[dof_adr] = qd
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    result = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, result)
    return result[dof_adr]
