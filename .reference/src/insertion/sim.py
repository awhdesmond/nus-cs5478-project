"""Thin wrapper around MjModel/MjData: stepping, state and sensor readout."""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.scene import UR5E_JOINTS


@dataclass
class PegContacts:
    """Ground-truth contact state of the peg (for logging and evaluation, not for controllers)."""

    force: np.ndarray  # total contact force on the peg, world frame [N]
    count: int
    max_penetration: float  # deepest contact overlap [m], 0 if none


class Sim:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)

        # Insertion-only scene: the welded peg. The assembly scene has free connectors instead (see assembly_sim).
        self.peg_geom_id = model.geom("peg").id if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg") >= 0 else -1
        self._ft_force_adr = model.sensor("ft_force").adr[0]
        self._ft_torque_adr = model.sensor("ft_torque").adr[0]
        self._wrench = np.zeros(6)

        # Look up arm joint addresses by name so extra bodies (e.g. test objects) can't shift them.
        self.qpos_adr = np.array([model.joint(j).qposadr[0] for j in UR5E_JOINTS])
        self.dof_adr = np.array([model.joint(j).dofadr[0] for j in UR5E_JOINTS])

    @property
    def time(self) -> float:
        return self.data.time

    def reset(self, keyframe: str = "home") -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray) -> None:
        """Arm joint torques for actuators 0–5 (any further actuators, e.g. a gripper, keep their ctrl)."""
        self.data.ctrl[: len(ctrl)] = ctrl
        mujoco.mj_step(self.model, self.data)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def qd(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def gravity_comp(self) -> np.ndarray:
        """Gravity + Coriolis/centrifugal joint torques at the current state."""
        return self.data.qfrc_bias[self.dof_adr].copy()

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and 3×3 rotation of a site."""
        site = self.data.site(name)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def ft_raw(self) -> np.ndarray:
        """Wrist F/T [fx, fy, fz, tx, ty, tz] in the `ft_site` frame, untared and unfiltered."""
        d = self.data.sensordata
        return np.concatenate([d[self._ft_force_adr : self._ft_force_adr + 3], d[self._ft_torque_adr : self._ft_torque_adr + 3]])

    def peg_contacts(self) -> PegContacts:
        """Sum contact forces acting on the peg over all its active contacts."""
        force = np.zeros(3)
        count = 0
        max_pen = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if self.peg_geom_id not in (con.geom1, con.geom2):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._wrench)
            # Contact frame rows are world axes; the normal points from geom1 to geom2, and the
            # returned force acts on geom2. Flip it when the peg is geom1.
            f_world = con.frame.reshape(3, 3).T @ self._wrench[:3]
            force += f_world if con.geom2 == self.peg_geom_id else -f_world
            count += 1
            max_pen = max(max_pen, -con.dist)
        return PegContacts(force=force, count=count, max_penetration=max_pen)
