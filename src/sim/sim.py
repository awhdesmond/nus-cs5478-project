"""The simulation: MjModel + MjData, stepping, the gripper and attach-on-grasp.

Attach-on-grasp belongs to the simulated world, not to any controller. While the gripper is commanded closed
and both finger pads touch the same peg, that peg is welded to the gripper where it is. Commanding the gripper
open releases it. A controller only ever says "close" or "open".
"""

import mujoco
import numpy as np

from sim.config import PEGS, SceneConfig
from sim.scene import FINGER_JOINTS
from sim.ur5e import UR5E_JOINTS


class Sim:
    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig):
        self.model = model
        self.data = mujoco.MjData(model)
        self.cfg = cfg

        # Look everything up by name once, so adding bodies to the scene can never shift an index.
        self.arm_qpos = np.array([model.joint(j).qposadr[0] for j in UR5E_JOINTS])
        self.arm_dof = np.array([model.joint(j).dofadr[0] for j in UR5E_JOINTS])
        self.arm_act = np.array([model.actuator(j.removesuffix("_joint")).id for j in UR5E_JOINTS])
        self.gripper_act = model.actuator("gripper").id
        self.finger_qpos = [model.joint(j).qposadr[0] for j in FINGER_JOINTS]
        self.finger_geoms = {model.geom(j).id for j in FINGER_JOINTS}
        self.peg_geoms = {k: model.geom(f"{k}_peg").id for k in PEGS}
        self.peg_bodies = {k: model.body(f"{k}_peg").id for k in PEGS}
        self.welds = {k: model.equality(f"attach_{k}").id for k in PEGS}
        self.gripper_body = model.body("gripper").id
        self.ft_force_adr = model.sensor("ft_force").adr[0]
        self.ft_torque_adr = model.sensor("ft_torque").adr[0]
        self.closing = False
        self.grip_target = 0.0

    @property
    def time(self) -> float:
        return self.data.time

    def reset(self, keyframe: str = "home") -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        for weld in self.welds.values():
            self.data.eq_active[weld] = 0
        self.closing, self.grip_target = False, 0.0
        mujoco.mj_forward(self.model, self.data)  # compute positions, sensors etc. for the new state

    def step(self, arm_torque: np.ndarray) -> None:
        """One physics step with these arm joint torques [N·m]."""
        # Speed limit: move the finger target toward open/closed by at most finger_speed·dt per step.
        current = self.data.ctrl[self.gripper_act]
        max_change = self.cfg.finger_speed * self.model.opt.timestep
        self.data.ctrl[self.gripper_act] = current + np.clip(self.grip_target - current, -max_change, max_change)
        self.data.ctrl[self.arm_act] = arm_torque
        mujoco.mj_step(self.model, self.data)
        if self.closing and self.attached() is None:
            kind = self._pinched_peg()
            if kind is not None:
                self.attach(kind)

    # --- proprioception and sensing ----------------------------------------------------------

    def q(self) -> np.ndarray:
        """Arm joint angles [rad]."""
        return self.data.qpos[self.arm_qpos].copy()

    def qd(self) -> np.ndarray:
        """Arm joint velocities [rad/s]."""
        return self.data.qvel[self.arm_dof].copy()

    def gravity_comp(self) -> np.ndarray:
        """Joint torques that cancel gravity and Coriolis forces at the current state [N·m]."""
        return self.data.qfrc_bias[self.arm_dof].copy()

    def ft_raw(self) -> np.ndarray:
        """Wrist [fx, fy, fz, tx, ty, tz] in the `ft_site` frame: raw, untared, unfiltered."""
        d = self.data.sensordata
        return np.concatenate([d[self.ft_force_adr:self.ft_force_adr + 3], d[self.ft_torque_adr:self.ft_torque_adr + 3]])

    def finger_closing(self) -> tuple[float, float]:
        """How far each finger has closed from fully open [m] (left, right)."""
        left, right = self.data.qpos[self.finger_qpos]
        return float(left), float(right)

    def finger_gap(self) -> float:
        """Distance between the finger pads [m]."""
        return self.cfg.finger_open_gap - sum(self.finger_closing())

    # --- gripper -----------------------------------------------------------------------------

    def command_gripper(self, close: bool) -> None:
        """Close (until a peg stops the fingers) or open the gripper. Opening releases any held peg."""
        self.closing = close
        self.grip_target = self.cfg.finger_open_gap / 2 if close else 0.0
        # Ramp from where the finger actually is. Otherwise, after closing on a peg (target = full travel,
        # finger stopped at the peg), opening would wait ~0.2 s for the target to ramp back down to the finger.
        self.data.ctrl[self.gripper_act] = self.finger_closing()[0]
        if not close:
            for weld in self.welds.values():
                self.data.eq_active[weld] = 0

    def attached(self) -> str | None:
        """Which peg is welded to the gripper, if any."""
        for kind, weld in self.welds.items():
            if self.data.eq_active[weld]:
                return kind
        return None

    def _pinched_peg(self) -> str | None:
        """The peg both finger pads are touching right now, if any."""
        touching = {k: set() for k in PEGS}
        for contact in self.data.contact[:self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            for kind, geom in self.peg_geoms.items():
                if geom in pair:
                    touching[kind] |= pair & self.finger_geoms  # which finger(s) touch this peg
        for kind, fingers in touching.items():
            if len(fingers) == 2:
                return kind
        return None

    def attach(self, kind: str) -> None:
        """Weld the peg to the gripper at their current relative pose, so nothing jumps.

        The weld's stored pose is the peg's pose expressed in the gripper's frame: rotate the world offset by the
        inverse of the gripper's orientation.
        """
        d, g, p = self.data, self.gripper_body, self.peg_bodies[kind]
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, d.xquat[g])  # inverse rotation of the gripper
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[p] - d.xpos[g], inverse)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inverse, d.xquat[p])
        weld = self.welds[kind]
        self.model.eq_data[weld, 0:3] = 0.0  # anchor at the peg's origin
        self.model.eq_data[weld, 3:6] = rel_pos
        self.model.eq_data[weld, 6:10] = rel_quat
        d.eq_active[weld] = 1
