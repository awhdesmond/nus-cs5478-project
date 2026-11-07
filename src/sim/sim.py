"""The simulation: MjModel + MjData, stepping, the gripper and attach-on-grasp.

Attach-on-grasp belongs to the simulated world, not to any controller. While the gripper is commanded closed
and both finger pads touch the same peg, that peg is welded to the gripper where it is. Commanding the gripper
open releases it. A controller only ever says "close" or "open".
"""

import mujoco
import numpy as np

from sim.config import PEGS, SceneConfig
from sim.scene import FINGER_JOINTS, port_hole_half
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

    # --- ground truth (for scoring and checks only: a controller must never call these) -----

    def peg_ends(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """World positions of a peg's insertion end and its top end."""
        return self.data.site(f"{kind}_tip").xpos.copy(), self.data.site(f"{kind}_top").xpos.copy()

    def end_in_port(self, kind: str, port: str, end: str = "tip", depth: float | None = None) -> bool:
        """Is this end of the peg inside `port`'s hole: within its footprint, at most `depth` above its floor?

        `end` is "tip" (the insertion end) or "top". depth None = anywhere below the port's entrance.
        """
        point = self.data.site(f"{kind}_{end}").xpos
        bottom = self.data.site(f"{port}_port_bottom")
        # Express the point in the port's own frame: then the hole is centred at the origin, axis along z.
        local = bottom.xmat.reshape(3, 3).T @ (point - bottom.xpos)
        h = port_hole_half(self.cfg, port)
        inside = np.hypot(local[0], local[1]) < h if port == "round" else max(abs(local[0]), abs(local[1])) < h
        return bool(inside and local[2] <= (self.cfg.port_depth if depth is None else depth))

    def seated(self, kind: str, tol: float = 0.002) -> bool:
        """Success: the insertion end is within `tol` of the floor of the peg's *own* port."""
        return self.end_in_port(kind, kind, depth=tol)

    def in_wrong_port(self, kind: str) -> str | None:
        """The other port, if either end of this peg has gone into it."""
        for port in PEGS:
            if port != kind and (self.end_in_port(kind, port) or self.end_in_port(kind, port, end="top")):
                return port
        return None

    def peg_contact_force(self, kind: str) -> np.ndarray:
        """Total contact force on a peg from everything except the fingers, world frame [N].

        This is the force the proposal caps at 30 N: what the peg feels from the port and the table.
        """
        geom = self.peg_geoms[kind]
        total, wrench = np.zeros(3), np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if geom not in pair or pair & self.finger_geoms:
                continue
            # mj_contactForce gives the force in the contact's own frame (rows = its axes in world), acting on
            # geom2. Rotate it to world, and flip it if our peg is geom1.
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            f = contact.frame.reshape(3, 3).T @ wrench[:3]
            total += f if contact.geom2 == geom else -f
        return total

    def max_penetration(self) -> float:
        """How deep the deepest contact overlaps right now [m] (MuJoCo contacts are soft, so slightly > 0)."""
        return max((-c.dist for c in self.data.contact[:self.data.ncon]), default=0.0)
