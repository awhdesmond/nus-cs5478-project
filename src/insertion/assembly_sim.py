"""Simulation wrapper for the assembly scene: gripper, attach-on-grasp and ground-truth state.

Attach-on-grasp is part of the environment, not the controller: while the gripper is commanded
closed and both finger pads touch the same connector, that connector is welded to the gripper at
its current relative pose. Commanding the gripper open releases it. Controllers only command the
gripper; ground-truth poses here are for scoring and scripted checks.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.assembly_scene import CONNECTORS, FINGER_JOINTS
from insertion.config import AssemblyConfig
from insertion.sim import Sim


@dataclass
class ConnectorState:
    kind: str
    tip: np.ndarray  # connector bottom centre, world [m]
    axis: np.ndarray  # connector +z axis, world
    attached: bool


class AssemblySim(Sim):
    def __init__(self, model: mujoco.MjModel, cfg: AssemblyConfig):
        super().__init__(model)
        self.cfg = cfg
        self.finger_qpos_adr = [model.joint(j).qposadr[0] for j in FINGER_JOINTS]
        self.finger_geom_ids = [model.geom(j).id for j in FINGER_JOINTS]
        self.connector_geom_ids = {k: model.geom(f"{k}_connector").id for k in CONNECTORS}
        self.connector_body_ids = {k: model.body(f"{k}_connector").id for k in CONNECTORS}
        self.weld_ids = {k: model.equality(f"attach_{k}").id for k in CONNECTORS}
        self.gripper_body_id = model.body("gripper").id
        self.gripper_actuator = model.actuator("gripper").id
        self.travel = cfg.finger_open_gap / 2
        self.closing = False

    def reset(self, keyframe: str = "camera_clear") -> None:
        super().reset(keyframe)
        for weld in self.weld_ids.values():
            self.data.eq_active[weld] = 0
        self.closing = False

    # --- gripper -----------------------------------------------------------------------------

    def command_gripper(self, close: bool) -> None:
        """Close (to full travel; the connector stops the fingers) or open the gripper."""
        self.closing = close
        self.data.ctrl[self.gripper_actuator] = self.travel if close else 0.0
        if not close:
            for weld in self.weld_ids.values():
                self.data.eq_active[weld] = 0

    def finger_gap(self) -> float:
        """Distance between finger inner faces [m]."""
        closing = self.data.qpos[self.finger_qpos_adr].sum()
        return self.cfg.finger_open_gap - closing

    def attached(self) -> str | None:
        for kind, weld in self.weld_ids.items():
            if self.data.eq_active[weld]:
                return kind
        return None

    def step(self, ctrl: np.ndarray) -> None:
        super().step(ctrl)
        if self.closing and self.attached() is None:
            kind = self._pinched_connector()
            if kind is not None:
                self.attach(kind)

    def _pinched_connector(self) -> str | None:
        """Connector touched by both finger pads, if any."""
        touching = {k: set() for k in CONNECTORS}
        for contact in self.data.contact[: self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            for kind, geom in self.connector_geom_ids.items():
                if geom in pair:
                    touching[kind] |= pair & set(self.finger_geom_ids)
        for kind, fingers in touching.items():
            if len(fingers) == 2:
                return kind
        return None

    def attach(self, kind: str) -> None:
        """Weld the connector to the gripper at their current relative pose (connector pose in gripper frame)."""
        g, c = self.gripper_body_id, self.connector_body_ids[kind]
        d = self.data
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[g])
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[c] - d.xpos[g], inv)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv, d.xquat[c])
        weld = self.weld_ids[kind]
        self.model.eq_data[weld, 0:3] = 0.0  # anchor at body2's origin
        self.model.eq_data[weld, 3:6] = rel_pos
        self.model.eq_data[weld, 6:10] = rel_quat
        d.eq_active[weld] = 1

    # --- ground truth (scoring only) ------------------------------------------------------------

    def connector_state(self, kind: str) -> ConnectorState:
        site = self.data.site(f"{kind}_tip")
        rot = site.xmat.reshape(3, 3)
        return ConnectorState(kind=kind, tip=site.xpos.copy(), axis=rot[:, 2].copy(), attached=self.attached() == kind)

    def seated(self, kind: str, depth_tol: float = 0.002) -> bool:
        """Connector tip within depth_tol of its own socket's hole bottom and inside the hole footprint."""
        tip = self.data.site(f"{kind}_tip").xpos
        bottom = self.data.site(f"{kind}_socket_bottom").xpos
        half = (self.cfg.round_radius if kind == "round" else self.cfg.square_half) + self.cfg.clearance / 2
        return bool(np.linalg.norm(tip[:2] - bottom[:2]) < half and tip[2] - bottom[2] <= depth_tol)

    def connector_in_wrong_socket(self, kind: str) -> bool:
        other = "square" if kind == "round" else "round"
        tip = self.data.site(f"{kind}_tip").xpos
        entrance = self.data.site(f"{other}_socket_entrance").xpos
        return bool(np.linalg.norm(tip[:2] - entrance[:2]) < 0.01 and tip[2] < entrance[2])
