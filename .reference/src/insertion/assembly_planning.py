"""Controller-side planning for the assembly: obstacles from perception, arm clearance checks, and the arm
configuration for inserting a connector held sideways (M4b).

Everything here uses only what the controller knows: its own kinematic model, the known fixture/table design,
and perceived poses. Obstacles are boxes placed on the scene's planning-only mocap geoms (never rendered, never in
contact) in a private MjData, and clearances come from `mujoco.mj_geomDistance`.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.assembly_check import rot_z
from insertion.assembly_scene import N_PLANNING_OBSTACLES, connector_half_width, socket_apothem
from insertion.config import AssemblyConfig
from insertion.control.ik import JacobianIK
from insertion.control.trajectory import QuinticTrajectory
from insertion.perception import SceneEstimate

ARM_BODIES = ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
              "gripper", "finger_left", "finger_right")


@dataclass
class Box:
    name: str
    centre: np.ndarray  # world [m]
    half: np.ndarray  # half extents along the box's own axes [m]
    yaw: float = 0.0

    @property
    def top(self) -> float:
        return float(self.centre[2] + self.half[2])


def perceived_obstacles(scene: SceneEstimate, cfg: AssemblyConfig, carried: str | None) -> list[Box]:
    """Table top, both socket blocks and every connector except the one in the hand, from perception."""
    table_z = scene.table_height
    tx, ty = cfg.table_center
    hx, hy = cfg.table_half
    boxes = [Box("table", np.array([tx, ty, table_z - 0.01]), np.array([hx, hy, 0.01]))]
    for socket in scene.sockets:
        half = socket_apothem(cfg, socket.kind) + cfg.socket_wall
        top = float(socket.entrance[2])
        boxes.append(Box(f"{socket.kind}_socket", np.array([*socket.entrance[:2], (top + table_z) / 2]),
                         np.array([half, half, (top - table_z) / 2]), socket.yaw))
    for connector in scene.connectors:
        if connector.kind == carried:
            continue
        hw = connector_half_width(cfg, connector.kind)
        if connector.lying:
            boxes.append(Box(f"{connector.kind}_connector", np.array([*connector.centre[:2], table_z + hw]),
                             np.array([connector.length / 2, hw, hw]), connector.yaw))
        else:
            top = float(connector.top[2])
            boxes.append(Box(f"{connector.kind}_connector", np.array([*connector.top[:2], (top + table_z) / 2]),
                             np.array([hw * math.sqrt(2), hw * math.sqrt(2), (top - table_z) / 2])))
    return boxes


class ClearanceChecker:
    """Minimum distance between the arm (at a joint configuration) and a set of obstacle boxes."""

    def __init__(self, model: mujoco.MjModel, ik: JacobianIK):
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos_adr = ik.qpos_adr
        bodies = {model.body(name).id for name in ARM_BODIES}
        self.arm_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
        self.obstacle_geoms = [model.geom(f"plan_obstacle_{i}").id for i in range(N_PLANNING_OBSTACLES)]
        self.mocap_ids = [model.body_mocapid[model.body(f"plan_obstacle_{i}").id] for i in range(N_PLANNING_OBSTACLES)]
        self.names: list[str] = []

    def set_obstacles(self, boxes: list[Box]) -> None:
        if len(boxes) > N_PLANNING_OBSTACLES:
            raise ValueError(f"at most {N_PLANNING_OBSTACLES} planning obstacles")
        self.names = [box.name for box in boxes]
        for i, geom in enumerate(self.obstacle_geoms):
            mocap = self.mocap_ids[i]
            if i < len(boxes):
                box = boxes[i]
                self.model.geom_size[geom] = box.half
                self.model.geom_rbound[geom] = float(np.linalg.norm(box.half))
                self.data.mocap_pos[mocap] = box.centre
                self.data.mocap_quat[mocap] = [math.cos(box.yaw / 2), 0, 0, math.sin(box.yaw / 2)]
            else:
                self.data.mocap_pos[mocap] = [0, 0, -5.0]  # parked far away
        self.active = self.obstacle_geoms[: len(boxes)]

    def clearance(self, q: np.ndarray, distmax: float = 0.05) -> tuple[float, str]:
        """(minimum arm–obstacle distance capped at distmax [m], name of the closest obstacle)."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        best, name = distmax, ""
        for index, obstacle in enumerate(self.active):
            for geom in self.arm_geoms:
                distance = mujoco.mj_geomDistance(self.model, self.data, geom, obstacle, distmax, None)
                if distance < best:
                    best, name = distance, self.names[index]
        return best, name


@dataclass
class SidewaysPlan:
    yaw: float  # connector yaw when inserted; the fingers point along (cos yaw, sin yaw)
    q_via: np.ndarray  # tip `via_height` above the pre-insertion pose
    q_pre: np.ndarray  # tip at the pre-insertion pose (the insertion controller starts here)
    q_seated: np.ndarray  # tip at the hole bottom
    clearance: float  # [m] worst arm–obstacle distance over via, pre and seated
    closest: str


def seed_configurations(model: mujoco.MjModel, ik: JacobianIK, q_now: np.ndarray, rng: np.random.Generator, randoms: int = 6) -> list[np.ndarray]:
    """IK seeds covering the wrist configurations a sideways tool needs (IK from `home` alone often flips the wrist)."""
    home = model.key("home").qpos[ik.qpos_adr]
    ready = model.key("ready").qpos[ik.qpos_adr]
    seeds = [q_now, home, ready]
    for wrist_1 in (-math.pi / 2, math.pi / 2):
        for wrist_3 in (0.0, math.pi / 2, -math.pi / 2):
            seeds.append(home + [0, 0, 0, wrist_1, 0, wrist_3])
    seeds += [home + rng.uniform(-0.6, 0.6, size=6) for _ in range(randoms)]
    return seeds


def plan_sideways_insertion(
    ik_tip: JacobianIK,
    checker: ClearanceChecker,
    entrance: np.ndarray,
    depth: float,
    yaw_options: list[float],
    q_now: np.ndarray,
    approach_height: float,
    via_height: float,
    margin: float = 0.005,
    seed: int = 0,
) -> SidewaysPlan | None:
    """Best (largest clearance, then least joint travel) collision-free configuration over the allowed yaws.

    The tip-site IK target is the connector upright at each yaw; the hand orientation follows from the in-hand
    estimate held in the tool-tip site. Via, pre-insertion and seated poses must be reachable in one arm
    configuration (no joint jumps between them) with every arm–obstacle distance ≥ margin.
    """
    rng = np.random.default_rng(seed)
    seeds = seed_configurations(ik_tip.model, ik_tip, q_now, rng)
    pre = np.asarray(entrance) + [0, 0, approach_height]
    via = pre + [0, 0, via_height]
    seated = np.asarray(entrance) - [0, 0, depth]
    best = None
    for yaw in yaw_options:
        rot = rot_z(yaw)
        for q_seed in seeds:
            q_pre, ok = ik_tip.solve(q_seed, pre, rot, iters=200)
            if not ok:
                continue
            q_via, ok_via = ik_tip.solve(q_pre, via, rot)
            q_seated, ok_seated = ik_tip.solve(q_pre, seated, rot)
            if not (ok_via and ok_seated) or max(np.abs(q_via - q_pre).max(), np.abs(q_seated - q_pre).max()) > 0.5:
                continue
            clearance, closest = min(checker.clearance(q) for q in (q_via, q_pre, q_seated))
            if clearance < margin:
                continue
            travel = float(np.abs(q_via - q_now).max())
            key = (round(clearance, 3), -travel)  # mm-level clearance ties broken by joint travel
            if best is None or key > best[0]:
                best = (key, SidewaysPlan(yaw, q_via, q_pre, q_seated, clearance, closest))
    return None if best is None else best[1]


def joint_path_clear(checker: ClearanceChecker, ik_tip: JacobianIK, q_from: np.ndarray, q_to: np.ndarray, traj: QuinticTrajectory,
                     margin: float, tip_floor: float, samples: int = 25) -> bool:
    """Arm clearance ≥ margin and the held connector's tip above `tip_floor` along a joint trajectory."""
    for t in np.linspace(0.0, traj.duration, samples):
        q = traj.sample(t)[0]
        if checker.clearance(q)[0] < margin or ik_tip.fk(q)[0][2] < tip_floor:
            return False
    return True
