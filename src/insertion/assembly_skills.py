"""Perception-driven skills for the two-connector assembly (M3 pick/release, M4 insertion).

`Robot` is the controller's view of the world: proprioception, wrist F/T, gripper command and finger gap,
and overhead camera images. Skills decide only from that. `AssemblyRun` sequences the skills and records
ground-truth outcomes separately (via the underlying `AssemblySim`) for scoring.

The run starts at the `ready` pose (tool down above the table's near edge, outside the camera's view) with one
capture. Then per connector, round first: antipodal grasp from the perceived outline → approach, close, lift →
in-hand estimate → spiral-search insertion into the perceived matching socket (compliant, with the tool-tip site
set to the in-hand estimate) → release → retract to `ready` → capture. That capture both verifies the insertion
visually (connector top at the seated height above its socket, centred on it) and is the scene the next pick
uses.
"""

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from insertion.assembly_check import rot_z, tool_down
from insertion.assembly_scene import CONNECTORS, build_assembly_model
from insertion.assembly_sim import AssemblySim
from insertion.camera import DepthNoise, OverheadCamera
from insertion.config import AssemblyConfig, JointPDConfig, MotionConfig, SensorConfig
from insertion.control.compliant import SpiralSearchInsert
from insertion.control.ik import JacobianIK
from insertion.control.motion import JointSegment, LineSegment, SegmentRunner
from insertion.control.scripted import plan_approach
from insertion.control.trajectory import LineTrajectory, QuinticTrajectory
from insertion.grasping import GraspConfig, GraspProposal, propose_grasps
from insertion.harness import InsertionTarget, Observation
from insertion.perception import ConnectorEstimate, GeometricEstimator, SceneEstimate, SocketEstimate
from insertion.scene import UR5E_JOINTS
from insertion.sensors import FTSensor

DOWN = np.diag([1.0, -1.0, -1.0])  # upright connector axes expressed in the (pointing-down) tool frame


class Robot:
    """Controller-side interface over AssemblySim: no ground-truth object state is exposed here."""

    def __init__(self, sim: AssemblySim, noise: DepthNoise | None, rng: np.random.Generator):
        self.sim = sim
        self.model = sim.model
        self.camera = OverheadCamera(sim.model, sim.cfg.image_width, sim.cfg.image_height, noise=noise, rng=rng)
        self.noise, self.rng = noise, rng
        self.ft = FTSensor(sim, SensorConfig())
        self.tool_tip_site = sim.model.site("tool_tip").id
        self.tcp_site = sim.model.site("tcp").id
        self.gripper_body = sim.model.body("gripper").id
        self.on_step = []  # callbacks(sim) after every physics step (ground-truth logging lives outside skills)

    @property
    def time(self) -> float:
        return self.sim.time

    def q(self) -> np.ndarray:
        return self.sim.q()

    def qd(self) -> np.ndarray:
        return self.sim.qd()

    def step(self, tau: np.ndarray) -> None:
        self.sim.step(tau)
        self.ft.update()
        for callback in self.on_step:
            callback(self.sim)

    def capture(self):
        return self.camera.capture(self.sim.data, self.noise, self.rng)

    def command_gripper(self, close: bool) -> None:
        self.sim.command_gripper(close)

    def finger_gap(self) -> float:
        return self.sim.finger_gap()

    def set_tool_tip(self, pos_in_gripper: np.ndarray, rot_in_gripper: np.ndarray) -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_in_gripper.flatten())
        self.model.site_pos[self.tool_tip_site] = pos_in_gripper
        self.model.site_quat[self.tool_tip_site] = quat
        self.model.site_sameframe[self.tool_tip_site] = 0  # otherwise MuJoCo may skip the site's own offset
        mujoco.mj_kinematics(self.model, self.sim.data)

    def finger_closing(self) -> tuple[float, float]:
        """Closing travel of the left and right finger joints [m] (proprioception)."""
        left, right = self.sim.data.qpos[self.sim.finger_qpos_adr]
        return float(left), float(right)

    def tare(self, hold_q: np.ndarray, runner: SegmentRunner, steps: int = 50) -> None:
        samples = []
        for _ in range(steps):
            self.step(runner.torque(self.q(), self.qd(), hold_q, np.zeros(6)))
            samples.append(self.sim.ft_raw())
        self.ft.tare(samples)

    def observation(self, t0: float) -> Observation:
        """Insertion-controller observation at the tool tip; F/T expressed in the tool-tip frame."""
        tip_pos, tip_rot = self.sim.site_pose("tool_tip")
        _, ft_rot = self.sim.site_pose("ft_site")
        to_tip = tip_rot.T @ ft_rot
        wrench = np.concatenate([to_tip @ self.ft.wrench[:3], to_tip @ self.ft.wrench[3:]])
        return Observation(self.time - t0, self.q(), self.qd(), wrench, tip_pos, tip_rot)


@dataclass
class InHandEstimate:
    tip_in_tool: np.ndarray  # connector tip in the tcp (tool) frame [m]
    rot_in_tool: np.ndarray  # connector axes in the tool frame


@dataclass
class ConnectorLog:
    kind: str
    identified: bool = False
    proposals: int = 0
    grasp: GraspProposal | None = None
    grasp_face_error_deg: float = float("nan")  # ground truth, square only: closing direction vs nearest face normal
    grasped: bool = False  # ground truth: welded to the gripper after closing
    lifted: bool = False  # ground truth: rose with the gripper
    other_disturbed_mm: float = float("nan")  # ground truth: the other connector's displacement during the pick
    in_hand_tip_error_mm: float = float("nan")  # estimate vs ground truth tip, after lift
    inserted_belief: bool = False  # F/T check: the controller believes it reached the hole bottom
    seated_visual: bool = False  # camera check after release: top at the seated height, centred on the socket
    visual_height_mm: float = float("nan")  # perceived connector top above its socket's top surface
    visual_offset_mm: float = float("nan")  # perceived connector top centre vs socket opening centre, xy
    verified: bool = False  # both checks agree it is seated (what the robot reports as done)
    seated: bool = False  # ground truth, before release
    stayed_seated: bool = False  # ground truth, after release and retract
    wrong_socket: bool = False
    peak_force_10ms: float = 0.0  # [N] ground-truth connector contact force (excluding fingers)
    insert_time: float = float("nan")  # [s]
    search_info: dict = field(default_factory=dict)
    failure: str = ""


def visual_seating(connector: ConnectorEstimate | None, socket: SocketEstimate | None, cfg: AssemblyConfig,
                   height_tolerance: float = 0.003, xy_tolerance: float = 0.002) -> tuple[bool, float, float]:
    """Is the connector seated in the socket, judged from an overhead capture? Returns (seated, height, offset) [m].

    A seated connector's top stands connector_length − socket_depth (30 mm) above the fixture top; one resting on
    the chamfer stands ~49 mm, one wedged part-way in somewhere between. The gap is many times the depth noise, and
    the same camera's calibration bias largely cancels between the two estimates.

    Pass the socket estimate from a capture with the hole still empty (the one the insertion used). With a
    connector standing in it, the off-centre camera sees the connector's side hide part of the fixture top, and the
    detected opening shifts 0.7–1.7 mm away from the image centre (ideal camera) — about the xy tolerance.
    """
    if connector is None or socket is None:
        return False, float("nan"), float("nan")
    height = float(connector.top[2] - socket.entrance[2])
    offset = float(np.linalg.norm(connector.top[:2] - socket.entrance[:2]))
    seated = abs(height - (cfg.connector_length - cfg.socket_depth)) <= height_tolerance and offset <= xy_tolerance
    return seated, height, offset


@dataclass
class AssemblyResult:
    seed: int
    connectors: list[ConnectorLog] = field(default_factory=list)
    both_seated_at_end: bool = False
    sim_time: float = 0.0
    trajectory: list = field(default_factory=list)  # (time, qpos) samples for replay
    captures: list = field(default_factory=list)  # (time, CameraObservation, SceneEstimate, proposals)


class AssemblyRun:
    def __init__(
        self,
        cfg: AssemblyConfig,
        seed: int = 0,
        noise: DepthNoise | None = None,
        insert_budget: float = 60.0,
        record: bool = False,
        grasp_cfg: GraspConfig | None = None,
    ):
        self.cfg, self.seed = cfg, seed
        self.motion = MotionConfig()
        self.sim = AssemblySim(build_assembly_model(cfg), cfg)
        self.sim.reset()
        self.robot = Robot(self.sim, noise, np.random.default_rng([seed, 5]))
        self.ik = JacobianIK(self.sim.model, "tcp")
        self.runner = SegmentRunner(self.sim.model, self.ik, JointPDConfig())
        self.estimator = GeometricEstimator()
        self.grasp_cfg = grasp_cfg or GraspConfig()
        self.insert_budget = insert_budget
        self.seat_force = 2.0  # [N] F/T force while pressing on the hole bottom
        self.seat_settle = 0.3  # [s] stopped for this long
        self.seat_depth_tolerance = 0.005  # [m] above the believed bottom (covers calibration height bias)
        self.record = record
        self.result = AssemblyResult(seed=seed)
        self._force_window = np.zeros(10)
        self._force_i = 0
        self._active_kind: str | None = None
        self._insert_sockets: dict[str, SocketEstimate] = {}  # socket estimates the insertions used (hole empty)
        self.robot.on_step.append(self._log_step)

    # --- ground-truth logging (outside the controller's view) ---------------------------------------

    def _log_step(self, sim: AssemblySim) -> None:
        if self._active_kind is not None:
            force = sim.connector_contact_force(self._active_kind)
            self._force_window[self._force_i % len(self._force_window)] = force
            self._force_i += 1
            log = self._log(self._active_kind)
            log.peak_force_10ms = max(log.peak_force_10ms, float(self._force_window.mean()))
        if self.record and int(round(sim.time / sim.model.opt.timestep)) % 10 == 0:
            self.result.trajectory.append((sim.time, sim.data.qpos.copy()))

    def _log(self, kind: str) -> ConnectorLog:
        return next(log for log in self.result.connectors if log.kind == kind)

    # --- motion helpers ----------------------------------------------------------------------------------

    def _play(self, segments, settle: float = 0.2, until=None) -> bool:
        start = self.robot.time
        self.runner.start(segments, now=start)
        end = start + sum(s.duration for s in segments) + settle
        q_hold = self.q_hold()
        while self.robot.time < end:
            q_des, qd_des = self.runner.target(self.robot.time)
            if q_des is None:
                q_des, qd_des = q_hold, np.zeros(6)
            self.robot.step(self.runner.torque(self.robot.q(), self.robot.qd(), q_des, qd_des))
            if until is not None and until():
                return True
        return until() if until is not None else True

    def q_hold(self) -> np.ndarray:
        return self.runner.hold_q if self.runner.hold_q is not None else self.robot.q()

    def _line(self, start, end, rot, speed=0.05) -> LineSegment:
        return LineSegment(self.ik, self.robot.q(), LineTrajectory(start, end, speed, self.motion.cart_amax), rot)

    def _tcp(self):
        return self.ik.fk(self.robot.q())

    def move_to_ready(self) -> None:
        """Up to a safe height, then a joint move to the `ready` pose (outside the camera's view)."""
        tcp, rot = self._tcp()
        if tcp[2] < 0.33:
            self._play([self._line(tcp, np.array([tcp[0], tcp[1], 0.33]), rot, speed=0.15)])
        q_ready = self.sim.model.key("ready").qpos[self.ik.qpos_adr]
        self._play([JointSegment(QuinticTrajectory.with_limits(self.robot.q(), q_ready, self.motion.joint_vmax, self.motion.joint_amax))], settle=0.3)

    # --- skills ------------------------------------------------------------------------------------------------

    def perceive(self) -> SceneEstimate:
        observation = self.robot.capture()
        scene = self.estimator.estimate(observation)
        self.result.captures.append((self.robot.time, observation, scene, []))
        return scene

    def pick(self, kind: str, scene: SceneEstimate) -> InHandEstimate | None:
        log = self._log(kind)
        estimate = scene.connector(kind)
        log.identified = estimate is not None
        if estimate is None:
            log.failure = "not_identified"
            return None
        others = [c.top[:2] for c in scene.connectors if c is not estimate]
        proposals = propose_grasps(estimate.outline_xy, estimate.top[2], self.cfg, self.grasp_cfg, others, preferred_tool_yaw=0.0)
        log.proposals = len(proposals)
        self.result.captures[-1][3].extend(proposals[:1])
        if not proposals:
            log.failure = "no_grasp"
            return None
        grasp = proposals[0]
        log.grasp = grasp
        if kind == "square":
            true_rot = self.sim.data.body(f"{kind}_connector").xmat.reshape(3, 3)
            true_yaw = math.atan2(true_rot[1, 0], true_rot[0, 0])
            log.grasp_face_error_deg = math.degrees(abs((grasp.closing_angle - true_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4))
        rot = tool_down(grasp.tool_yaw)
        above = grasp.tcp + [0, 0, 0.08]
        other_kind = "square" if kind == "round" else "round"
        other_start = self.sim.data.body(f"{other_kind}_connector").xpos.copy()

        _, approach = plan_approach(self.ik, self.robot.q(), above, rot, self.motion, self.seed)
        self._play(approach)
        self._play([self._line(above, grasp.tcp, rot)])
        self.robot.command_gripper(close=True)
        # The controller sees only finger proprioception. Read the finger positions when the gap first closes to
        # the connector's width (both pads touching): that midpoint is where the connector is locked in the hand.
        # Later readings drift as the pads keep squeezing into the contact.
        expected = grasp.width
        touch = {}

        def fingers_touching() -> bool:
            if not touch and self.robot.finger_gap() <= expected + 0.0003:
                touch["closing"] = self.robot.finger_closing()
            return False

        self._play([], settle=1.2, until=fingers_touching)  # 15 mm of travel at 20 mm/s, then settle
        gap = self.robot.finger_gap()
        log.grasped = self.sim.attached() == kind  # ground truth
        if abs(gap - expected) > 0.003:
            log.failure = "grasp_gap_mismatch"
            self.robot.command_gripper(close=False)
            return None
        conn_z0 = self.sim.data.body(f"{kind}_connector").xpos[2]
        tcp_grasp, rot_grasp = self._tcp()
        in_hand = self._in_hand_estimate(estimate.top, tcp_grasp, rot_grasp, *touch.get("closing", self.robot.finger_closing()))
        self._play([self._line(grasp.tcp, above, rot)])
        log.lifted = self.sim.data.body(f"{kind}_connector").xpos[2] > conn_z0 + 0.06
        log.other_disturbed_mm = float(np.linalg.norm(self.sim.data.body(f"{other_kind}_connector").xpos - other_start) * 1e3)
        tcp, rot_now = self._tcp()
        true_tip = self.sim.data.site(f"{kind}_tip").xpos
        log.in_hand_tip_error_mm = float(np.linalg.norm(tcp + rot_now @ in_hand.tip_in_tool - true_tip) * 1e3)
        if not log.lifted:
            log.failure = "not_lifted"
        return in_hand

    def _in_hand_estimate(self, perceived_top: np.ndarray, tcp: np.ndarray, rot: np.ndarray, left: float, right: float) -> InHandEstimate:
        """Connector tip in the tool frame.

        Along the closing axis (tool y) the connector sits midway between the finger pads, which the finger
        joint positions give directly (the left pad closes toward −y, the right toward +y): y = (right − left)/2.
        Across the fingers and in height, use the perceived connector top. A square gripped across its flats is
        squared up to the fingers, so its axes align with the tool's.
        """
        tip_world = np.array([perceived_top[0], perceived_top[1], perceived_top[2] - self.cfg.connector_length])
        offset = rot.T @ (tip_world - tcp)
        offset[1] = (right - left) / 2
        return InHandEstimate(tip_in_tool=offset, rot_in_tool=DOWN.copy())

    def insert(self, kind: str, scene: SceneEstimate, in_hand: InHandEstimate) -> None:
        log = self._log(kind)
        socket = scene.socket(kind)
        if socket is None:
            log.failure = "socket_not_identified"
            return
        self._insert_sockets[kind] = socket
        tcp_local = self.sim.model.site_pos[self.robot.tcp_site].copy()
        self.robot.set_tool_tip(tcp_local + in_hand.tip_in_tool, in_hand.rot_in_tool)

        _, tip_rot = self.sim.site_pose("tool_tip")
        current_yaw = math.atan2(tip_rot[1, 0], tip_rot[0, 0])
        if kind == "square":
            # Nearest socket-aligned yaw (90° symmetry) to the connector's current yaw.
            k = round((current_yaw - socket.yaw) / (math.pi / 2))
            target_yaw = socket.yaw + k * math.pi / 2
        else:
            target_yaw = current_yaw
        target = InsertionTarget(believed_entrance=socket.entrance, depth=self.cfg.socket_depth, tip_rot=rot_z(target_yaw))

        self.robot.tare(self.q_hold(), self.runner)
        controller = SpiralSearchInsert(tip_site="tool_tip")
        t0 = self.robot.time
        controller.reset(self.sim.model, self.robot.q(), target, self.seed)
        bottom_z = socket.entrance[2] - self.cfg.socket_depth
        self._active_kind = kind
        still_since = None
        while self.robot.time - t0 < self.insert_budget:
            obs = self.robot.observation(t0)
            self.robot.step(controller.act(obs))
            # Completion is mechanical, not a height threshold: the estimated tip height carries the camera's
            # calibration bias, so a height test alone can stop short and let the connector drop in on release.
            # Seated = not searching, pushing against the bottom (F/T force), stopped for `settle`, and deep enough.
            bottomed = (
                controller.phase != SpiralSearchInsert.SEARCH  # straight in (descend) or after search (insert)
                and np.linalg.norm(obs.ft[:3]) > self.seat_force
                and np.linalg.norm(obs.qd) < 0.02
                and obs.tip_pos[2] < bottom_z + self.seat_depth_tolerance
            )
            still_since = (still_since or self.robot.time) if bottomed else None
            if still_since is not None and self.robot.time - still_since >= self.seat_settle:
                log.inserted_belief = True
                break
        self._active_kind = None
        log.insert_time = self.robot.time - t0
        log.search_info = controller.info()
        log.seated = self.sim.seated(kind)
        log.wrong_socket = self.sim.connector_in_wrong_socket(kind)
        if not log.inserted_belief:
            log.failure = "insert_timeout"
        # Hold the final impedance posture as a joint target while releasing.
        self.runner.hold_q = self.robot.q()
        self.runner.current = None

    def release_and_retract(self, kind: str) -> None:
        self.robot.command_gripper(close=False)
        self._play([], settle=0.5)
        tcp, rot = self._tcp()
        self._play([self._line(tcp, tcp + [0, 0, 0.08], rot)])
        tcp_local = self.sim.model.site_pos[self.robot.tcp_site].copy()
        self.robot.set_tool_tip(tcp_local, np.eye(3))

    def verify(self, kind: str, scene: SceneEstimate) -> None:
        """Visual insertion check on a capture taken after release; combined with the F/T check into `verified`."""
        log = self._log(kind)
        socket = self._insert_sockets.get(kind)
        log.seated_visual, height, offset = visual_seating(scene.connector(kind), socket, self.cfg)
        log.visual_height_mm, log.visual_offset_mm = height * 1e3, offset * 1e3
        log.verified = log.inserted_belief and log.seated_visual
        if log.inserted_belief and not log.seated_visual:
            log.failure = "visual_check_failed"

    def run(self) -> AssemblyResult:
        self.result.connectors = [ConnectorLog(kind=k) for k in CONNECTORS]
        self.move_to_ready()  # already there after reset; settles and gives the PD a hold target
        q0 = self.robot.q()
        self.runner.hold_q = q0
        self.robot.tare(q0, self.runner)
        scene = self.perceive()
        for kind in CONNECTORS:  # round first
            in_hand = self.pick(kind, scene)
            inserted = in_hand is not None and self._log(kind).lifted
            if inserted:
                self.insert(kind, scene, in_hand)
                self.release_and_retract(kind)
            else:
                self.robot.command_gripper(close=False)
                self._play([], settle=0.3)
            self.move_to_ready()
            scene = self.perceive()  # verifies this insertion and is the scene for the next pick
            if inserted:
                self.verify(kind, scene)
        self._play([], settle=0.5)
        for log in self.result.connectors:
            log.stayed_seated = self.sim.seated(log.kind)
        self.result.both_seated_at_end = all(log.stayed_seated for log in self.result.connectors)
        self.result.sim_time = self.robot.time
        return self.result
