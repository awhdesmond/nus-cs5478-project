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
from insertion.assembly_planning import ClearanceChecker, joint_path_clear, perceived_obstacles, plan_sideways_insertion
from insertion.assembly_scene import CONNECTORS, PAD_DEPTH, build_assembly_model, connector_half_width
from insertion.assembly_sim import AssemblySim
from insertion.camera import DepthNoise, OverheadCamera
from insertion.config import AssemblyConfig, ImpedanceInsertConfig, JointPDConfig, MotionConfig, SensorConfig, SpiralSearchConfig
from insertion.control.compliant import SpiralSearchInsert
from insertion.control.ik import JacobianIK
from insertion.control.motion import JointSegment, LineSegment, SegmentRunner
from insertion.control.scripted import plan_approach
from insertion.control.trajectory import LineTrajectory, QuinticTrajectory
from insertion.grasping import GraspConfig, GraspProposal, propose_grasps
from insertion.harness import InsertionTarget, Observation
from insertion.perception import ConnectorEstimate, GeometricEstimator, SceneEstimate, SocketEstimate
from insertion.perception_noise import ObjectOffsetEstimator
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
    tip_in_tool: np.ndarray  # connector's marked (insertion) end in the tcp (tool) frame [m]
    rot_in_tool: np.ndarray  # connector axes in the tool frame (+z from the marked end toward the top)
    lying: bool = False  # picked up lying: the hand must turn sideways to insert it


@dataclass
class ConnectorLog:
    kind: str
    identified: bool = False
    proposals: int = 0
    grasp: GraspProposal | None = None
    lying: bool = False  # ground truth: started lying on its side
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
    wrong_end: bool = False  # ground truth: inserted unmarked end first
    sideways_yaw_deg: float = float("nan")  # lying: chosen insertion yaw (finger pointing direction)
    sideways_clearance_mm: float = float("nan")  # lying: planned worst arm-obstacle clearance
    # Perception at the capture this connector's pick used (ground truth comparison, for scoring)
    identified_correctly: bool = False  # the estimate labelled with this kind lies on the true connector of this kind
    connector_xy_error_mm: float = float("nan")  # upright: top centre; lying: axis centre
    socket_xy_error_mm: float = float("nan")  # entrance centre
    relative_xy_error_mm: float = float("nan")  # |connector error − socket error|: what doesn't cancel on insertion
    cycle_time: float = float("nan")  # [s] pick start → post-release capture and check (or the recapture after a failed pick)
    peak_force_10ms: float = 0.0  # [N] ground-truth connector contact force (excluding fingers)
    insert_time: float = float("nan")  # [s]
    search_info: dict = field(default_factory=dict)
    failure: str = ""

    @property
    def failure_stage(self) -> str:
        """Pipeline stage the connector failed at ("" if it ended seated and verified)."""
        if self.failure:
            return FAILURE_STAGES.get(self.failure, "other")
        if self.wrong_socket or self.wrong_end:
            return "insertion"
        if self.seated and not self.stayed_seated:
            return "release"
        if self.stayed_seated and not self.verified:
            return "verification"  # physically fine, but the robot doesn't believe it
        return ""


FAILURE_STAGES = {
    "not_identified": "perception", "tip_not_found": "perception", "socket_not_identified": "perception",
    "no_grasp": "grasp", "grasp_gap_mismatch": "grasp", "not_lifted": "grasp",
    "no_sideways_plan": "planning", "carry_path_blocked": "planning",
    "insert_timeout": "insertion",
    "visual_check_failed": "verification",
}


def visual_seating(connector: ConnectorEstimate | None, socket: SocketEstimate | None, cfg: AssemblyConfig,
                   height_tolerance: float = 0.003, xy_margin: float = 0.001) -> tuple[bool, float, float]:
    """Is the connector seated in the socket, judged from an overhead capture? Returns (seated, height, offset) [m].

    A seated connector's top stands connector_length − socket_depth (30 mm) above the fixture top; one resting on
    the chamfer stands ~49 mm, one wedged part-way in somewhere between. The gap is many times the depth noise, and
    the same camera's calibration bias largely cancels between the two estimates.

    Pass the socket estimate from a capture with the hole still empty (the one the insertion used). With a
    connector standing in it, the off-centre camera sees the connector's side hide part of the fixture top, and the
    detected opening shifts 0.7-1.7 mm away from the image centre (ideal camera).

    xy tolerance = half the clearance (a seated connector can sit that far off the hole axis) + `xy_margin` for
    perception: over 160 benchmark runs the realistic camera's offset exceeded half the clearance by ≤ 0.51 mm, and
    heights stayed within 0.54 mm of 30 mm.
    """
    if connector is None or socket is None:
        return False, float("nan"), float("nan")
    height = float(connector.top[2] - socket.entrance[2])
    offset = float(np.linalg.norm(connector.top[:2] - socket.entrance[:2]))
    seated = abs(height - (cfg.connector_length - cfg.socket_depth)) <= height_tolerance and offset <= cfg.clearance / 2 + xy_margin
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
        sim: AssemblySim | None = None,
        object_error: float = 0.0,
    ):
        """`sim`: reuse an existing simulation (e.g. one shown in a viewer), moved to this layout in place.
        `object_error`: independent per-object perception offsets, uniform in ±object_error per horizontal axis [m]."""
        self.cfg, self.seed = cfg, seed
        self.motion = MotionConfig()
        if sim is None:
            self.sim = AssemblySim(build_assembly_model(cfg), cfg)
            self.sim.reset()
        else:
            self.sim = sim
            self.sim.reset()
            self.sim.apply_layout(cfg)
        self.robot = Robot(self.sim, noise, np.random.default_rng([seed, 5]))
        self.ik = JacobianIK(self.sim.model, "tcp")
        self.ik_tip = JacobianIK(self.sim.model, "tool_tip")
        self.checker = ClearanceChecker(self.sim.model, self.ik)
        self.plan_margin = 0.005  # [m] arm-obstacle clearance required by the sideways-insertion planner
        self.carry_height = 0.36  # [m] tcp height for lifting a lying connector before turning it sideways
        self.runner = SegmentRunner(self.sim.model, self.ik, JointPDConfig())
        self.object_error = object_error
        self.estimator = ObjectOffsetEstimator(GeometricEstimator(), object_error, np.random.default_rng([seed, 8]))
        # Visual check xy margin: 1 mm, plus the worst connector-vs-socket offset the per-object error spec allows.
        self.visual_xy_margin = 0.001 + 2 * math.sqrt(2) * object_error
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

    def _score_perception(self, kind: str, scene: SceneEstimate) -> None:
        """Ground-truth comparison of the estimates this pick uses (scoring only; the controller never sees it)."""
        log = self._log(kind)
        data = self.sim.data
        estimate = scene.connector(kind)
        if estimate is not None:
            xy = estimate.centre[:2] if estimate.lying else estimate.top[:2]
            nearest = min(CONNECTORS, key=lambda k: np.linalg.norm(data.body(f"{k}_connector").xpos[:2] - xy))
            log.identified_correctly = nearest == kind
            connector_error = xy - data.body(f"{kind}_connector").xpos[:2]
            log.connector_xy_error_mm = float(np.linalg.norm(connector_error) * 1e3)
        socket = scene.socket(kind)
        if socket is not None:
            socket_error = socket.entrance[:2] - data.site(f"{kind}_socket_entrance").xpos[:2]
            log.socket_xy_error_mm = float(np.linalg.norm(socket_error) * 1e3)
            if estimate is not None:
                log.relative_xy_error_mm = float(np.linalg.norm(connector_error - socket_error) * 1e3)

    def pick(self, kind: str, scene: SceneEstimate) -> InHandEstimate | None:
        log = self._log(kind)
        self._score_perception(kind, scene)
        estimate = scene.connector(kind)
        log.identified = estimate is not None
        log.lying = (self.cfg.round_pose if kind == "round" else self.cfg.square_pose).lying  # ground truth, for scoring
        if estimate is None:
            log.failure = "not_identified"
            return None
        # Keep the fingers clear of every other perceived part's outline (not just its centre: lying parts are long).
        others = [point for c in scene.connectors if c is not estimate for point in c.outline_xy[::3]]
        if estimate.lying:
            if not estimate.tip_found:
                log.failure = "tip_not_found"
                return None
            grasp = self._lying_grasp(kind, estimate, scene)
            proposals = [grasp]
        else:
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
            closing = np.array([math.cos(grasp.closing_angle), math.sin(grasp.closing_angle)])
            if estimate.lying:  # face normals across the width: the body's horizontal y axis
                log.grasp_face_error_deg = math.degrees(math.acos(min(1.0, abs(float(closing @ true_rot[:2, 1])))))
            else:
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
        closing = touch.get("closing", self.robot.finger_closing())
        if estimate.lying:
            in_hand = self._in_hand_lying(kind, estimate, tcp_grasp, rot_grasp, *closing)
        else:
            in_hand = self._in_hand_estimate(estimate.top, tcp_grasp, rot_grasp, *closing)
        self._play([self._line(grasp.tcp, above, rot)])
        log.lifted = self.sim.data.body(f"{kind}_connector").xpos[2] > conn_z0 + 0.06
        log.other_disturbed_mm = float(np.linalg.norm(self.sim.data.body(f"{other_kind}_connector").xpos - other_start) * 1e3)
        tcp, rot_now = self._tcp()
        true_tip = self.sim.data.site(f"{kind}_tip").xpos
        log.in_hand_tip_error_mm = float(np.linalg.norm(tcp + rot_now @ in_hand.tip_in_tool - true_tip) * 1e3)
        if not log.lifted:
            log.failure = "not_lifted"
        return in_hand

    def _lying_grasp(self, kind: str, estimate, scene: SceneEstimate) -> GraspProposal:
        """Pinch a lying connector across its width, pads centred `grasp_below_top` from its unmarked end.

        Flat pads touch a cylinder along its widest line and a lying square on its side faces, so the grasp is
        antipodal with the contact normals exactly along the closing direction. The fingertips stop
        `lying_fingertip_clearance` above the perceived table.
        """
        cfg = self.cfg
        axis = np.asarray(estimate.axis_xy)  # marked end → top
        width = 2 * connector_half_width(cfg, kind)
        tcp_xy = estimate.centre[:2] + axis * (cfg.connector_length / 2 - cfg.grasp_below_top)
        tcp = np.array([*tcp_xy, estimate.centre[2] + cfg.lying_fingertip_clearance + PAD_DEPTH])
        # tool_down(yaw) closes along yaw − 90° and has its x axis along yaw: put the connector axis along tool ±x,
        # choosing the wrist yaw nearest zero.
        yaw = math.atan2(axis[1], axis[0])
        tool_yaw = min((math.remainder(yaw + k * math.pi, 2 * math.pi) for k in (0, 1)), key=abs)
        cone = math.degrees(math.atan(self.grasp_cfg.friction_assumed))
        return GraspProposal(tcp, tool_yaw, math.remainder(tool_yaw - math.pi / 2, math.pi), width, cone, cfg.finger_width, 0.0)

    def _in_hand_lying(self, kind: str, estimate, tcp: np.ndarray, rot: np.ndarray, left: float, right: float) -> InHandEstimate:
        """Marked end and connector axes in the tool frame for a connector picked up lying.

        From perception: the axis direction (marked end → top), the centre along it, and the table height (the axis
        rests half a width above it). Along the closing axis, the finger pad midpoint from proprioception.
        """
        cfg = self.cfg
        axis = np.array([*estimate.axis_xy, 0.0])
        down = np.array([0.0, 0.0, -1.0])  # a lying square rests on the face whose normal is its local x
        centre = np.array([*estimate.centre[:2], estimate.centre[2] + connector_half_width(cfg, kind)])
        tip_world = centre - axis * cfg.connector_length / 2
        offset = rot.T @ (tip_world - tcp)
        offset[1] = (right - left) / 2
        rot_world = np.column_stack([down, np.cross(axis, down), axis])
        return InHandEstimate(tip_in_tool=offset, rot_in_tool=rot.T @ rot_world, lying=True)

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

    def _turn_sideways(self, kind: str, scene: SceneEstimate, socket) -> float | None:
        """Carry a connector picked up lying to above its socket with the hand turned sideways; returns its yaw.

        Lift straight up, joint-move to a via above the pre-insertion pose (checked: arm clear of perceived obstacles,
        the held connector's tip above every obstacle top), then a straight line down to the pre-insertion pose, so
        the insertion controller starts there without re-planning the arm configuration.
        """
        log = self._log(kind)
        boxes = perceived_obstacles(scene, self.cfg, carried=kind)
        self.checker.set_obstacles(boxes)
        if kind == "square":
            yaws = [socket.yaw + k * math.pi / 2 for k in range(4)]
        else:
            yaws = [2 * math.pi * k / 12 for k in range(12)]
        plan = plan_sideways_insertion(self.ik_tip, self.checker, socket.entrance, self.cfg.socket_depth, yaws, self.robot.q(),
                                       self.motion.approach_height, self.motion.via_height, self.plan_margin, self.seed)
        if plan is None:
            log.failure = "no_sideways_plan"
            return None
        log.sideways_yaw_deg = math.degrees(plan.yaw) % 360
        log.sideways_clearance_mm = plan.clearance * 1e3

        tcp, rot = self._tcp()
        lifted = np.array([tcp[0], tcp[1], max(tcp[2], self.carry_height)])
        self._play([self._line(tcp, lifted, rot, speed=0.15)])
        traj = QuinticTrajectory.with_limits(self.robot.q(), plan.q_via, self.motion.joint_vmax, self.motion.joint_amax)
        tip_floor = max(box.top for box in boxes) + 0.01
        if not joint_path_clear(self.checker, self.ik_tip, self.robot.q(), plan.q_via, traj, self.plan_margin, tip_floor):
            log.failure = "carry_path_blocked"
            return None
        self._play([JointSegment(traj)], settle=0.2)
        via, _ = self.ik_tip.fk(plan.q_via)
        pre = np.asarray(socket.entrance) + [0, 0, self.motion.approach_height]
        self._play([LineSegment(self.ik_tip, self.robot.q(), LineTrajectory(via, pre, 0.05, self.motion.cart_amax), rot_z(plan.yaw))], settle=0.3)
        self.runner.hold_q = self.robot.q()
        return plan.yaw

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
        if in_hand.lying:
            target_yaw = self._turn_sideways(kind, scene, socket)
            if target_yaw is None:
                return
        elif kind == "square":
            # Nearest socket-aligned yaw (90° symmetry) to the connector's current yaw.
            k = round((current_yaw - socket.yaw) / (math.pi / 2))
            target_yaw = socket.yaw + k * math.pi / 2
        else:
            target_yaw = current_yaw
        target = InsertionTarget(believed_entrance=socket.entrance, depth=self.cfg.socket_depth, tip_rot=rot_z(target_yaw))

        self.robot.tare(self.q_hold(), self.runner)
        if in_hand.lying:
            # The along-axis perception error of a connector picked up lying becomes depth error once it hangs down
            # (it doesn't cancel against the socket estimate as an upright connector's height does): up to ~5 mm.
            # Judge surface contact and seating depth against half the hole depth, and keep pushing past the goal.
            depth_band = self.cfg.socket_depth / 2
            controller = SpiralSearchInsert(tip_site="tool_tip", cfg=ImpedanceInsertConfig(push_after_path=True),
                                            search=SpiralSearchConfig(surface_contact_depth=depth_band))
        else:
            depth_band = self.seat_depth_tolerance
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
                and obs.tip_pos[2] < bottom_z + depth_band
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
        log.wrong_end = self.sim.seated_wrong_end(kind)
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
        log.seated_visual, height, offset = visual_seating(scene.connector(kind), socket, self.cfg, xy_margin=self.visual_xy_margin)
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
            t_start = self.robot.time
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
            self._log(kind).cycle_time = self.robot.time - t_start
        self._play([], settle=0.5)
        for log in self.result.connectors:
            log.stayed_seated = self.sim.seated(log.kind)
        self.result.both_seated_at_end = all(log.stayed_seated for log in self.result.connectors)
        self.result.sim_time = self.robot.time
        return self.result
