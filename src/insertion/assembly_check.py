"""M1 scene checks: a hand-scripted, ground-truth two-connector assembly.

Uses true connector and socket poses (no perception) and stiff joint-PD motion (no compliance or
search), so it validates only the scene mechanics: reach, gripper, attach-on-grasp, transport,
alignment, insertion clearance, release and retract. Round first, then square.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from insertion.assembly_sim import AssemblySim
from insertion.assembly_scene import build_assembly_model
from insertion.config import AssemblyConfig, JointPDConfig, MotionConfig
from insertion.control.ik import JacobianIK
from insertion.control.motion import JointSegment, LineSegment, SegmentRunner
from insertion.control.scripted import plan_approach
from insertion.control.trajectory import LineTrajectory, QuinticTrajectory


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_down(yaw: float) -> np.ndarray:
    """Tool (tcp) frame pointing straight down; fingers close along the world direction yaw − 90°."""
    return rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])


def wrap_quarter(yaw: float) -> float:
    """Equivalent yaw of a square (90° symmetry) closest to zero."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class StageLog:
    connector: str
    grasped: bool = False
    seated: bool = False
    released: bool = False
    wrong_socket: bool = False
    tip_error_at_insert: float = float("nan")  # [m] tip xy vs socket axis when the descent started


@dataclass
class ScriptedAssemblyResult:
    stages: list[StageLog] = field(default_factory=list)
    both_seated_at_end: bool = False
    sim_time: float = 0.0
    max_other_connector_drift: float = 0.0  # [m] how far the waiting connector moved during the first cycle
    trajectory: list = field(default_factory=list)  # qpos snapshots (every 10 steps) for replay


class _Runner:
    def __init__(self, sim: AssemblySim, ik: JacobianIK, record: bool):
        self.sim, self.ik = sim, ik
        self.runner = SegmentRunner(sim.model, ik, JointPDConfig())
        self.dt = sim.model.opt.timestep
        self.record = record
        self.snapshots = []
        self._t = 0.0

    def play(self, segments, settle: float = 0.2, until=None) -> bool:
        self.runner.start(segments, now=self._t)
        end = self._t + sum(s.duration for s in segments) + settle
        while self._t < end:
            q_des, qd_des = self.runner.target(self._t)
            self.sim.step(self.runner.torque(self.sim.q(), self.sim.qd(), q_des, qd_des))
            self._t += self.dt
            if self.record and int(round(self._t / self.dt)) % 10 == 0:
                self.snapshots.append(self.sim.data.qpos.copy())
            if until is not None and until():
                return True
        return until() if until is not None else True

    def hold(self, seconds: float, until=None) -> bool:
        return self.play([], settle=seconds, until=until)


def scripted_assembly(cfg: AssemblyConfig | None = None, record: bool = False) -> ScriptedAssemblyResult:
    cfg = cfg or AssemblyConfig()
    motion = MotionConfig()
    sim = AssemblySim(build_assembly_model(cfg), cfg)
    sim.reset()
    ik = JacobianIK(sim.model, "tcp")
    run = _Runner(sim, ik, record)
    result = ScriptedAssemblyResult()
    other_start = sim.data.body("square_connector").xpos.copy()

    for kind in ("round", "square"):
        log = StageLog(connector=kind)
        result.stages.append(log)
        conn = sim.data.body(f"{kind}_connector")
        conn_yaw = float(np.arctan2(conn.xmat.reshape(3, 3)[1, 0], conn.xmat.reshape(3, 3)[0, 0]))
        grasp_yaw = wrap_quarter(conn_yaw) if kind == "square" else 0.0
        rot = tool_down(grasp_yaw)
        top = conn.xpos.copy() + [0, 0, cfg.connector_length / 2]
        grasp = top - [0, 0, cfg.grasp_below_top]
        above = grasp + [0, 0, 0.08]

        # Approach from wherever the arm is, descend, close.
        q_now = sim.q()
        q_above, approach = plan_approach(ik, q_now, above, rot, motion, seed=0)
        run.play(approach)
        q_grasp, _ = ik.solve(q_above, grasp, rot)
        run.play([LineSegment(ik, q_above, LineTrajectory(above, grasp, 0.05, motion.cart_amax), rot)])
        sim.command_gripper(close=True)
        log.grasped = run.hold(1.0, until=lambda: sim.attached() == kind)
        if not log.grasped:
            break
        run.hold(0.2)

        # Lift, then carry to above the socket with the connector aligned to the socket.
        run.play([LineSegment(ik, q_grasp, LineTrajectory(grasp, above, 0.05, motion.cart_amax), rot)])
        tcp_pos, tcp_rot = ik.fk(sim.q())
        tip = sim.data.site(f"{kind}_tip").xpos.copy()
        tip_in_tool = tcp_rot.T @ (tip - tcp_pos)  # ground-truth in-hand offset
        conn_rot_in_tool = tcp_rot.T @ sim.data.body(f"{kind}_connector").xmat.reshape(3, 3)
        conn_yaw_now = float(np.arctan2(*(tcp_rot @ conn_rot_in_tool)[[1, 0], 0]))
        target_conn_yaw = conn_yaw_now - wrap_quarter(conn_yaw_now) if kind == "square" else conn_yaw_now
        # Connector upright: its rotation is a pure yaw, so tool rot = R_conn_des · R_conn_in_tool⁻¹.
        target_tool_rot = rot_z(target_conn_yaw) @ conn_rot_in_tool.T

        bottom = sim.data.site(f"{kind}_socket_bottom").xpos.copy()
        entrance = sim.data.site(f"{kind}_socket_entrance").xpos.copy()

        def tcp_for_tip(tip_target):
            return tip_target - target_tool_rot @ tip_in_tool

        carry = tcp_for_tip(entrance + [0, 0, 0.06])
        q_carry, ok = ik.solve_with_restarts(sim.q(), carry, target_tool_rot, rng=np.random.default_rng(0))
        if not ok:
            raise RuntimeError(f"IK failed for the {kind} carry pose {carry}")
        run.play([JointSegment(QuinticTrajectory.with_limits(sim.q(), q_carry, motion.joint_vmax, motion.joint_amax))], settle=0.5)
        log.tip_error_at_insert = float(np.linalg.norm(sim.data.site(f"{kind}_tip").xpos[:2] - bottom[:2]))

        # Straight descent until seated (aim slightly below the bottom so the stiff PD keeps pushing).
        start = tcp_for_tip(entrance + [0, 0, 0.06])
        goal = tcp_for_tip(bottom - [0, 0, 0.001])
        log.seated = run.play(
            [LineSegment(ik, q_carry, LineTrajectory(start, goal, 0.01, motion.cart_amax), target_tool_rot)],
            settle=1.0, until=lambda: sim.seated(kind),
        )
        log.wrong_socket = sim.connector_in_wrong_socket(kind)

        # Release and retract.
        sim.command_gripper(close=False)
        run.hold(0.5)
        log.released = sim.attached() is None
        tcp_now, _ = ik.fk(sim.q())
        run.play([LineSegment(ik, sim.q(), LineTrajectory(tcp_now, tcp_now + [0, 0, 0.10], 0.05, motion.cart_amax), target_tool_rot)])
        if kind == "round":
            result.max_other_connector_drift = float(np.linalg.norm(sim.data.body("square_connector").xpos - other_start))

    run.hold(1.0)
    result.both_seated_at_end = all(sim.seated(k) for k in ("round", "square"))
    result.sim_time = run._t
    result.trajectory = run.snapshots
    return result
