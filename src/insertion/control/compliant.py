"""Compliant insertion controllers built on Cartesian impedance.

`ImpedanceInsert`: collision-avoiding joint-space approach (joint PD) to the pre-insertion pose,
then switch to Cartesian impedance and move the setpoint straight down through the believed
entrance to just below the nominal hole bottom. The setpoint slows in a touchdown zone around
the entrance to limit the impact spike. Lateral and rotational compliance let the chamfer guide
the peg in when the offset is within its capture range; if the peg lands outside it, the arm
just presses on the surface with force bounded by the spring saturation.

`SpiralSearchInsert`: the same approach and touchdown, but when the wrist F/T detects contact
above the hole it searches: press with a fixed normal force while the setpoint sweeps an
Archimedean spiral around the believed entrance. A small drop below the contact height pauses
the spiral (the chamfer may be guiding the peg in); a drop deeper than the chamfer confirms the
hole, and the controller switches back to soft insertion gains and pushes the peg down.
"""

import mujoco
import numpy as np

from insertion.config import ImpedanceConfig, ImpedanceInsertConfig, JointPDConfig, MotionConfig, SensorConfig, SpiralSearchConfig
from insertion.control.ik import JacobianIK
from insertion.control.impedance import CartesianImpedance
from insertion.control.motion import SegmentRunner
from insertion.control.scripted import plan_approach, pointing_down
from insertion.control.trajectory import LineTrajectory
from insertion.harness import InsertionTarget, Observation
from insertion.sensors import ContactDetector


class SetpointPath:
    """Consecutive straight-line setpoint moves, sampled by time since the path started."""

    def __init__(self, lines: list[LineTrajectory]):
        self.lines = lines
        self.starts = np.cumsum([0.0] + [line.duration for line in lines[:-1]])
        self.duration = float(self.starts[-1] + lines[-1].duration)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        index = int(np.searchsorted(self.starts, t, side="right")) - 1
        line = self.lines[max(index, 0)]
        return line.sample(t - self.starts[max(index, 0)])


class ImpedanceInsert:
    def __init__(
        self,
        cfg: ImpedanceInsertConfig | None = None,
        impedance: ImpedanceConfig | None = None,
        motion: MotionConfig | None = None,
        pd: JointPDConfig | None = None,
    ):
        self.cfg = cfg or ImpedanceInsertConfig()
        self.impedance_cfg = impedance or ImpedanceConfig()
        self.motion = motion or MotionConfig()
        self.pd = pd or JointPDConfig()

    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget, seed: int = 0) -> None:
        c, m = self.cfg, self.motion
        self.rot = pointing_down()
        ik = JacobianIK(model, "peg_tip")
        entrance = np.asarray(target.believed_entrance, dtype=float)
        pre = entrance + [0.0, 0.0, m.approach_height]
        q_pre, approach = plan_approach(ik, np.asarray(q0), pre, self.rot, m, seed)
        self.runner = SegmentRunner(model, ik, self.pd)
        self.runner.start(approach, now=0.0)

        above = entrance + [0.0, 0.0, c.touchdown_zone]
        below = entrance - [0.0, 0.0, c.touchdown_zone]
        goal = entrance - [0.0, 0.0, target.depth + c.press_depth]
        self.path = SetpointPath([
            LineTrajectory(pre, above, c.descend_speed, m.cart_amax),
            LineTrajectory(above, below, c.touchdown_speed, m.cart_amax),
            LineTrajectory(below, goal, c.descend_speed, m.cart_amax),
        ])
        self.impedance = CartesianImpedance(model, self.impedance_cfg)
        self.impedance.set_damping_at(q_pre)
        self.path_time: float | None = None  # None until the approach finishes
        self.last_time = 0.0

    def act(self, obs: Observation) -> np.ndarray:
        dt, self.last_time = obs.time - self.last_time, obs.time
        if self.path_time is None:
            q_des, qd_des = self.runner.target(obs.time)
            if not self.runner.idle:
                return self.runner.torque(obs.q, obs.qd, q_des, qd_des)
            self.path_time = 0.0  # approach finished: switch to impedance

        return self._descend_torque(obs, dt)

    def _descend_torque(self, obs: Observation, dt: float) -> np.ndarray:
        pos_des, vel_des = self.path.sample(self.path_time)
        lag = obs.tip_pos[2] - pos_des[2]
        if lag > self.cfg.max_lead:
            pos_des = pos_des.copy()
            pos_des[2] = obs.tip_pos[2] - self.cfg.max_lead  # leash: hold the setpoint max_lead ahead
            vel_des = np.zeros(3)
        else:
            self.path_time += dt  # path only advances while the tip keeps up
        tau, _ = self.impedance.torque(obs.q, obs.qd, pos_des, self.rot, vel_des)
        return tau


class SpiralSearchInsert(ImpedanceInsert):
    DESCEND, SEARCH, INSERT = "descend", "search", "insert"

    def __init__(self, search: SpiralSearchConfig | None = None, sensor: SensorConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.search_cfg = search or SpiralSearchConfig()
        self.sensor_cfg = sensor or SensorConfig()

    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget, seed: int = 0) -> None:
        super().reset(model, q0, target, seed)
        self.entrance = np.asarray(target.believed_entrance, dtype=float)
        self.depth = target.depth
        self.insert_stiffness = self.impedance.stiffness.copy()
        self.search_stiffness = self.insert_stiffness.copy()
        self.search_stiffness[:2] = self.search_cfg.search_stiffness_xy
        self.detector = ContactDetector(self.sensor_cfg)
        self.phase = self.DESCEND
        self.contact_z = None
        self.theta = 0.0
        self.paused_until = 0.0
        self.search_start = None
        self.found_time = None
        self.pauses = 0
        self.restarts = 0
        self.stalls = 0.0  # [s] spiral held by the lateral leash
        self.stall_run = 0.0  # [s] current continuous stall
        self.drop_ref = None
        self.drop_at_pause = None

    def _spiral_xy(self) -> np.ndarray:
        radius = self.search_cfg.pitch * self.theta / (2 * np.pi)
        return self.entrance[:2] + radius * np.array([np.cos(self.theta), np.sin(self.theta)])

    def act(self, obs: Observation) -> np.ndarray:
        c = self.search_cfg
        dt, self.last_time = obs.time - self.last_time, obs.time
        if self.path_time is None:
            q_des, qd_des = self.runner.target(obs.time)
            if not self.runner.idle:
                return self.runner.torque(obs.q, obs.qd, q_des, qd_des)
            self.path_time = 0.0

        in_contact = self.detector.update(obs.ft)
        tip = obs.tip_pos

        if self.phase == self.DESCEND and in_contact and tip[2] > self.entrance[2] - c.hole_drop:
            # Blocked at the surface or chamfer: start searching from the believed entrance. Begin paused
            # (soft) so an offset within the chamfer's capture range can slide in without spiralling.
            self.phase = self.SEARCH
            self.contact_z = tip[2]
            self.search_start = obs.time
            self.search_xy = self.entrance[:2].copy()
            self._pause(obs)

        if self.phase == self.SEARCH:
            drop = self.contact_z - tip[2]
            if drop > c.hole_drop:
                self._start_insert(obs)
            else:
                if obs.time < self.paused_until:
                    self._center(obs, dt)
                elif self.drop_at_pause is not None and drop - self.drop_at_pause > c.progress_drop:
                    self.paused_until = obs.time + c.pause_time  # still sliding in: keep centring
                    self.drop_at_pause = drop
                else:
                    if self.drop_ref is None:
                        self.drop_ref = drop  # pause just ended: measure new drops from where the peg settled
                        self.drop_at_pause = None
                    if drop - self.drop_ref > c.partial_drop or self.stall_run > c.stall_timeout:
                        self.pauses += 1
                        self._pause(obs)  # new dip or stalled on the chamfer: stop sweeping, centre on force
                    else:
                        self._advance_spiral(dt, tip)
                pos_des = np.array([*self.search_xy, self.contact_z - c.search_force / self.impedance.stiffness[2]])
                return self.impedance.torque(obs.q, obs.qd, pos_des, self.rot)[0]

        if self.phase == self.INSERT:
            pos_des, vel_des = self.insert_path.sample(obs.time - self.insert_start)
            if tip[2] - pos_des[2] > self.cfg.max_lead:
                pos_des = np.array([pos_des[0], pos_des[1], tip[2] - self.cfg.max_lead])
                vel_des = None
            return self.impedance.torque(obs.q, obs.qd, pos_des, self.rot, vel_des)[0]

        return super()._descend_torque(obs, dt)

    def _pause(self, obs: Observation) -> None:
        """Stop the spiral: soft insertion gains, xy setpoint at the tip, then centre along the sensed force."""
        self.paused_until = obs.time + self.search_cfg.pause_time
        self.drop_ref = None
        self.drop_at_pause = self.contact_z - obs.tip_pos[2]
        self.stall_run = 0.0
        self.search_xy = obs.tip_pos[:2].copy()
        self.impedance.set_stiffness(self.insert_stiffness)

    def _center(self, obs: Observation, dt: float) -> None:
        """Move the xy setpoint along the lateral external force (world frame) felt at the wrist."""
        c = self.search_cfg
        force_xy = (obs.tip_rot @ obs.ft[:3])[:2]
        velocity = c.center_gain * force_xy
        speed = np.linalg.norm(velocity)
        if speed > c.center_speed_max:
            velocity *= c.center_speed_max / speed
        self.search_xy = self.search_xy + velocity * dt

    def _advance_spiral(self, dt: float, tip: np.ndarray) -> None:
        c = self.search_cfg
        if self.impedance.stiffness[0] != self.search_stiffness[0]:
            self.impedance.set_stiffness(self.search_stiffness)
        lead = self.search_xy - tip[:2]
        if np.linalg.norm(lead) > c.max_lateral_lead:
            self.stalls += dt  # tip held back (friction or a wall): don't wind the spiral further ahead
            self.stall_run += dt
            return
        self.stall_run = 0.0
        radius = c.pitch * self.theta / (2 * np.pi)
        self.theta += min(c.max_angular_speed, c.search_speed / max(radius, 1e-4)) * dt
        if radius > c.search_radius:
            self.theta = 0.0
            self.restarts += 1
        self.search_xy = self._spiral_xy()

    def _start_insert(self, obs: Observation) -> None:
        self.phase = self.INSERT
        self.found_time = obs.time
        self.impedance.set_stiffness(self.insert_stiffness)
        start = obs.tip_pos.copy()
        goal = np.array([start[0], start[1], self.entrance[2] - self.depth - self.cfg.press_depth])
        self.insert_path = SetpointPath([LineTrajectory(start, goal, self.cfg.descend_speed, self.motion.cart_amax)])
        self.insert_start = obs.time

    def info(self) -> dict:
        return {
            "searched": self.search_start is not None,
            "search_time": (self.found_time or self.last_time) - self.search_start if self.search_start is not None else 0.0,
            "found": self.found_time is not None,
            "pauses": self.pauses,
            "spiral_restarts": self.restarts,
            "stall_time": self.stalls,
        }
