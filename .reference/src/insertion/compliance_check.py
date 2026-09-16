"""Validate the impedance controller in isolation (no insertion).

- `surface_press`: descend onto the flat socket top and press. A compliant arm should settle at
  contact force ≈ K·(setpoint depth below the surface), push straight along the normal, and sit
  still. Compared against joint PD pressing to a similar force.
- `external_push`: push the peg tip sideways with a known force in free space. Displacement
  should be ≈ F/K and recover after release.
- `tilted_press`: command a tilted peg and press it onto the surface. With low rotational
  stiffness the contact torque should flatten the peg against the surface.

Starts are teleported (IK + zero velocity) so each scenario isolates the contact response.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.config import ImpedanceConfig, JointPDConfig, SceneConfig, SensorConfig
from insertion.control.ik import JacobianIK, orientation_error
from insertion.control.impedance import CartesianImpedance
from insertion.control.joint_pd import joint_pd_torque
from insertion.control.scripted import pointing_down
from insertion.scene import build_model, hole_width
from insertion.sensors import ContactDetector, FTSensor
from insertion.sim import Sim


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def flat_top_point(scene: SceneConfig) -> np.ndarray:
    """A point on the socket top where the whole peg face rests on flat surface."""
    return np.array(scene.socket_pos) + [hole_width(scene) / 2 + scene.socket_chamfer + scene.peg_radius + 0.002, 0, scene.socket_depth]


def teleport(sim: Sim, pos: np.ndarray, rot: np.ndarray) -> None:
    ik = JacobianIK(sim.model)
    q, ok = ik.solve(sim.q(), pos, rot)
    assert ok, f"IK failed for {pos}"
    sim.data.qpos[sim.qpos_adr] = q
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def tare(sim: Sim, ft: FTSensor, hold) -> None:
    samples = []
    for _ in range(ft.cfg.tare_steps):
        sim.step(hold())
        samples.append(sim.ft_raw())
    ft.tare(samples)


@dataclass
class PressResult:
    controller: str
    setpoint_depth: float  # [m] below the surface (impedance) / joint target depth (PD)
    expected_force: float  # [N] K·depth, saturated (impedance only; nan for PD)
    mean_force: float  # [N] steady normal contact force
    lateral_ratio: float  # |F_xy| / |F_z| at steady state
    peak_force: float  # [N] single-step, whole run
    peak_force_10ms: float
    max_tip_speed: float  # [m/s] over the hold window
    ft_detect_latency: float  # [s] F/T contact detection after the 10 ms-average true contact force reaches the detection threshold


def surface_press(controller: str, depth: float, scene: SceneConfig | None = None, imp_cfg: ImpedanceConfig | None = None,
                  start_height: float = 0.005, speed: float = 0.01, hold: float = 1.5, window: float = 0.5) -> PressResult:
    scene = scene or SceneConfig()
    imp_cfg = imp_cfg or ImpedanceConfig()
    sim = Sim(build_model(scene))
    sim.reset()
    rot = pointing_down()
    surface = flat_top_point(scene)
    start = surface + [0, 0, start_height]
    teleport(sim, start, rot)
    dt = sim.model.opt.timestep

    imp = CartesianImpedance(sim.model, imp_cfg)
    imp.set_damping_at(sim.q())
    ik = JacobianIK(sim.model)
    pd = JointPDConfig()
    q_start = sim.q()

    def setpoint(t):
        return start - [0, 0, min(speed * t, start_height + depth)]

    def command(t):
        if controller == "impedance":
            return imp.torque(sim.q(), sim.qd(), setpoint(t), rot, np.array([0, 0, -speed]) if speed * t < start_height + depth else None)[0]
        q_des, _ = ik.solve(q_start, setpoint(t), rot, iters=20)
        return joint_pd_torque(sim, q_des, pd)

    ft = FTSensor(sim, SensorConfig())
    tare(sim, ft, lambda: command(0.0))
    detector = ContactDetector(SensorConfig())
    t0 = sim.time
    travel = (start_height + depth) / speed
    n = int((travel + hold) / dt)
    window_buf = np.zeros(round(0.010 / dt))
    peak = peak10 = 0.0
    first_contact = first_detect = None
    forces, speeds = [], []
    for i in range(n):
        t = sim.time - t0
        sim.step(command(t))
        ft.update()
        contacts = sim.peg_contacts()
        f = float(np.linalg.norm(contacts.force))
        peak = max(peak, f)
        window_buf[i % len(window_buf)] = f
        peak10 = max(peak10, float(window_buf.mean()))
        if first_contact is None and window_buf.mean() >= detector.cfg.contact_on:
            first_contact = t
        if first_detect is None and detector.update(ft.wrench):
            first_detect = t
        if i >= n - int(window / dt):
            forces.append(contacts.force.copy())
            jac = np.zeros((3, sim.model.nv))
            mujoco.mj_jacSite(sim.model, sim.data, jac, None, sim.model.site("peg_tip").id)
            speeds.append(np.linalg.norm(jac @ sim.data.qvel))
    forces = np.array(forces)
    fz = float(forces[:, 2].mean())
    expected = min(imp_cfg.stiffness_pos[2] * depth, imp_cfg.max_spring_force) if controller == "impedance" else float("nan")
    return PressResult(
        controller=controller,
        setpoint_depth=depth,
        expected_force=expected,
        mean_force=fz,
        lateral_ratio=float(np.linalg.norm(forces[:, :2].mean(axis=0)) / abs(fz)) if fz else float("inf"),
        peak_force=peak,
        peak_force_10ms=peak10,
        max_tip_speed=float(max(speeds)),
        ft_detect_latency=(first_detect - first_contact) if first_contact is not None and first_detect is not None else float("nan"),
    )


@dataclass
class PushResult:
    controller: str
    push_force: float  # [N] applied at the peg tip along +x
    displacement: float  # [m] steady displacement along x while pushed
    expected_displacement: float  # [m] F/K (impedance) — nan for PD
    residual_after_release: float  # [m] distance from the original pose 1 s after release
    peak_rotation_deg: float  # peg tilt induced while pushed


def external_push(controller: str, push: float = 3.0, imp_cfg: ImpedanceConfig | None = None,
                  push_time: float = 1.5, release_time: float = 1.5) -> PushResult:
    imp_cfg = imp_cfg or ImpedanceConfig()
    scene = SceneConfig()
    sim = Sim(build_model(scene))
    sim.reset()
    rot = pointing_down()
    hold_pos = flat_top_point(scene) + [0, 0, 0.05]
    teleport(sim, hold_pos, rot)
    imp = CartesianImpedance(sim.model, imp_cfg)
    imp.set_damping_at(sim.q())
    q_hold = sim.q()
    pd = JointPDConfig()
    dt = sim.model.opt.timestep
    peg_body = sim.model.body("peg").id

    displacement = []
    peak_rot = 0.0
    for i in range(int((push_time + release_time) / dt)):
        pushing = i * dt < push_time
        sim.data.qfrc_applied[:] = 0
        if pushing:
            tip, _ = sim.site_pose("peg_tip")
            mujoco.mj_applyFT(sim.model, sim.data, np.array([push, 0, 0]), np.zeros(3), tip, peg_body, sim.data.qfrc_applied)
        tau = imp.torque(sim.q(), sim.qd(), hold_pos, rot)[0] if controller == "impedance" else joint_pd_torque(sim, q_hold, pd)
        sim.step(tau)
        tip, tip_rot = sim.site_pose("peg_tip")
        if pushing:
            if i * dt > push_time - 0.3:
                displacement.append(tip[0] - hold_pos[0])
            peak_rot = max(peak_rot, float(np.degrees(np.linalg.norm(orientation_error(tip_rot, rot)))))
    return PushResult(
        controller=controller,
        push_force=push,
        displacement=float(np.mean(displacement)),
        expected_displacement=push / imp_cfg.stiffness_pos[0] if controller == "impedance" else float("nan"),
        residual_after_release=float(np.linalg.norm(tip - hold_pos)),
        peak_rotation_deg=peak_rot,
    )


@dataclass
class TiltResult:
    stiffness_rot: float  # [Nm/rad]
    commanded_tilt_deg: float
    final_tilt_deg: float  # peg axis angle from vertical while pressed
    mean_force: float  # [N]


def tilted_press(stiffness_rot: float, tilt_deg: float = 4.0, depth: float = 0.015, hold: float = 2.0) -> TiltResult:
    scene = SceneConfig()
    cfg = ImpedanceConfig(stiffness_rot=(stiffness_rot,) * 3)
    sim = Sim(build_model(scene))
    sim.reset()
    rot = rot_x(np.radians(tilt_deg)) @ pointing_down()
    surface = flat_top_point(scene)
    start = surface + [0, 0, 0.005]
    teleport(sim, start, rot)
    imp = CartesianImpedance(sim.model, cfg)
    imp.set_damping_at(sim.q())
    dt = sim.model.opt.timestep
    speed = 0.01
    travel = (0.005 + depth) / speed
    forces = []
    for i in range(int((travel + hold) / dt)):
        t = i * dt
        sp = start - [0, 0, min(speed * t, 0.005 + depth)]
        sim.step(imp.torque(sim.q(), sim.qd(), sp, rot)[0])
        if t > travel + hold - 0.5:
            forces.append(sim.peg_contacts().force[2])
    _, tip_rot = sim.site_pose("peg_tip")
    axis = tip_rot[:, 2]
    return TiltResult(
        stiffness_rot=stiffness_rot,
        commanded_tilt_deg=tilt_deg,
        final_tilt_deg=float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))),
        mean_force=float(np.mean(forces)),
    )
