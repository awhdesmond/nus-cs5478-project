"""the scene steps stably, and each peg physically fits only its own port.

- Rest: with the arm holding still, the pegs must not move, and nothing but peg–table contacts may exist.
- Keying: press each peg, upright and lined up with the opening (the best case for getting in), into each
  port with up to the proposal's 30 N force cap. Its own port must take it to the floor; the other port must
  stop it on the rim, unless the peg genuinely fits geometrically (the round and square pegs swap at 2–3 mm
  clearance, as the proposal notes). Contact penetration must stay under half the 1 mm clearance.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim.config import PEGS, SceneConfig
from sim.scene import build_model, build_spec, peg_half, port_hole_half
from sim.sim import Sim
from sim.utils import quat_z

KP = np.array([400, 400, 400, 100, 100, 50.0])
KD = np.array([40, 40, 40, 10, 10, 5.0])
MAX_DRIFT = 1e-4  # [m] a resting peg may move at most 0.1 mm
MAX_PENETRATION = 0.0005  # [m] half the per-side gap at 1 mm clearance
FORCE_CAP = 30.0  # [N] proposal §2
GUIDE_MASS = 2.0  # [kg] about the arm's effective mass at the tool
PRESS_SPEED = 0.005  # [m/s] slow, like a careful insertion


def arm_hold_torque(sim: Sim, q_hold: np.ndarray) -> np.ndarray:
    """Gravity compensation + PD toward q_hold. (Step 3 moves this into motion/.)"""
    return sim.gravity_comp() + KP * (q_hold - sim.q()) - KD * sim.qd()


@dataclass
class RestResult:
    max_drift: float  # [m] furthest any peg moved
    unexpected_contacts: list[tuple[str, str]]  # anything touching other than a peg on the table

    @property
    def passed(self) -> bool:
        return self.max_drift < MAX_DRIFT and not self.unexpected_contacts


def rest_check(cfg: SceneConfig, seconds: float = 2.0) -> RestResult:
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    q_ready = sim.q()
    start = {k: sim.data.body(f"{k}_peg").xpos.copy() for k in PEGS}
    unexpected = set()
    for _ in range(round(seconds / sim.model.opt.timestep)):
        sim.step(arm_hold_torque(sim, q_ready))
        for c in sim.data.contact[:sim.data.ncon]:
            pair = tuple(sorted((sim.model.geom(c.geom1).name, sim.model.geom(c.geom2).name)))
            if pair not in {("round_peg", "table"), ("square_peg", "table")}:
                unexpected.add(pair)
    drift = max(float(np.linalg.norm(sim.data.body(f"{k}_peg").xpos - start[k])) for k in PEGS)
    return RestResult(drift, sorted(unexpected))


def fits(cfg: SceneConfig, peg: str, port: str) -> bool:
    """Pure geometry: does the peg's cross-section fit through the port's hole in its best orientation?"""
    w, h = peg_half(cfg, peg), port_hole_half(cfg, port)
    if port == "square":
        return w < h  # a round or square peg, lined up with the hole's sides
    vertex = h / math.cos(math.pi / cfg.round_sides)  # the 16-gon's widest point (h is its apothem)
    return w < h if peg == "round" else w * math.sqrt(2) < vertex  # a square peg, corners to the vertices


@dataclass
class KeyingResult:
    peg: str
    port: str
    yaw_offset_deg: float  # how the peg was turned relative to the port
    depth: float  # [m] deepest the insertion end got below the port's entrance
    seated: bool  # reached the floor of its own port
    entered: bool  # past the chamfer into the hole
    max_penetration: float  # [m]
    fits: bool  # geometrically, in its best orientation

    @property
    def passed(self) -> bool:
        ok = self.seated if self.peg == self.port else (self.fits or not self.entered)
        return ok and self.max_penetration < MAX_PENETRATION


def guided_peg_model(cfg: SceneConfig, peg: str, pos, yaw: float, push: float) -> mujoco.MjModel:
    """The scene with one peg riding on x/y/z slides instead of a free joint, pressed down by a motor.

    Pushing a free 4 g peg with 30 N accelerates it at ~7,500 m/s², and it tunnels straight through 6 mm walls
    (seen in the previous attempt). A gripper would push it with the arm's mass behind it, so each slide gets
    ~2 kg of extra inertia (armature). x and y are free so the chamfer can centre the peg; it can't tilt or turn,
    which keeps it lined up with the opening: the best case for a wrong peg to get in.
    """
    spec = build_spec(cfg)
    for key in list(spec.keys):
        spec.delete(key)  # replacing a joint changes qpos, so the keyframes no longer fit; we set the arm below
    body = spec.body(f"{peg}_peg")
    body.pos, body.quat = list(pos), quat_z(yaw)
    spec.delete(spec.joint(f"{peg}_peg"))  # remove the free joint
    for axis_name, axis in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        body.add_joint(name=f"guide_{axis_name}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, armature=GUIDE_MASS)
    # Position actuator on z with its force capped at `push`: the setpoint runs ahead, so it presses with `push`.
    stiffness = push / 0.01  # reaches the cap 10 mm behind its setpoint
    spec.add_actuator(name="guide_push", target="guide_z", trntype=mujoco.mjtTrn.mjTRN_JOINT, forcelimited=True,
                      forcerange=[-push, push]).set_to_position(kp=stiffness, kv=2 * math.sqrt(stiffness * GUIDE_MASS))
    return spec.compile()


def press_into_port(cfg: SceneConfig, peg: str, port: str, yaw_offset: float = 0.0,
                    push: float = FORCE_CAP, settle: float = 0.5) -> KeyingResult:
    """Press `peg`, upright, centred 2 mm above `port`'s entrance, down with at most `push` newtons."""
    probe = Sim(build_model(cfg), cfg)  # a normal scene, just to read where the port is and the ready pose
    probe.reset("ready")
    entrance = probe.data.site(f"{port}_port_entrance").xpos.copy()
    port_yaw = cfg.port_poses[port].yaw

    model = guided_peg_model(cfg, peg, entrance + [0, 0, 0.002 + cfg.peg_length / 2], port_yaw + yaw_offset, push)
    sim = Sim(model, cfg)
    sim.data.qpos[sim.arm_qpos] = probe.q()  # arm at ready, out of the way
    mujoco.mj_forward(model, sim.data)
    q_ready = sim.q()
    guide = model.actuator("guide_push").id
    travel = 0.002 + cfg.port_depth + 0.010  # setpoint ends 10 mm below the floor, so it keeps pressing
    depth = penetration = 0.0
    dt = model.opt.timestep
    for i in range(round((travel / PRESS_SPEED + settle) / dt)):
        sim.data.ctrl[guide] = -min(travel, PRESS_SPEED * i * dt)
        sim.step(arm_hold_torque(sim, q_ready))
        depth = max(depth, float(entrance[2] - sim.data.site(f"{peg}_tip").xpos[2]))
        penetration = max(penetration, sim.max_penetration())
    entered = sim.end_in_port(peg, port) and depth > cfg.port_chamfer + 0.002
    return KeyingResult(peg, port, math.degrees(yaw_offset), depth, peg == port and sim.seated(peg), bool(entered),
                        penetration, fits(cfg, peg, port))


def keying_check(cfg: SceneConfig) -> list[KeyingResult]:
    """Every peg into every port. The square peg goes into the round hole twice: faces toward the polygon's
    faces, and corners toward its vertices (its widest orientation, the best chance of getting in)."""
    results = []
    for peg in PEGS:
        for port in PEGS:
            offsets = [0.0, math.pi / cfg.round_sides] if (peg, port) == ("square", "round") else [0.0]
            results.extend(press_into_port(cfg, peg, port, offset) for offset in offsets)
    return results
