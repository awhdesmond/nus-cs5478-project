"""Compliant insertion: impedance control at the peg's tip, force-triggered contact,
spiral search under light force, depth-drop transition, then a force-capped push to the port's floor.

Starts with the peg held, its tip hovering over the perceived entrance (grasp/pick.py: carry_to_port).

    descend ──contact──▶ press ──no drop──▶ spiral ◀──no drop── centre
       │                   │                  │ └──caught by chamfer──▶ │
       │                   │                  drop                      drop
       └────── drop ───────┴──────────────────┴───────▶ insert ◀────────┘ ──floor──▶ release

Everything it senses is the arm's own: joint encoders (through the in-hand estimate, for the tip) and the
wrist force sensor. The peg's true position is never read.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from grasp.pick import InHand
from motion.arm import Arm
from motion.impedance import CartesianImpedance, ImpedanceConfig
from motion.wrist import WristFT
from perception.estimate import SceneEstimate


@dataclass(frozen=True)
class InsertConfig:
    impedance: ImpedanceConfig = ImpedanceConfig()  # soft sideways (300 N/m): lets the chamfer centre the peg
    # While searching, friction under the pressing force drags the tip behind its setpoint by μF/K; stiffer
    # sideways keeps that lag well under the spiral's pitch.
    search_stiffness: float = 1500.0  # [N/m] x and y, during the spiral only
    approach_speed: float = 0.01  # [m/s] setpoint descent before contact
    contact_force: float = 2.0  # [N] upward force that means the tip has touched something
    press_force: float = 4.0  # [N] light downward force held while waiting and searching
    press_wait: float = 0.3  # [s] pressing at the first contact point before starting to search
    drop: float = 0.003  # [m] tip this far below where it first touched = into the hole, past the chamfer
    # Tip this far below the first touch while searching = the chamfer has caught its edge: stop the spiral
    # (which would drag it back out) and let the chamfer's slope push it to the centre, softly.
    engage: float = 0.0008  # [m]
    centre_wait: float = 0.3  # [s] centring time before resuming the spiral if it hasn't dropped in
    max_lateral_force: float = 8.0  # [N] the spiral's spring never pulls the tip sideways harder than this
    spiral_pitch: float = 0.001  # [m] the spiral's radius grows by this per turn: below the ~2.5 mm capture radius
    spiral_speed: float = 0.005  # [m/s] setpoint speed along the spiral
    spiral_max_radius: float = 0.006  # [m] give up searching beyond this
    insert_force: float = 15.0  # [N] downward force pushing to the floor (half the 30 N cap)
    insert_speed: float = 0.01  # [m/s]
    port_depth: float = 0.020  # [m] nominal, from the drawing
    seated_tolerance: float = 0.0015  # [m] tip this close to the expected floor = seated
    stall_time: float = 1.0  # [s] no downward progress for this long while inserting = jammed
    timeout: float = 60.0  # [s] proposal: each insertion within 60 s
    force_cap: float = 30.0  # [N] proposal: sustained contact force (10 ms average) must stay under this
    sustained_window: float = 0.010  # [s]
    lift: float = 0.05  # [m] tool lift after releasing


@dataclass
class InsertResult:
    ok: bool
    stage: str  # where it ended: "descend", "press", "spiral", "centre", "insert", "force" (over the cap),
    # "release" (ok = released seated)
    duration: float = 0.0  # [s] from the start of the descent to release
    searched: bool = False  # did it need the spiral?
    search_radius: float = 0.0  # [m] how far out the spiral got
    sustained_force: float = 0.0  # [N] largest 10 ms average contact force magnitude (the proposal's measure)
    peak_force: float = 0.0  # [N] largest single-step reading (impacts, stick-slip: reported, not capped)
    trace: list = field(default_factory=list)  # (t, phase, tip estimate xyz, force xyz), every 10 ms


def spiral_offset(t: float, cfg: InsertConfig) -> np.ndarray:
    """Setpoint offset t seconds into an Archimedean spiral, r = pitch · θ / 2π, at (nearly) constant speed.

    Going round at speed v means r·dθ/dt = v. With r = bθ (b = pitch / 2π) that integrates to θ = √(2vt / b),
    so r = √(2bvt). This ignores the outward part of the motion (dr/dt), which adds ~8% to the speed in the
    first turn and under 0.3% after two.
    """
    b = cfg.spiral_pitch / (2 * math.pi)
    theta = math.sqrt(2 * cfg.spiral_speed * t / b)
    return b * theta * np.array([math.cos(theta), math.sin(theta), 0.0])


def insert(arm: Arm, scene: SceneEstimate, in_hand: InHand, cfg: InsertConfig = InsertConfig()) -> InsertResult:
    sim, dt = arm.sim, arm.sim.model.opt.timestep
    arm.hold(0.2)  # settle, so the tare below sees a still arm
    ft = WristFT(sim, arm.kin)
    ft.tare()  # zero out the weight of the gripper and peg
    impedance = CartesianImpedance(sim, arm.kin, in_hand.tip, cfg.impedance)
    soft = impedance.K.copy()
    stiff_search = soft.copy()
    stiff_search[0, 0] = stiff_search[1, 1] = cfg.search_stiffness
    kz = impedance.K[2, 2]

    tip, R = impedance.pose()
    target = tip.copy()  # the setpoint: moved by each phase, the spring does the rest
    entrance = scene.ports[in_hand.kind].entrance
    floor = entrance[2] - cfg.port_depth
    centre = target[:2].copy()
    phase, t_phase, contact_z = "descend", 0.0, None
    t_spiral = 0.0  # time spent spiralling: kept across centring attempts, so the spiral resumes where it was
    best_z, t_best = math.inf, 0.0
    result = InsertResult(False, "descend")
    window = np.zeros((round(cfg.sustained_window / dt), 3))  # the last 10 ms of raw readings, a ring buffer

    def switch(new):
        nonlocal phase, t_phase
        phase, t_phase = new, 0.0

    for step in range(round(cfg.timeout / dt)):
        t = step * dt
        tip, _ = impedance.pose()
        force = ft.update()  # filtered: for detecting contact
        window[step % len(window)] = ft.raw
        result.peak_force = max(result.peak_force, float(np.linalg.norm(ft.raw)))
        result.sustained_force = max(result.sustained_force, float(np.linalg.norm(window.mean(axis=0))))
        if result.sustained_force > cfg.force_cap:
            result.stage = "force"  # stop pushing: a real arm would be damaging something
            break
        if step % 10 == 0:
            result.trace.append((t, phase, *tip, *force))

        if phase == "descend":
            target[2] -= cfg.approach_speed * dt
            if force[2] > cfg.contact_force:
                contact_z = tip[2]
                switch("press")
            elif tip[2] < entrance[2] - cfg.drop:  # slid straight in without a noticeable touch
                contact_z = entrance[2]
                switch("insert")
        elif phase == "press":
            # Hold a light push where it touched, with no sideways pull (the setpoint follows the tip): on the
            # chamfer, the slope slides it in; on the block's flat top, friction holds it still.
            target[:2] = tip[:2]
            target[2] = tip[2] - cfg.press_force / kz
            if tip[2] < contact_z - cfg.drop:
                switch("insert")
            elif t_phase > cfg.press_wait:
                result.searched = True
                impedance.K = stiff_search
                switch("spiral")
        elif phase == "spiral":
            t_spiral += dt
            offset = spiral_offset(t_spiral, cfg)
            result.search_radius = float(np.linalg.norm(offset))
            # Pull toward the spiral point, but never harder than max_lateral_force: if friction holds the tip
            # back, the setpoint waits for it instead of stretching the spring.
            pull = centre + offset[:2] - tip[:2]
            reach = cfg.max_lateral_force / cfg.search_stiffness
            target[:2] = tip[:2] + pull * min(1.0, reach / max(np.linalg.norm(pull), 1e-12))
            target[2] = tip[2] - cfg.press_force / kz
            if tip[2] < contact_z - cfg.engage:
                impedance.K = soft
                switch("centre")
            elif result.search_radius > cfg.spiral_max_radius:
                result.stage = "spiral"
                break
        elif phase == "centre":
            # Force-guided centring: no sideways spring, so the only sideways push is the chamfer's, toward
            # the hole's axis. Keep pressing lightly.
            target[:2] = tip[:2]
            target[2] = tip[2] - cfg.press_force / kz
            if tip[2] < contact_z - cfg.drop:
                switch("insert")
            elif t_phase > cfg.centre_wait:
                impedance.K = stiff_search  # not a hole after all (or not yet): carry on searching
                switch("spiral")
        elif phase == "insert":
            # Setpoint descends, but never more than insert_force / kz below the tip: the spring's push is
            # capped at insert_force however hard the peg sticks.
            target[2] = max(target[2] - cfg.insert_speed * dt, tip[2] - cfg.insert_force / kz)
            target[:2] = tip[:2]  # follow the hole sideways rather than drag the peg against a wall
            if tip[2] < best_z - 1e-4:
                best_z, t_best = tip[2], t
            if tip[2] < floor + cfg.seated_tolerance:
                result.stage = "release"
                break
            if t - t_best > cfg.stall_time:
                result.stage = "insert"  # jammed
                break

        tau, _ = impedance.torque(target, R)
        sim.step(tau)
        t_phase += dt
    else:
        result.stage = phase  # timed out in this phase

    result.duration = t
    if result.stage != "release":
        return result
    # Seated (as far as the arm can tell): let go and back off.
    arm.grip(False)
    p, R = arm.tcp()
    arm.move_line(p + [0, 0, cfg.lift], R)
    result.ok = True
    return result
