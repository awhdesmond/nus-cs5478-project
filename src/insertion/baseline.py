"""Position-controlled baseline: the same insertion without compliance or search.

A conventional industrial arm is a stiff position servo: it goes where it's told and pushes back hard on
anything in the way. Modelled here as the same impedance controller made stiff in every direction, driving
the peg's tip straight down from the hover point to the perceived floor. No contact detection, no spiral:
if the perceived port is off, the peg either rides the chamfer in (small errors) or jams until the force cap stops it.
"""

from dataclasses import dataclass

import numpy as np

from grasp.pick import InHand
from insertion.compliant import InsertResult
from motion.arm import Arm
from motion.impedance import CartesianImpedance, ImpedanceConfig
from motion.wrist import WristFT
from perception.estimate import SceneEstimate


@dataclass(frozen=True)
class BaselineConfig:
    impedance: ImpedanceConfig = ImpedanceConfig(stiffness=(20000.0, 20000.0, 20000.0), rotational_stiffness=200.0)
    speed: float = 0.01  # [m/s] straight down, same speed as the compliant insert
    port_depth: float = 0.020  # [m]
    seated_tolerance: float = 0.0015  # [m]
    stall_time: float = 1.0  # [s]
    timeout: float = 60.0  # [s]
    force_cap: float = 30.0  # [N] sustained (10 ms average): a real cell's safety limit stops the arm
    sustained_window: float = 0.010  # [s]
    lift: float = 0.05  # [m]


def insert_stiff(arm: Arm, scene: SceneEstimate, in_hand: InHand, cfg: BaselineConfig = BaselineConfig()) -> InsertResult:
    sim, dt = arm.sim, arm.sim.model.opt.timestep
    arm.hold(0.2)
    ft = WristFT(sim, arm.kin)
    ft.tare()
    servo = CartesianImpedance(sim, arm.kin, in_hand.tip, cfg.impedance)
    tip, R = servo.pose()
    target = tip.copy()
    floor = scene.ports[in_hand.kind].entrance[2] - cfg.port_depth
    result = InsertResult(False, "insert")
    window = np.zeros((round(cfg.sustained_window / dt), 3))
    best_z, t_best = np.inf, 0.0

    for step in range(round(cfg.timeout / dt)):
        t = step * dt
        tip, _ = servo.pose()
        force = ft.update()
        window[step % len(window)] = ft.raw
        result.peak_force = max(result.peak_force, float(np.linalg.norm(ft.raw)))
        result.sustained_force = max(result.sustained_force, float(np.linalg.norm(window.mean(axis=0))))
        if step % 10 == 0:
            result.trace.append((t, "insert", *tip, *force))
        if result.sustained_force > cfg.force_cap:
            result.stage = "force"  # jammed hard enough to trip the safety limit
            break
        if tip[2] < floor + cfg.seated_tolerance:
            result.stage = "release"
            break
        if tip[2] < best_z - 1e-4:
            best_z, t_best = tip[2], t
        if t - t_best > cfg.stall_time:
            break  # stuck without (yet) reaching the force cap
        target[2] = max(target[2] - cfg.speed * dt, floor)  # straight down, xy never changes
        tau, _ = servo.torque(target, R)
        sim.step(tau)

    result.duration = t
    if result.stage != "release":
        return result
    arm.grip(False)
    p, R = arm.tcp()
    arm.move_line(p + [0, 0, cfg.lift], R)
    result.ok = True
    return result
