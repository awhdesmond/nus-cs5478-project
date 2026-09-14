"""Contact sanity check: drive the peg into deliberate contact and measure solver health.

Scenarios press the peg on the socket top, the chamfer lip, the hole bottom, and a hole
wall, then hold. Over the hold window we measure penetration, force jitter, residual
motion and MuJoCo warnings. The arm is position-controlled (joint PD toward a target
inside the surface), so force scales with how far the target is past the surface.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.config import JointPDConfig, SceneConfig
from insertion.control.jacobian_step import JacobianIK
from insertion.control.joint_pd import joint_pd_torque
from insertion.scene import build_model, hole_width
from insertion.sim import Sim

# Thresholds a scene must meet before it's trusted at the tightest clearance.
MAX_PENETRATION = 0.0005  # [m] per-side gap at 1 mm clearance
MAX_FORCE_JITTER = 0.05  # std / mean of contact force magnitude over the hold window
MAX_TIP_SPEED = 1e-3  # [m/s] residual motion while holding (no chatter / creep)


@dataclass
class ContactCheck:
    scenario: str
    timestep: float
    mean_force: float  # [N]
    force_jitter: float  # std / mean
    max_penetration: float  # [m], whole run
    max_tip_speed: float  # [m/s] over the hold window
    warnings: int
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.warnings == 0
            and self.mean_force > 0.5
            and self.max_penetration < MAX_PENETRATION
            and self.force_jitter < MAX_FORCE_JITTER
            and self.max_tip_speed < MAX_TIP_SPEED
        )


def press(
    scene: SceneConfig,
    scenario: str,
    start: np.ndarray,
    goal: np.ndarray,
    move_time: float = 2.0,
    travel_time: float = 2.0,
    hold_time: float = 1.5,
    window: float = 0.5,
) -> ContactCheck:
    """Joint-move above `start`, follow a straight line to `goal` (inside a surface), hold."""
    sim = Sim(build_model(scene))
    sim.reset()
    pd = JointPDConfig()
    ik = JacobianIK(sim.model, "peg_tip")
    q0 = sim.q()
    _, rot_down = ik.fk(q0)
    q_start, ok = ik.solve(q0, start, rot_down)
    assert ok, f"IK failed for {scenario} start {start}"
    dt = sim.model.opt.timestep

    q_des = q_start
    forces, speeds = [], []
    max_pen = 0.0
    n_move, n_travel, n_hold = (int(t / dt) for t in (move_time, travel_time, hold_time))
    n_window = int(window / dt)
    for i in range(n_move + n_travel + n_hold):
        if i < n_move:
            s = i / n_move
            q_des = q0 + (3 * s**2 - 2 * s**3) * (q_start - q0)
        elif i < n_move + n_travel:
            s = (i - n_move) / n_travel
            q_des, _ = ik.solve(q_des, start + s * (goal - start), rot_down, iters=3)
        sim.step(joint_pd_torque(sim, q_des, pd))

        contacts = sim.peg_contacts()
        max_pen = max(max_pen, contacts.max_penetration)
        if i >= n_move + n_travel + n_hold - n_window:
            forces.append(np.linalg.norm(contacts.force))
            jac = np.zeros((3, sim.model.nv))
            mujoco.mj_jacSite(sim.model, sim.data, jac, None, sim.model.site("peg_tip").id)
            speeds.append(np.linalg.norm(jac @ sim.data.qvel))

    forces = np.array(forces)
    mean_force = float(forces.mean())
    return ContactCheck(
        scenario=scenario,
        timestep=dt,
        mean_force=mean_force,
        force_jitter=float(forces.std() / mean_force) if mean_force > 0 else float("inf"),
        max_penetration=max_pen,
        max_tip_speed=float(max(speeds)),
        warnings=int(sum(w.number for w in sim.data.warning)),
        finite=bool(np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))),
    )


def scenarios(scene: SceneConfig) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, start, goal) in world frame. Goals sit inside surfaces so the PD presses."""
    bottom = np.array(scene.socket_pos)
    top = bottom + [0, 0, scene.socket_depth]
    h = hole_width(scene) / 2
    r = scene.peg_radius
    up = np.array([0, 0, 0.03])
    on_top = top + [h + scene.socket_chamfer + r + 0.002, 0, 0]  # peg fully on the flat top
    lip = top + [h + r * 0.5, 0, 0]  # peg edge over the chamfer
    wall_start = bottom + [0, 0, scene.socket_depth / 2]
    out = [
        (f"top surface, target {d * 1e3:.0f} mm deep", on_top + up, on_top - [0, 0, d])
        for d in (0.005, 0.010, 0.025)
    ]
    out += [
        ("chamfer lip, target 10 mm deep", lip + up, lip - [0, 0, 0.010]),
        ("hole bottom, target 10 mm deep", top + up, bottom - [0, 0, 0.010]),
        ("hole wall, target 5 mm into wall", wall_start, wall_start + [h - r + 0.005, 0, 0]),
    ]
    return out


def run_all(clearance: float = 0.001, timesteps=(0.001, 0.0005)) -> list[ContactCheck]:
    results = []
    for dt in timesteps:
        scene = SceneConfig(clearance=clearance, timestep=dt)
        for name, start, goal in scenarios(scene):
            results.append(press(scene, name, start, goal))
    return results
