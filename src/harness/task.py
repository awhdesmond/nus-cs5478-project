"""One full-task trial (proposal §1): from the camera's picture to both pegs seated in their own ports.

    perceive once ─▶ for each peg: pick ─▶ carry to its port ─▶ insert ─▶ release
                                                                         ▶ success = both seated (ground truth)

The controller side sees only the camera observation and the arm's own sensing. The harness alone reads
ground truth: to score the outcome, log perception error, and name the first stage that went wrong
(proposal §4.2: perception / grasp / planning / insertion / release).
"""

import math

import numpy as np

from grasp.pick import carry_to_port, pick
from harness.insertion_check import CONTROLLERS
from harness.perception_check import body_yaw
from motion.arm import Arm
from perception.estimate import perceive
from perception.pegs import wrap_quarter_turn
from sim.camera import OverheadCamera
from sim.config import PEGS, SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim

STAGES = ("perception", "grasp", "planning", "insertion", "release")


def add_pose_error(scene, error: float, rng: np.random.Generator) -> None:
    """Shift every perceived object independently by `error` metres in a random direction (proposal §4.2's
    robustness condition: per-object calibration error, on top of camera noise). Used in step 7."""
    for estimate in (*scene.pegs.values(), *scene.ports.values()):
        angle = rng.uniform(0, 2 * math.pi)
        shift = error * np.array([math.cos(angle), math.sin(angle), 0.0])
        if hasattr(estimate, "entrance"):
            estimate.entrance = estimate.entrance + shift
        else:
            estimate.top = estimate.top + shift
            estimate.outline = estimate.outline + shift[:2]


def perception_errors(sim: Sim, scene) -> dict:
    """xy error of each estimate against ground truth [m], and the square objects' yaw errors [rad]."""
    errors = {}
    for kind in PEGS:
        errors[f"{kind}_peg_xy"] = float(np.linalg.norm(scene.pegs[kind].top[:2] - sim.data.site(f"{kind}_top").xpos[:2]))
        errors[f"{kind}_port_xy"] = float(np.linalg.norm(
            scene.ports[kind].entrance[:2] - sim.data.site(f"{kind}_port_entrance").xpos[:2]))
    errors["square_peg_yaw"] = abs(wrap_quarter_turn(scene.pegs["square"].yaw - body_yaw(sim.data, "square_peg")))
    errors["square_port_yaw"] = abs(wrap_quarter_turn(scene.ports["square"].yaw - body_yaw(sim.data, "square_port")))
    return errors


def run_task(base: SceneConfig, seed: int, controller: str, pose_error: float = 0.0) -> dict:
    """One trial; returns a flat row: outcome, failed stage, timing, forces, perception errors."""
    cfg = sample_layout(base, seed)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    row = dict(seed=seed, controller=controller, clearance=cfg.clearance, friction=cfg.friction,
               noise=cfg.depth_noise, pose_error=pose_error, success=False, failed_stage=None)

    with OverheadCamera(sim.model, cfg) as camera:
        obs = camera.capture(sim.data, np.random.default_rng([seed, 9]))
    try:
        scene = perceive(obs)
    except RuntimeError:
        row["failed_stage"] = "perception"
        return row
    if pose_error > 0:
        add_pose_error(scene, pose_error, np.random.default_rng([seed, 15]))
    row.update(perception_errors(sim, scene))  # after the injected error: it's part of what the controller gets

    arm, insert = Arm(sim), CONTROLLERS[controller]
    task_start = sim.time
    for kind in PEGS:
        # Ground truth, for blame only: did perception even pick out the right object?
        if row[f"{kind}_peg_xy"] > 0.005 or row[f"{kind}_port_xy"] > 0.005:
            stage_if_fails = "perception"  # off by > 5 mm: a later failure is perception's fault
        else:
            stage_if_fails = None
        start = sim.time
        try:
            picked = pick(arm, scene, kind)
        except ValueError:  # no grasp passed the planner, or no IK solution
            row["failed_stage"] = stage_if_fails or "planning"
            break
        if not picked.ok:
            row["failed_stage"] = stage_if_fails or "grasp"
            break
        try:
            held = carry_to_port(arm, scene, picked)
        except ValueError:
            row["failed_stage"] = stage_if_fails or "planning"
            break
        result = insert(arm, scene, held.in_hand)
        row[f"{kind}_time"] = sim.time - start
        row[f"{kind}_insert_time"] = result.duration
        row[f"{kind}_sustained_force"] = result.sustained_force
        row[f"{kind}_peak_force"] = result.peak_force
        row[f"{kind}_searched"] = result.searched
        within_caps = result.duration <= 60 and result.sustained_force <= 30
        if not (result.ok and within_caps):
            row["failed_stage"] = stage_if_fails or "insertion"
            break
        if not sim.seated(kind) or sim.in_wrong_port(kind) or sim.attached() == kind:
            row["failed_stage"] = "release"  # the arm thought it was done, but the peg isn't where it should be
            break
    else:
        # Both pegs placed. Check both again: inserting the second must not have disturbed the first.
        row["success"] = all(sim.seated(k) and sim.in_wrong_port(k) is None for k in PEGS)
        if not row["success"]:
            row["failed_stage"] = "release"
    row["task_time"] = sim.time - task_start
    return row
