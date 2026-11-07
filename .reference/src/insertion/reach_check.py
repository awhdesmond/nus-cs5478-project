"""Pre-insertion reach from varied start configurations.

For each seeded start (home + joint noise, collision-free), solve IK to the pre-insertion
pose above the true socket entrance, execute the quintic joint move under joint PD +
gravity comp on the full dynamics, settle, and measure the final tip error and any
collisions along the way.
"""

from dataclasses import dataclass

import numpy as np

from insertion.config import JointPDConfig, MotionConfig, SceneConfig, StartConfig, TrialConfig
from insertion.control.ik import JacobianIK, orientation_error
from insertion.control.joint_pd import joint_pd_torque
from insertion.control.motion import SegmentRunner
from insertion.control.scripted import plan_approach, pointing_down
from insertion.harness import perturbed_start
from insertion.scene import build_model
from insertion.sim import Sim

MAX_POS_ERROR = 0.0005  # [m]
MAX_ROT_ERROR_DEG = 0.5


@dataclass
class ReachResult:
    seed: int
    start_joint_change: float  # max |q_pre - q_start| [rad]
    ik_converged: bool
    move_time: float  # [s]
    pos_error: float  # [m] after settling
    rot_error_deg: float
    collided: bool  # any contact during the move

    @property
    def passed(self) -> bool:
        return self.ik_converged and not self.collided and self.pos_error < MAX_POS_ERROR and self.rot_error_deg < MAX_ROT_ERROR_DEG


def reach_from_start(seed: int, joint_noise: float, scene: SceneConfig | None = None, settle: float = 0.5) -> ReachResult:
    scene = scene or SceneConfig()
    motion = MotionConfig()
    sim = Sim(build_model(scene))
    sim.reset()
    perturbed_start(sim, TrialConfig(seed=seed, scene=scene, start=StartConfig(joint_noise=joint_noise)))
    q_start = sim.q()
    ik = JacobianIK(sim.model)
    rot = pointing_down()
    entrance, _ = sim.site_pose("socket_entrance")
    pre = entrance + [0, 0, motion.approach_height]
    try:
        q_pre, segments = plan_approach(ik, q_start, pre, rot, motion, seed)
    except RuntimeError:
        return ReachResult(seed, float("nan"), False, float("nan"), float("nan"), float("nan"), False)

    runner = SegmentRunner(sim.model, ik, JointPDConfig())
    runner.start(segments, now=0.0)
    move_time = sum(seg.duration for seg in segments)
    dt = sim.model.opt.timestep
    collided = False
    for i in range(int((move_time + settle) / dt)):
        q_des, qd_des = runner.target(i * dt)
        sim.step(joint_pd_torque(sim, q_des, JointPDConfig(), qd_des))
        collided |= sim.data.ncon > 0
    tip, tip_rot = sim.site_pose("peg_tip")
    return ReachResult(
        seed=seed,
        start_joint_change=float(np.abs(q_pre - q_start).max()),
        ik_converged=True,
        move_time=move_time,
        pos_error=float(np.linalg.norm(tip - pre)),
        rot_error_deg=float(np.degrees(np.linalg.norm(orientation_error(tip_rot, rot)))),
        collided=bool(collided),
    )
