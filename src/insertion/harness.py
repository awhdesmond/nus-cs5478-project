"""Evaluation harness: run N trials per condition and log one row per trial.

Kept policy-agnostic: a controller is anything implementing `Controller` (joint torques
from observations), so a learned policy can later drop in behind the same interface.

Information boundary: controllers get the model (for kinematics), proprioception, the
tared/filtered wrist F/T, and a *believed* socket entrance. Ground truth (true socket pose,
contact forces, penetration) is only used here, for scoring.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import mujoco
import numpy as np
import pandas as pd

from insertion.config import JointPDConfig, SensorConfig, TrialConfig
from insertion.control.joint_pd import joint_pd_torque
from insertion.scene import UR5E_JOINTS, build_model, hole_width
from insertion.sensors import FTSensor
from insertion.sim import Sim


@dataclass
class InsertionTarget:
    believed_entrance: np.ndarray  # hole-top centre as the controller believes it, world [m]
    depth: float  # nominal insertion depth from entrance to hole bottom [m]


@dataclass
class Observation:
    time: float
    q: np.ndarray
    qd: np.ndarray
    ft: np.ndarray  # tared + filtered external wrench on the peg, sensor frame
    tip_pos: np.ndarray  # peg tip from forward kinematics (proprioceptive)
    tip_rot: np.ndarray


class Controller(Protocol):
    def reset(self, model: mujoco.MjModel, q0: np.ndarray, target: InsertionTarget, seed: int) -> None: ...

    def act(self, obs: Observation) -> np.ndarray:
        """Joint torques [Nm]."""
        ...

    # Optional: `info() -> dict` of controller-specific stats (e.g. retry count), logged per trial.


@dataclass
class TrialResult:
    seed: int
    success: bool  # seated without exceeding the force cap
    seated: bool
    reached_contact: bool  # first peg contact within contact_step_budget
    contact_time: float  # [s sim time], nan if never
    cycle_time: float  # [s sim time] to seated, or elapsed time if not seated
    peak_force: float  # [N] ground-truth contact force magnitude, single-step (includes ~1 ms impact spikes)
    peak_force_10ms: float  # [N] peak of the 10 ms moving average (sustained force)
    peak_ft_force: float  # [N] filtered F/T force magnitude (what the controller saw)
    max_penetration: float  # [m]
    final_depth: float  # [m] tip depth below the true entrance
    failure_reason: str = ""
    controller_info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Per-step log, kept LeRobot-convertible: (observation, action) per timestep."""

    time: list = field(default_factory=list)
    q: list = field(default_factory=list)
    qd: list = field(default_factory=list)
    ft: list = field(default_factory=list)
    tip_pos: list = field(default_factory=list)
    action: list = field(default_factory=list)
    contact_force: list = field(default_factory=list)
    penetration: list = field(default_factory=list)

    def save(self, path: Path, cfg: TrialConfig) -> None:
        """Arrays per field, plus the trial config as JSON so the scene can be rebuilt for replay."""
        arrays = {k: np.asarray(v) for k, v in asdict(self).items()}
        np.savez_compressed(path, config=json.dumps(cfg.to_dict()), **arrays)


def believed_entrance(cfg: TrialConfig, true_entrance: np.ndarray) -> np.ndarray:
    """True entrance plus a seeded xy offset, uniform in ±translation_max.

    TODO: rotation uncertainty (needs a rotatable socket).
    """
    rng = np.random.default_rng(cfg.seed)
    t = cfg.uncertainty.translation_max
    return true_entrance + np.array([*rng.uniform(-t, t, size=2), 0.0])


def perturbed_start(sim: Sim, cfg: TrialConfig, max_tries: int = 100) -> None:
    """Move the arm to home + seeded uniform joint noise, resampling until collision-free."""
    noise = cfg.start.joint_noise
    if noise <= 0:
        return
    rng = np.random.default_rng([cfg.seed, 1])  # separate stream from the socket-pose uncertainty
    q_home = sim.q()
    lower, upper = sim.model.jnt_range[[sim.model.joint(j).id for j in UR5E_JOINTS]].T
    for _ in range(max_tries):
        sim.data.qpos[sim.qpos_adr] = np.clip(q_home + rng.uniform(-noise, noise, size=6), lower, upper)
        mujoco.mj_forward(sim.model, sim.data)
        if sim.data.ncon == 0:
            return
    raise RuntimeError(f"no collision-free start found for seed {cfg.seed} with joint_noise {noise}")


def tare_while_holding(sim: Sim, ft: FTSensor) -> None:
    """Hold the current pose with joint PD + gravity comp and tare the F/T on the readings."""
    q_hold = sim.q()
    samples = []
    for _ in range(ft.cfg.tare_steps):
        sim.step(joint_pd_torque(sim, q_hold, JointPDConfig()))
        samples.append(sim.ft_raw())
    ft.tare(samples)


def observe(sim: Sim, ft: FTSensor, t_start: float) -> Observation:
    tip_pos, tip_rot = sim.site_pose("peg_tip")
    return Observation(sim.time - t_start, sim.q(), sim.qd(), ft.wrench.copy(), tip_pos, tip_rot)


def run_trial(
    cfg: TrialConfig,
    controller: Controller,
    sensor_cfg: SensorConfig | None = None,
    record: bool = False,
) -> tuple[TrialResult, Trajectory | None]:
    sensor_cfg = sensor_cfg or SensorConfig()
    sim = Sim(build_model(cfg.scene))
    sim.reset()
    perturbed_start(sim, cfg)
    ft = FTSensor(sim, sensor_cfg)
    dt = sim.model.opt.timestep

    tare_while_holding(sim, ft)
    t_start = sim.time

    bottom, _ = sim.site_pose("socket_bottom")
    entrance, _ = sim.site_pose("socket_entrance")
    half_width = hole_width(cfg.scene) / 2
    target = InsertionTarget(believed_entrance(cfg, entrance), cfg.scene.socket_depth)
    controller.reset(sim.model, sim.q(), target, cfg.seed)

    traj = Trajectory() if record else None
    peak_force = peak_ft = max_pen = 0.0
    window = np.zeros(max(1, round(0.010 / dt)))  # ring buffer for the 10 ms moving average
    peak_force_10ms = 0.0
    contact_step = None
    seated = False
    max_steps = int(cfg.task.cycle_time_budget / dt)

    for step in range(max_steps):
        obs = observe(sim, ft, t_start)
        tip_pos = obs.tip_pos
        action = controller.act(obs)
        sim.step(action)
        ft.update()

        contacts = sim.peg_contacts()
        force = float(np.linalg.norm(contacts.force))
        peak_force = max(peak_force, force)
        window[step % len(window)] = force
        peak_force_10ms = max(peak_force_10ms, float(window.mean()))
        peak_ft = max(peak_ft, float(np.linalg.norm(ft.wrench[:3])))
        max_pen = max(max_pen, contacts.max_penetration)
        if contact_step is None and contacts.count > 0:
            contact_step = step
        if traj is not None:
            for key, value in (
                ("time", obs.time), ("q", obs.q), ("qd", obs.qd), ("ft", obs.ft), ("tip_pos", tip_pos),
                ("action", action), ("contact_force", contacts.force), ("penetration", contacts.max_penetration),
            ):
                getattr(traj, key).append(value)

        tip_pos, _ = sim.site_pose("peg_tip")
        in_hole = np.all(np.abs(tip_pos[:2] - bottom[:2]) < half_width)
        if in_hole and tip_pos[2] - bottom[2] <= cfg.task.seated_depth_tol:
            seated = True
            break

    tip_pos, _ = sim.site_pose("peg_tip")
    reached_contact = contact_step is not None and contact_step <= cfg.task.contact_step_budget
    over_cap = peak_force > cfg.task.force_cap
    if seated and not over_cap:
        reason = ""
    elif seated:
        reason = "force_cap"
    elif not reached_contact:
        reason = "no_contact"
    else:
        reason = "timeout"
    result = TrialResult(
        seed=cfg.seed,
        success=seated and not over_cap,
        seated=seated,
        reached_contact=reached_contact,
        contact_time=contact_step * dt if contact_step is not None else float("nan"),
        cycle_time=sim.time - t_start,
        peak_force=peak_force,
        peak_force_10ms=peak_force_10ms,
        peak_ft_force=peak_ft,
        max_penetration=max_pen,
        final_depth=float(entrance[2] - tip_pos[2]),
        failure_reason=reason,
        controller_info=controller.info() if hasattr(controller, "info") else {},
    )
    return result, traj


def _run_one(args) -> tuple[dict, TrialConfig, Trajectory | None]:
    cfg, controller_factory, record = args
    result, traj = run_trial(cfg, controller_factory(), record=record)
    row = asdict(result)
    info = row.pop("controller_info")
    config_cols = pd.json_normalize(cfg.to_dict(), sep=".").iloc[0].to_dict()
    return {**row, **{f"info.{k}": v for k, v in info.items()}, **{f"cfg.{k}": v for k, v in config_cols.items()}}, cfg, traj


def run_trials(
    configs: list[TrialConfig],
    controller_factory: Callable[[], Controller],
    out_csv: Path,
    record_dir: Path | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Run each config with a fresh controller; write one CSV row per trial (result + info + flattened config).

    `workers > 1` runs trials in parallel processes (the factory must be picklable, e.g. a class).
    """
    jobs = [(cfg, controller_factory, record_dir is not None) for cfg in configs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_run_one, jobs))
    else:
        outputs = [_run_one(job) for job in jobs]

    rows = []
    for i, (row, cfg, traj) in enumerate(outputs):
        if traj is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            traj.save(record_dir / f"trial_{i:04d}_seed{cfg.seed}.npz", cfg)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
