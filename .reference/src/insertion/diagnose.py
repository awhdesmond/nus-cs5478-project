"""Failure-mode diagnostics for the spiral search controller.

Re-runs a trial with a logging subclass and summarises, using ground truth, where the peg centre
went during the search relative to the true hole, and what the controller was doing.
"""

from dataclasses import dataclass

import numpy as np

from insertion.config import TrialConfig
from insertion.control.compliant import SpiralSearchInsert
from insertion.harness import run_trial
from insertion.scene import hole_width


class LoggedSpiral(SpiralSearchInsert):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.log = []  # (time, phase, paused, stalled, tip xyz, lateral F/T force, theta)

    def act(self, obs):
        stall_before = self.stalls
        tau = super().act(obs)
        force_xy = float(np.linalg.norm((obs.tip_rot @ obs.ft[:3])[:2]))
        self.log.append((obs.time, self.phase, obs.time < self.paused_until, self.stalls > stall_before, *obs.tip_pos, force_xy, self.theta))
        return tau


@dataclass
class SpiralDiagnosis:
    seed: int
    friction: float
    clearance: float
    uncertainty: float
    success: bool
    failure_reason: str
    peak_force: float  # [N] single-step
    peak_force_10ms: float  # [N]
    time_inserting: float  # [s] in the insert phase
    offset: float  # [m] believed-to-true entrance distance
    final_phase: str
    search_time: float  # [s]
    min_centre_distance: float  # [m] closest the peg centre got to the hole axis while searching
    time_in_direct_zone: float  # [s] centre within the direct-entry square while searching
    time_in_chamfer_zone: float  # [s] centre within the chamfer-capture square (excl. direct zone)
    time_paused: float  # [s]
    time_stalled: float  # [s]
    time_sweeping: float  # [s]
    spiral_turns: float
    final_centre_distance: float  # [m]
    max_drop: float  # [m] deepest below contact height while searching
    mean_lateral_force_stalled: float  # [N]


def diagnose(cfg: TrialConfig) -> SpiralDiagnosis:
    ctrl = LoggedSpiral()
    result, _ = run_trial(cfg, ctrl)
    log = ctrl.log
    t = np.array([row[0] for row in log])
    phase = np.array([row[1] for row in log])
    paused = np.array([row[2] for row in log])
    stalled = np.array([row[3] for row in log])
    tip = np.array([row[4:7] for row in log])
    force_xy = np.array([row[7] for row in log])
    theta = np.array([row[8] for row in log])
    dt = np.diff(t, prepend=t[0])

    hole = np.array(cfg.scene.socket_pos[:2])
    h = hole_width(cfg.scene) / 2
    direct = h - cfg.scene.peg_radius
    chamfer = direct + cfg.scene.socket_chamfer
    offset_xy = np.abs(tip[:, :2] - hole)
    in_direct = np.all(offset_xy <= direct, axis=1)
    in_chamfer = np.all(offset_xy <= chamfer, axis=1) & ~in_direct
    search = phase == SpiralSearchInsert.SEARCH
    centre_dist = np.linalg.norm(tip[:, :2] - hole, axis=1)
    believed = ctrl.entrance

    return SpiralDiagnosis(
        seed=cfg.seed,
        friction=cfg.scene.friction,
        clearance=cfg.scene.clearance,
        uncertainty=cfg.uncertainty.translation_max,
        success=result.success,
        failure_reason=result.failure_reason,
        peak_force=result.peak_force,
        peak_force_10ms=result.peak_force_10ms,
        time_inserting=float(dt[phase == SpiralSearchInsert.INSERT].sum()),
        offset=float(np.linalg.norm(believed[:2] - hole)),
        final_phase=str(phase[-1]),
        search_time=float(dt[search].sum()),
        min_centre_distance=float(centre_dist[search].min()) if search.any() else float("nan"),
        time_in_direct_zone=float(dt[search & in_direct].sum()),
        time_in_chamfer_zone=float(dt[search & in_chamfer].sum()),
        time_paused=float(dt[search & paused].sum()),
        time_stalled=float(dt[search & stalled].sum()),
        time_sweeping=float(dt[search & ~paused & ~stalled].sum()),
        spiral_turns=float(theta.max() / (2 * np.pi)),
        final_centre_distance=float(centre_dist[-1]),
        max_drop=float((ctrl.contact_z - tip[search, 2]).max()) if search.any() else 0.0,
        mean_lateral_force_stalled=float(force_xy[search & stalled].mean()) if (search & stalled).any() else 0.0,
    )
