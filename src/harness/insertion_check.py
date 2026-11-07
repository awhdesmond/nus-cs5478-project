"""Week-4 check (proposal §5): the insertion-module grid, compliant vs position-controlled.

Each trial: a seeded layout, perception, pick, carry; then the *perceived* port is shifted sideways by
`port_error` (a stand-in for calibration error, in a seeded direction), so the arm hovers that far off the
real hole; then one controller inserts. Ground truth only scores the outcome.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from grasp.pick import carry_to_port, pick
from insertion.baseline import insert_stiff
from insertion.compliant import insert
from motion.arm import Arm
from perception.estimate import perceive
from sim.camera import OverheadCamera
from sim.config import PEGS, SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim

CONTROLLERS = {"compliant": insert, "stiff": insert_stiff}


def trial(controller: str, kind: str, seed: int, port_error: float, clearance: float, friction: float,
          depth_noise: float = 0.002) -> dict:
    """One insertion trial; returns a row for the results table."""
    base = replace(SceneConfig(), clearance=clearance, friction=friction)
    cfg = replace(sample_layout(base, seed), depth_noise=depth_noise)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    with OverheadCamera(sim.model, cfg) as camera:
        scene = perceive(camera.capture(sim.data, np.random.default_rng([seed, 9])))
    angle = np.random.default_rng([seed, 12]).uniform(0, 2 * math.pi)  # same direction for both controllers
    scene.ports[kind].entrance = scene.ports[kind].entrance + port_error * np.array([math.cos(angle), math.sin(angle), 0])

    arm = Arm(sim)
    row = dict(controller=controller, kind=kind, seed=seed, port_error=port_error, clearance=clearance,
               friction=friction, success=False, stage="pick", duration=math.nan, searched=False,
               sustained_force=math.nan, peak_force=math.nan)
    picked = pick(arm, scene, kind)
    if not picked.ok:
        return row
    held = carry_to_port(arm, scene, picked)
    result = CONTROLLERS[controller](arm, scene, held.in_hand)
    # Success is judged on ground truth (proposal §1): seated in its own port, within the time and force caps.
    row.update(stage=result.stage, duration=result.duration, searched=result.searched,
               sustained_force=result.sustained_force, peak_force=result.peak_force,
               success=bool(sim.seated(kind) and sim.in_wrong_port(kind) is None
                            and result.duration <= 60 and result.sustained_force <= 30))
    return row


def insertion_grid(seeds, port_errors, clearances=(0.001, 0.003), frictions=(0.3, 0.8),
                   controllers=tuple(CONTROLLERS), progress=None) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(controllers, clearances, frictions, port_errors, PEGS, seeds))
    for i, (ctrl, cl, fr, err, kind, seed) in enumerate(combos):
        rows.append(trial(ctrl, kind, seed, err, cl, fr))
        if progress:
            progress(i + 1, len(combos))
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Success rate, median time and worst sustained force for each controller × condition."""
    table = results.groupby(["controller", "clearance", "friction", "port_error"]).agg(
        trials=("success", "size"), success=("success", "mean"),
        time_median=("duration", "median"), force_max=("sustained_force", "max"),
        searched=("searched", "mean"))
    table.index = table.index.set_levels([lvl * 1e3 if name in ("clearance", "port_error") else lvl
                                          for name, lvl in zip(table.index.names, table.index.levels)])
    table.index = table.index.rename({"clearance": "clearance_mm", "port_error": "error_mm"})
    return table.round(2)
