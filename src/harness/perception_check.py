"""perception error against ground truth, logged over seeded layouts.
"""

import math
from dataclasses import replace

import mujoco
import numpy as np
import pandas as pd

from perception.estimate import PerceptionConfig, perceive
from perception.pegs import wrap_quarter_turn
from sim.camera import OverheadCamera
from sim.config import PEGS, SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim


def body_yaw(data: mujoco.MjData, body: str) -> float:
    """A body's true yaw (rotation about world z), from its rotation matrix."""
    R = data.body(body).xmat.reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def check_layout(cfg: SceneConfig, seed: int, perception: PerceptionConfig = PerceptionConfig()) -> list[dict]:
    """Perceive one layout from the ready pose. One row per object (2 pegs, 2 ports) with its errors.

    xy_error: horizontal distance between estimated and true point (peg: top-face centre; port: entrance).
    z_error: estimated minus true height. yaw_error: square objects only, wrapped to ±45°.
    found = False (errors NaN) if perception failed outright.
    """
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    with OverheadCamera(sim.model, cfg) as camera:
        obs = camera.capture(sim.data, np.random.default_rng([seed, 9]))  # its own stream, like the layout's
    try:
        scene = perceive(obs, perception)
    except RuntimeError:
        scene = None

    rows = []
    for obj in ("peg", "port"):
        for kind in PEGS:
            # Ground truth: where the point that matters really is, and how the object is turned.
            site = f"{kind}_top" if obj == "peg" else f"{kind}_port_entrance"
            true_point = sim.data.site(site).xpos
            true_yaw = body_yaw(sim.data, f"{kind}_{obj}")
            row = dict(seed=seed, noise=cfg.depth_noise, object=obj, kind=kind, found=scene is not None,
                       xy_error=math.nan, z_error=math.nan, yaw_error=math.nan)
            if scene is not None:
                est = scene.pegs[kind] if obj == "peg" else scene.ports[kind]
                point = est.top if obj == "peg" else est.entrance
                row["xy_error"] = float(np.linalg.norm(point[:2] - true_point[:2]))
                row["z_error"] = float(point[2] - true_point[2])
                if kind == "square":
                    row["yaw_error"] = wrap_quarter_turn(est.yaw - true_yaw)
            rows.append(row)
    return rows


def perception_check(seeds, noises, base: SceneConfig = SceneConfig()) -> pd.DataFrame:
    """Every seed at every noise level: a table with one row per object per trial."""
    rows = []
    for noise in noises:
        for seed in seeds:
            rows += check_layout(replace(sample_layout(base, seed), depth_noise=noise), seed)
    return pd.DataFrame(rows)


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """Per noise level and object: how often found, and the error's median / 95th percentile / max, in mm and °."""
    e = errors.assign(xy_mm=errors.xy_error * 1e3, z_mm=errors.z_error.abs() * 1e3,
                      yaw_deg=np.degrees(errors.yaw_error.abs()))
    stats = e.groupby(["noise", "object"]).agg(
        found=("found", "mean"),
        xy_med=("xy_mm", "median"), xy_p95=("xy_mm", lambda s: s.quantile(0.95)), xy_max=("xy_mm", "max"),
        z_p95=("z_mm", lambda s: s.quantile(0.95)), z_max=("z_mm", "max"),
        yaw_p95=("yaw_deg", lambda s: s.quantile(0.95)), yaw_max=("yaw_deg", "max"))
    return stats.round(3)
