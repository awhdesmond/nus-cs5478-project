"""Seeded random layouts: where the pegs start and where the ports are.

The same seed always gives the same layout, so two controllers can be compared on identical trials.
"""

from dataclasses import replace

import numpy as np

from sim.config import PEGS, PlanarPose, SceneConfig


def sample_two_poses(rng: np.random.Generator, center, half, spacing: float, max_tries: int = 1000):
    """One pose per peg kind, uniform in the rectangle center ± half, at least `spacing` apart, any yaw.

    Rejection sampling: draw both, keep them only if they're far enough apart, else draw again.
    """
    lo, hi = np.array(center) - half, np.array(center) + half
    for _ in range(max_tries):
        poses = {}
        for kind in PEGS:
            x, y = rng.uniform(lo, hi)
            poses[kind] = PlanarPose(float(x), float(y), float(rng.uniform(-np.pi, np.pi)))
        a, b = poses.values()
        if np.hypot(a.x - b.x, a.y - b.y) >= spacing:
            return poses
    raise RuntimeError(f"no layout found with spacing {spacing} m")


def sample_layout(cfg: SceneConfig, seed: int) -> SceneConfig:
    """A copy of `cfg` with this seed's peg and port poses (cfg itself is left unchanged).

    Pegs and ports draw from separate random streams (seed, n), so changing one never shifts the other.
    """
    pegs = sample_two_poses(np.random.default_rng([seed, 3]), cfg.pick_area_center, cfg.pick_area_half,
                            cfg.min_peg_spacing)
    ports = sample_two_poses(np.random.default_rng([seed, 6]), cfg.port_area_center, cfg.port_area_half,
                             cfg.min_port_spacing)
    return replace(cfg, peg_poses=pegs, port_poses=ports)
