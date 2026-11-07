from dataclasses import replace

import numpy as np

from sim.config import PEGS, PegPose, PlanarPose, SceneConfig

# The proposal's orientation conditions: all upright, or each peg independently upright or lying (50/50).
# "lying" (both lying) is extra, handy for testing.
ORIENTATIONS = ("upright", "mixed", "lying")


def peg_axis_segment(cfg: SceneConfig, pose: PegPose) -> tuple[np.ndarray, np.ndarray]:
    """The peg's long axis seen from above, as two (x, y) end points. Upright, both ends are the same point."""
    centre = np.array([pose.x, pose.y])
    if not pose.lying:
        return centre, centre
    half = cfg.peg_length / 2 * np.array([np.cos(pose.yaw), np.sin(pose.yaw)])
    return centre - half, centre + half


def segment_distance(a0, a1, b0, b1) -> float:
    """Shortest distance between 2D segments a0-a1 and b0-b1 (a segment may be a single point)."""

    def point_to_segment(p, s0, s1):
        d = s1 - s0
        # Project p onto the segment's line, clamped to the segment (t = 0 at s0, 1 at s1).
        t = 0.0 if not d.any() else float(np.clip(np.dot(p - s0, d) / np.dot(d, d), 0.0, 1.0))
        return float(np.linalg.norm(p - (s0 + t * d)))

    def side(p, q, r):
        """+1 or -1: which side of line p→q the point r is on (0 if on it)."""
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    # If each segment's ends lie on opposite sides of the other's line, they cross: distance 0.
    if side(a0, a1, b0) * side(a0, a1, b1) < 0 and side(b0, b1, a0) * side(b0, b1, a1) < 0:
        return 0.0
    # Otherwise the closest pair always involves an end point of one of them.
    return min(point_to_segment(a0, b0, b1), point_to_segment(a1, b0, b1),
               point_to_segment(b0, a0, a1), point_to_segment(b1, a0, a1))


def sample_peg_poses(cfg: SceneConfig, seed: int, orientation: str = "upright",
                     max_tries: int = 1000) -> dict[str, PegPose]:
    """Peg poses in the pick zone: wholly inside it, axes at least min_peg_spacing apart, any yaw."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}")
    # Separate random streams (seed, n): changing one draw never shifts another. E.g. the lying flags come
    # from their own stream, so turning "mixed" on doesn't change where the pegs are drawn from.
    flips = np.random.default_rng([seed, 7]).random(len(PEGS)) < 0.5
    lying = {k: orientation == "lying" or (orientation == "mixed" and bool(f)) for k, f in zip(PEGS, flips)}
    rng = np.random.default_rng([seed, 3])
    lo = np.array(cfg.pick_area_center) - cfg.pick_area_half
    hi = np.array(cfg.pick_area_center) + cfg.pick_area_half
    for _ in range(max_tries):
        # Rejection sampling: draw both, keep them only if they fit and are far enough apart.
        poses = {}
        for kind in PEGS:
            x, y = rng.uniform(lo, hi)
            poses[kind] = PegPose(float(x), float(y), float(rng.uniform(0, 2 * np.pi)), lying[kind])
        segments = [peg_axis_segment(cfg, p) for p in poses.values()]
        inside = all(np.all(end >= lo) and np.all(end <= hi) for seg in segments for end in seg)
        if inside and segment_distance(*segments[0], *segments[1]) >= cfg.min_peg_spacing:
            return poses
    raise RuntimeError(f"no peg layout found with spacing {cfg.min_peg_spacing} m")


def sample_port_poses(cfg: SceneConfig, seed: int, max_tries: int = 1000) -> dict[str, PlanarPose]:
    """Port block poses in the port area: centres at least min_port_spacing apart, any yaw."""
    rng = np.random.default_rng([seed, 6])
    lo = np.array(cfg.port_area_center) - cfg.port_area_half
    hi = np.array(cfg.port_area_center) + cfg.port_area_half
    for _ in range(max_tries):
        poses = {}
        for kind in PEGS:
            x, y = rng.uniform(lo, hi)
            poses[kind] = PlanarPose(float(x), float(y), float(rng.uniform(-np.pi, np.pi)))
        a, b = poses.values()
        if np.hypot(a.x - b.x, a.y - b.y) >= cfg.min_port_spacing:
            return poses
    raise RuntimeError(f"no port layout found with spacing {cfg.min_port_spacing} m")


def sample_layout(cfg: SceneConfig, seed: int, orientation: str = "upright") -> SceneConfig:
    """A copy of `cfg` with this seed's peg and port poses (cfg itself is left unchanged)."""
    return replace(cfg, peg_poses=sample_peg_poses(cfg, seed, orientation), port_poses=sample_port_poses(cfg, seed))
