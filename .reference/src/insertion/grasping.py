"""Antipodal grasp proposals for the parallel-jaw gripper from the perceived connector outline.

An upright connector is a prism, so its side-surface normals are horizontal and equal to the outward
normals of its top-face outline (from the camera). For each candidate closing direction d:

- the fingers touch the object's extreme points along ±d: each contact patch is the outline within
  `patch_depth` of the extreme, and its normal comes from a line fit over the whole patch (a longer
  baseline than local outline normals, which pixel staircasing biases by a few degrees);
- the grasp is antipodal (force closure with two contacts) if the line between the contacts lies inside
  both friction cones, i.e. the angle between each patch normal and ±d is ≤ atan(μ) − margin;
- a patch shorter than `min_patch_length` is a corner or edge contact that would twist, so it is rejected;
- the grasp must fit the gripper opening and keep the fingers clear of any other object.

Candidates are ranked by friction-cone margin (then patch length), with a small preference for tool yaws
near the current wrist yaw. Each passing candidate's direction is then refined (twice) to the mean of its two
patch normals (n_hi − n_lo), so a square is gripped square-on to its flats rather than anywhere inside the
passing window of the direction grid. For a square this selects closing across opposite flats; for a round connector
any diameter passes.
"""

import math
from dataclasses import dataclass

import numpy as np

from insertion.config import AssemblyConfig


@dataclass
class GraspConfig:
    friction_assumed: float = 0.3  # planner's conservative μ (sets the friction-cone half-angle)
    friction_margin_deg: float = 5.0  # required margin inside the cone (normal-estimate noise)
    patch_depth: float = 0.0004  # [m] outline points within this of the extreme form a contact patch
    min_patch_length: float = 0.003  # [m] shorter patches are corner/edge contacts
    normal_window: int = 4  # outline neighbours on each side used for the local normal fit
    direction_step_deg: float = 2.0
    opening_clearance: float = 0.004  # [m] per side, between the open fingers and the object
    keepout_radius: float = 0.012  # [m] around other objects' centres that finger footprints must avoid
    yaw_preference_weight: float = 0.002  # score penalty per degree away from the preferred tool yaw


@dataclass
class GraspProposal:
    tcp: np.ndarray  # tcp position (finger pad centre), world [m]
    tool_yaw: float  # yaw for `tool_down`: fingers close along the world direction tool_yaw − 90°
    closing_angle: float  # world angle of the closing direction d [rad]
    width: float  # [m] object width along d
    friction_margin_deg: float
    patch_length: float  # [m] shorter of the two contact patches
    score: float


def outline_normals(outline: np.ndarray, window: int) -> np.ndarray:
    """Outward unit normals of an ordered closed outline, from a local principal-direction fit."""
    n = len(outline)
    normals = np.zeros_like(outline)
    centre = outline.mean(axis=0)
    for i in range(n):
        idx = [(i + k) % n for k in range(-window, window + 1)]
        local = outline[idx] - outline[idx].mean(axis=0)
        _, _, vt = np.linalg.svd(local, full_matrices=False)
        tangent = vt[0]
        normal = np.array([tangent[1], -tangent[0]])
        if np.dot(normal, outline[i] - centre) < 0:
            normal = -normal
        normals[i] = normal
    return normals


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))))))


def _evaluate(outline, normals, theta, cfg: GraspConfig, cone_half: float, max_width: float) -> dict | None:
    """Contact patches, friction-cone margin and patch length for closing direction theta (None if it fails)."""
    d = np.array([math.cos(theta), math.sin(theta)])
    perp = np.array([-d[1], d[0]])
    proj = outline @ d
    hi, lo = proj.max(), proj.min()
    width = hi - lo
    if width > max_width:
        return None
    patches = []
    for side, sign in ((proj >= hi - cfg.patch_depth, 1.0), (proj <= lo + cfg.patch_depth, -1.0)):
        points = outline[side]
        span = (points @ perp).max() - (points @ perp).min()
        if len(points) >= 3 and span >= cfg.min_patch_length:
            _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
            patch_normal = np.array([vt[0][1], -vt[0][0]])
        else:
            patch_normal = normals[side].mean(axis=0)
        patch_normal /= np.linalg.norm(patch_normal)
        if np.dot(patch_normal, normals[side].mean(axis=0)) < 0:
            patch_normal = -patch_normal  # outward, as the local normals
        patches.append((points.mean(axis=0), span, _angle_between(patch_normal, sign * d), patch_normal))
    (c_hi, span_hi, ang_hi, n_hi), (c_lo, span_lo, ang_lo, n_lo) = patches
    margin = cone_half - cfg.friction_margin_deg - max(ang_hi, ang_lo)
    patch_length = min(span_hi, span_lo)
    if margin < 0 or patch_length < cfg.min_patch_length:
        return None
    return dict(theta=theta, d=d, width=width, margin=margin, patch_length=patch_length,
                patch_mid=(c_hi + c_lo) / 2, normal_hi=n_hi, normal_lo=n_lo)


def propose_grasps(
    outline: np.ndarray,
    top_z: float,
    assembly: AssemblyConfig,
    cfg: GraspConfig | None = None,
    obstacles_xy: list[np.ndarray] | None = None,
    preferred_tool_yaw: float = 0.0,
) -> list[GraspProposal]:
    """All antipodal grasp proposals for one upright connector, best first."""
    cfg = cfg or GraspConfig()
    normals = outline_normals(outline, cfg.normal_window)
    cone_half = math.degrees(math.atan(cfg.friction_assumed))
    max_width = assembly.finger_open_gap - 2 * cfg.opening_clearance
    finger_reach = assembly.finger_open_gap / 2 + assembly.finger_thickness  # finger outer face from tcp, along d
    proposals = []
    for deg in np.arange(0.0, 180.0, cfg.direction_step_deg):
        theta = math.radians(deg)
        candidate = _evaluate(outline, normals, theta, cfg, cone_half, max_width)
        if candidate is None:
            continue
        # Refine to the direction the two contact patches actually face, then re-check (patches grow as d aligns).
        for _ in range(2):
            refined_d = candidate["normal_hi"] - candidate["normal_lo"]
            refined = _evaluate(outline, normals, math.atan2(refined_d[1], refined_d[0]), cfg, cone_half, max_width)
            if refined is None or refined["margin"] < candidate["margin"] - 1e-9:
                break
            candidate = refined
        theta, d = candidate["theta"], candidate["d"]
        perp = np.array([-d[1], d[0]])
        centre = outline.mean(axis=0)
        # Along the closing direction the coupled fingers centre the object; across it, use the patch midpoint.
        centre = centre + perp * float(np.dot(candidate["patch_mid"] - centre, perp))
        if obstacles_xy:
            finger_centres = [centre + d * finger_reach, centre - d * finger_reach]
            if any(np.linalg.norm(f - o) < cfg.keepout_radius for f in finger_centres for o in obstacles_xy):
                continue
        # Closing along d or −d is the same grasp; pick the tool yaw nearest the preference.
        options = [theta + math.pi / 2 + k * math.pi for k in (-2, -1, 0, 1, 2)]
        tool_yaw = math.remainder(min(options, key=lambda y: abs(math.remainder(y - preferred_tool_yaw, 2 * math.pi))), 2 * math.pi)
        yaw_away = abs(math.degrees(math.remainder(tool_yaw - preferred_tool_yaw, 2 * math.pi)))
        score = candidate["margin"] / cone_half + candidate["patch_length"] / 0.01 - cfg.yaw_preference_weight * yaw_away
        tcp = np.array([*centre, top_z - assembly.grasp_below_top])
        proposals.append(GraspProposal(tcp, tool_yaw, math.remainder(theta, math.pi), candidate["width"], candidate["margin"], candidate["patch_length"], score))
    return sorted(proposals, key=lambda p: p.score, reverse=True)
