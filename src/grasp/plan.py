"""Grasp planning: where to put the parallel gripper on a peg, from the peg's top-face outline alone.

The peg is a prism (upright), so its sides are the top-face outline extruded downward: the outline's normals
are the side faces' normals. For each candidate closing direction d (through the face's centre), each pad
is a flat plate perpendicular to d, so it touches the outline's points that stick out furthest along ±d.
A grasp is kept only if, on both sides:
  - the contact is a flat or gently curved patch, not a corner (a pad on a corner slips or twists the peg);
  - the surface normal there is inside the friction cone around d (force closure: squeezing can't slide it);
and the open fingers don't hit any other object on the way down.
"""

import math
from dataclasses import dataclass

import numpy as np

from perception.pegs import PegEstimate


@dataclass(frozen=True)
class GraspConfig:
    depth: float = 0.012  # [m] tcp this far below the peg's top: pads cover the top ~20 mm, fingertips stay
    # well above the table, and the peg sticks out 30 mm below them, enough to reach a port's floor
    friction: float = 0.5  # assumed pad-peg friction for planning: the pads are ~1.0, plan with margin
    directions: int = 72  # closing directions tried, evenly over 180° (2.5° apart)
    band: float = 0.0005  # [m] outline points within this of the outermost one are in contact with the pad
    # A pad on a flat or curved side touches a patch at least this wide; on a corner, less. Measured on
    # perceived outlines with band 0.5 mm: round pegs ≥ 2.3 mm in every direction; a square's patch is
    # ≥ 3.9 mm within 5° of a face, ≤ 2.0 mm once 20° or more off (its corner).
    min_contact: float = 0.002  # [m]
    finger_reach: float = 0.021  # [m] open fingers' outer faces from the gripper's centre line: 15 + 6 mm
    finger_half_width: float = 0.008  # [m] half the pads' 16 mm width
    clearance: float = 0.003  # [m] extra room kept between the open fingers and anything else
    # Grasps whose normals are within this of the best one count as equally good; among those the smallest
    # wrist turn wins. Pixel steps make a circle's normals vary by ~1-2° with direction, so a tighter tie
    # would pick grasps on noise.
    tie: float = math.radians(5)


@dataclass
class Grasp:
    kind: str
    tcp: np.ndarray  # [m] world position for the tcp when the fingers close, (3,)
    yaw: float  # [rad] tcp yaw (tool_down(yaw)): the fingers close along (sin yaw, -cos yaw)
    width: float  # [m] expected distance between the pads once closed on the peg
    normal_angle: float  # [rad] worst angle between a contact normal and the closing direction


def closing_direction(yaw: float) -> np.ndarray:
    """Unit vector the fingers close along, for tcp yaw `yaw` (the tcp's y axis; see tool_down)."""
    return np.array([math.sin(yaw), -math.cos(yaw)])


def contact_patch(outline: np.ndarray, centre: np.ndarray, d: np.ndarray, band: float):
    """Where a flat pad pushing along -d meets the outline: points within `band` of the furthest along d.

    Returns (width of the patch across d [m], unit outward surface normal there, distance from the centre).
    """
    along = (outline - centre) @ d
    patch = outline[along > along.max() - band]
    n = np.array([-d[1], d[0]])  # across the closing direction, along the pad
    width = float(np.ptp(patch @ n)) if len(patch) > 1 else 0.0
    if len(patch) < 3:
        return width, d, float(along.max())
    # Normal = the direction the patch varies least in (principal component analysis), pointing outward.
    _, _, vt = np.linalg.svd(patch - patch.mean(axis=0))
    normal = vt[1] if vt[1] @ d > 0 else -vt[1]
    return width, normal, float(along.max())


def antipodal_axis(outline: np.ndarray, centre: np.ndarray, d: np.ndarray, band: float, rounds: int = 3):
    """Settle the closing direction onto the axis the two contact normals define (their mean).

    A few degrees off a square's face, the patch catches some of the corner too and its normal is skewed;
    re-measuring along the corrected axis gives the whole face, and its exact normal. On a circle every
    direction is already its own antipodal axis, so nothing moves. Returns (axis, width across the contacts).
    """
    for _ in range(rounds):
        (_, n1, a1), (_, n2, a2) = (contact_patch(outline, centre, s * d, band) for s in (1, -1))
        d = (n1 - n2) / np.linalg.norm(n1 - n2)
    return d, a1 + a2


def fingers_clear(centre: np.ndarray, d: np.ndarray, obstacles: list[np.ndarray], cfg: GraspConfig) -> bool:
    """Do the open fingers, lowered around `centre` closing along d, stay off every obstacle outline?

    Each finger sweeps a rectangle: from the peg's side out to finger_reach along ±d, finger_half_width
    across. Keep `clearance` from it.
    """
    n = np.array([-d[1], d[0]])
    for points in obstacles:
        rel = points - centre
        a, c = np.abs(rel @ d), np.abs(rel @ n)
        if np.any((a < cfg.finger_reach + cfg.clearance) & (c < cfg.finger_half_width + cfg.clearance)):
            return False
    return True


def plan_grasp(peg: PegEstimate, obstacles: list[np.ndarray] = (), reference_yaw: float = 0.0,
               cfg: GraspConfig = GraspConfig()) -> Grasp:
    """The best grasp on `peg`, or ValueError if none passes.

    obstacles: outlines (N, 2) [m] of everything else on the table. Among passing grasps, the one with the
    best-aligned normals wins; near-ties (within cfg.tie) go to the yaw closest to reference_yaw (the wrist's
    current yaw), so the wrist turns as little as possible.
    """
    centre = peg.top[:2]
    cone = math.atan(cfg.friction)  # friction cone half-angle: steeper than this and the pad slides
    candidates = []
    for i in range(cfg.directions):
        theta = math.pi * i / cfg.directions  # d and -d are the same pair of pads
        d = np.array([math.cos(theta), math.sin(theta)])
        sides = [contact_patch(peg.outline, centre, s * d, cfg.band) for s in (1, -1)]
        if min(w for w, _, _ in sides) < cfg.min_contact:
            continue  # a corner on at least one side
        angle = max(math.acos(np.clip(normal @ (s * d), -1, 1)) for (_, normal, _), s in zip(sides, (1, -1)))
        if angle > cone or not fingers_clear(centre, d, list(obstacles), cfg):
            continue
        axis, width = antipodal_axis(peg.outline, centre, d, cfg.band)
        # The tcp yaw whose closing direction is ±axis, picked as close to the reference as possible.
        yaw = math.atan2(axis[0], -axis[1])
        yaw += math.pi * round((reference_yaw - yaw) / math.pi)
        candidates.append((angle, abs(yaw - reference_yaw), yaw, width))
    if not candidates:
        raise ValueError(f"no grasp passes on the {peg.kind} peg")
    best = min(a for a, *_ in candidates)
    angle, _, yaw, width = min((c for c in candidates if c[0] <= best + cfg.tie), key=lambda c: c[1])
    tcp = np.array([*centre, peg.top[2] - cfg.depth])
    return Grasp(peg.kind, tcp, yaw, width, angle)
