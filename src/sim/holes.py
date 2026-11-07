import math
import mujoco

from sim.utils import quat_z, quat_mul

BOX = mujoco.mjtGeom.mjGEOM_BOX
WALL_RGBA = [0.25, 0.25, 0.3, 1.0]
CHAMFER_RGBA = [0.45, 0.35, 0.2, 1.0]  # a different colour so the lead-in is visible


def add_square_hole(
        body: mujoco.MjsBody,
        half: float,
        wall: float,
        depth: float,
        chamfer: float,
        contact: dict,
        prefix: str
) -> None:
    """Walls around a chamfered square hole of half-width `half`, each `wall` thick.

    Frame: origin at the hole-bottom centre, +z up; the hole spans |x|, |y| < half, 0 < z < depth.
    Each wall is a lower box plus, near the top, an upper box set back by the chamfer and a box rotated
    45° (a "diamond") whose sloped face is the chamfer. Cross-section of the +x wall (c = chamfer):

        z=d      ╱██████      upper box:  x ∈ [h+c, h+t],  z ∈ [d-c, d]
               / ◇ █████      diamond:    centred at (h+c, d-c), rotated 45° about the wall's long axis
        z=d-c  ██◇███████
               ██████████     lower box:  x ∈ [h, h+t],    z ∈ [0, d-c]
        z=0    ██████████
              x=h

    The diamonds run `chamfer` past the hole's corners, so the chamfer ring has no notches to catch on.
    """
    h, H, d, c = half, half + wall, depth, chamfer

    def box(name, center, size, quat=(1, 0, 0, 0), rgba=WALL_RGBA):
        # MuJoCo box `size` is HALF-extents.
        body.add_geom(name=f"{prefix}_{name}", type=BOX, pos=list(center), size=list(size), quat=list(quat),
                      rgba=rgba, **contact)

    lower_top = d - c
    # Lower walls. ±x walls run the block's full length in y; ±y walls fit between them (no overlap).
    for sign, side in ((1, "px"), (-1, "nx")):
        box(f"wall_{side}", (sign * (h + H) / 2, 0, lower_top / 2), ((H - h) / 2, H, lower_top / 2))
    for sign, side in ((1, "py"), (-1, "ny")):
        box(f"wall_{side}", (0, sign * (h + H) / 2, lower_top / 2), (h, (H - h) / 2, lower_top / 2))
    # Upper walls, set back by the chamfer.
    for sign, side in ((1, "px"), (-1, "nx")):
        box(f"upper_{side}", (sign * (h + c + H) / 2, 0, d - c / 2), ((H - h - c) / 2, H, c / 2))
    for sign, side in ((1, "py"), (-1, "ny")):
        box(f"upper_{side}", (0, sign * (h + c + H) / 2, d - c / 2), (H, (H - h - c) / 2, c / 2))
    # Diamonds: a square cross-section of side c·√2, turned 45°, so its corners sit c from its centre.
    s = c * math.sqrt(2) / 2
    tilt_about_y = (math.cos(math.pi / 8), 0, math.sin(math.pi / 8), 0)  # 45° about y
    tilt_about_x = (math.cos(math.pi / 8), math.sin(math.pi / 8), 0, 0)  # 45° about x
    for sign, side in ((1, "px"), (-1, "nx")):
        box(f"chamfer_{side}", (sign * (h + c), 0, d - c), (s, h + c, s), tilt_about_y, CHAMFER_RGBA)
    for sign, side in ((1, "py"), (-1, "ny")):
        box(f"chamfer_{side}", (0, sign * (h + c), d - c), (h + c, s, s), tilt_about_x, CHAMFER_RGBA)


def add_ngon_hole(
        body: mujoco.MjsBody,
        apothem: float,
        sides: int,
        wall: float,
        depth: float,
        chamfer: float,
        contact: dict,
        prefix: str
) -> None:
    """Walls around a chamfered regular N-gon hole (our round hole), each `wall` thick.

    Each of the N faces is one wall section like the square hole's, rotated about z to angle 2πi/N, with
    its inner face `apothem` from the axis. Neighbouring boxes overlap a little near the outside; overlaps
    only add material, and the hole itself is exactly the N-gon.
    """
    R, t, d, c = apothem, wall, depth, chamfer
    half_chord = (R + t) * math.tan(math.pi / sides) + 1e-4  # half a wall's length along the rim, +0.1 mm
    s = c * math.sqrt(2) / 2
    tilt = (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0)  # 45° about the wall's own y

    def box(name, phi, radius, z, size, quat, rgba=WALL_RGBA):
        # Centre `radius` out from the axis in direction phi.
        body.add_geom(name=f"{prefix}_{name}", type=BOX, pos=[radius * math.cos(phi), radius * math.sin(phi), z],
                      size=list(size), quat=list(quat), rgba=rgba, **contact)

    for i in range(sides):
        phi = 2 * math.pi * i / sides  # direction of this face's outward normal
        turn = quat_z(phi)
        box(f"wall_{i}", phi, R + t / 2, (d - c) / 2, (t / 2, half_chord, (d - c) / 2), turn)
        box(f"upper_{i}", phi, R + c + (t - c) / 2, d - c / 2, ((t - c) / 2, half_chord, c / 2), turn)
        box(f"chamfer_{i}", phi, R + c, d - c, (s, (R + 2 * c) * math.tan(math.pi / sides) + 1e-4, s),
            quat_mul(turn, tilt), CHAMFER_RGBA)
