"""Find each peg's top face, classify it round or square, and estimate where it is and how it's turned.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from perception.camera import CameraObservation
from perception.segment import Region, Segmentation


@dataclass(frozen=True)
class PegConfig:
    min_saturation: int = 60  # [0-255] amber, not the grey table or blocks
    min_value: int = 180  # [0-255] bright: the lit top face, not the dark sides

    square_fill: float = 0.86


@dataclass
class PegEstimate:
    kind: str  # "round" or "square"
    top: np.ndarray  # [m] world position of the top face's centre, (3,)
    yaw: float  # [rad] square: in [-π/4, π/4), since it looks the same every 90°; round: 0
    size: float  # [m] square: side; round: diameter
    fill: float  # area / smallest-rectangle area (1 = square, 0.785 = circle)
    outline: np.ndarray  # [m] top-face boundary in world xy, (N, 2): for grasp planning in step 4


def top_face_mask(obs: CameraObservation, region: Region, cfg: PegConfig) -> np.ndarray:
    """The pixels of this peg's top face: bright amber pixels within its region, as one solid blob."""
    hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)  # hue, saturation, value (brightness)
    bright = (hsv[..., 1] >= cfg.min_saturation) & (hsv[..., 2] >= cfg.min_value)
    mask = (region.mask & bright).astype(np.uint8)
    # Keep the largest blob, filled in: drawing its outer contour filled removes any holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return solid.astype(bool)


def wrap_quarter_turn(angle: float) -> float:
    """An angle into [-π/4, π/4): a square's yaw is only defined modulo 90°."""
    return (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4


@dataclass
class TopFace:
    """A top face measured on its plane, before deciding what kind of peg it is."""
    centre: np.ndarray  # [m] world, (3,)
    area: float  # [m²]
    fill: float  # area / smallest-rectangle area
    square_yaw: float # [rad] its yaw if read as a square, in [-π/4, π/4)
    outline: np.ndarray  # [m] (N, 2)


def measure_top_face(obs: CameraObservation, mask: np.ndarray, z: float) -> TopFace:
    """Area, centre, fill and rectangle yaw of a top face seen as `mask`, lying in the plane at height z."""
    cam = obs.camera
    v, u = np.nonzero(mask)
    centre = cam.on_plane(u, v, z)[:, :2].mean(axis=0)  # centroid of all its pixels, on the plane
    # Each pixel covers (distance / focal)² of a plane facing the camera (ours looks straight down).
    pixel = (cam.position[2] - z) / cam.focal  # [m] side of one pixel on that plane
    area = len(u) * pixel**2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(contours, key=len)[:, 0, :]  # (N, 2) pixel (u, v) of the boundary pixels' centres
    outline = cam.on_plane(boundary[:, 0], boundary[:, 1], z)[:, :2]
    # Smallest rectangle around the outline, in millimetres about the centre (float32 keeps precision there).
    _, (w, h), _ = cv2.minAreaRect(((outline - centre) * 1000).astype(np.float32))
    # The contour runs through boundary pixels' centres, so the true shape is one pixel wider and taller.
    w, h = w / 1000 + pixel, h / 1000 + pixel
    return TopFace(np.array([*centre, z]), area, area / (w * h), square_yaw(outline, centre), outline)



def square_yaw(outline: np.ndarray, centre: np.ndarray) -> float:
    """A square's yaw from its whole outline, in [-π/4, π/4).

    Walking around a square, the distance r from the centre peaks 4 times, at the corners, which sit at
    yaw + 45° + k·90°. The 4th Fourier harmonic of r(φ) finds that pattern's phase using every outline point,
    so pixel steps average out (the smallest rectangle uses only a few extreme points: twice the error).
    At a corner, 4φ = 4·yaw + 180° + k·360°, so e^{4iφ} = -e^{4i·yaw}: hence the minus sign.
    """
    d = outline - centre
    phi, r = np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1])
    harmonic = np.sum((r - r.mean()) * np.exp(4j * phi))  # big where r is big, at each corner's angle × 4
    return wrap_quarter_turn(float(np.angle(-harmonic)) / 4)



def as_peg(face: TopFace, kind: str) -> PegEstimate:
    """Read a measured top face as a peg of this kind."""
    if kind == "square":
        return PegEstimate("square", face.centre, face.square_yaw, math.sqrt(face.area), face.fill, face.outline)
    # A circle has no yaw; its size is the diameter of a circle of that area.
    return PegEstimate("round", face.centre, 0.0, 2 * math.sqrt(face.area / math.pi), face.fill, face.outline)


def estimate_pegs(obs: CameraObservation, seg: Segmentation, cfg: PegConfig = PegConfig()) -> dict[str, PegEstimate]:
    """Both pegs, keyed by kind.

    Each face is square if it fills its smallest rectangle well enough. There is one peg of each kind, so if
    both look alike (a borderline fill), the one with the higher fill is the square one.
    """
    faces = [measure_top_face(obs, top_face_mask(obs, r, cfg), seg.table_z + r.top) for r in seg.of_kind("peg")]
    if len(faces) != 2:
        raise RuntimeError(f"expected 2 pegs, found {len(faces)}")
    kinds = ["square" if f.fill > cfg.square_fill else "round" for f in faces]
    if kinds[0] == kinds[1]:
        kinds = ["square", "round"] if faces[0].fill > faces[1].fill else ["round", "square"]
    return {kind: as_peg(face, kind) for face, kind in zip(faces, kinds)}
