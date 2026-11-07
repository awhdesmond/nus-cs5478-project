"""Find each port's hole: which port it is, where its entrance is, and how it's turned.

Each hole has a 45° chamfer (lead-in) around its rim, coloured brown, unlike the grey-blue block. The
chamfer ring's outer edge is the rim, and the rim lies exactly in the block's top plane. So: take the
chamfer pixels, fill in everything inside the ring, and the result is the opening as seen at the top plane.
Measure it on that plane exactly as a peg's top face (perception/pegs.py), and its centre is the entrance.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from perception.camera import CameraObservation
from perception.pegs import measure_top_face
from perception.segment import Region, Segmentation


@dataclass(frozen=True)
class PortConfig:
    min_saturation: int = 90  # [0-255] the brown chamfer is ~140, the grey-blue block ~35
    # Fill of the opening (area / smallest-rectangle area): a square hole is ~1, the 16-sided round hole ~0.8.
    square_fill: float = 0.89


@dataclass
class PortEstimate:
    kind: str  # "round" or "square"
    entrance: np.ndarray  # [m] world position of the hole's centre at the block's top surface, (3,)
    yaw: float  # [rad] square: in [-π/4, π/4); round: 0
    size: float  # [m] opening across, chamfer included: square side, or round equal-area diameter
    fill: float


def opening_mask(obs: CameraObservation, region: Region, cfg: PortConfig) -> np.ndarray:
    """The hole's opening at the top plane: the chamfer ring and everything inside it."""
    hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
    ring = (region.mask & (hsv[..., 1] >= cfg.min_saturation)).astype(np.uint8)
    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    opening = np.zeros_like(ring)
    # The ring's outer contour, filled: covers the chamfer and the dark hole inside it.
    cv2.drawContours(opening, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return opening.astype(bool)


def estimate_ports(obs: CameraObservation, seg: Segmentation, cfg: PortConfig = PortConfig()) -> dict[str, PortEstimate]:
    """Both ports, keyed by kind. As with pegs, one of each: if both look alike, the squarer one is square."""
    blocks = seg.of_kind("block")
    if len(blocks) != 2:
        raise RuntimeError(f"expected 2 port blocks, found {len(blocks)}")
    faces = [measure_top_face(obs, opening_mask(obs, b, cfg), seg.table_z + b.top) for b in blocks]
    kinds = ["square" if f.fill > cfg.square_fill else "round" for f in faces]
    if kinds[0] == kinds[1]:
        kinds = ["square", "round"] if faces[0].fill > faces[1].fill else ["round", "square"]
    ports = {}
    for face, kind in zip(faces, kinds):
        if kind == "square":
            ports[kind] = PortEstimate(kind, face.centre, face.square_yaw, math.sqrt(face.area), face.fill)
        else:
            ports[kind] = PortEstimate(kind, face.centre, 0.0, 2 * math.sqrt(face.area / math.pi), face.fill)
    return ports
