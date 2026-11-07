"""Split a depth image into the table and the objects standing on it, and tell pegs from port blocks.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from perception.camera import CameraObservation


@dataclass(frozen=True)
class SegmentConfig:
    peg_height: float = 0.050  # [m] nominal, above the table
    block_height: float = 0.030  # [m] nominal: port base 10 mm + hole depth 20 mm
    min_object_height: float = 0.006  # [m] a pixel this far above the table belongs to an object (3σ at 2 mm noise)
    min_area: int = 200  # [px] smaller blobs are noise; a peg's top face alone is ~700 px
    open_radius: int = 2  # [px] morphological opening: erases isolated noisy pixels before grouping


@dataclass
class Region:
    kind: str  # "peg" or "block"
    mask: np.ndarray  # (height, width) bool: every pixel of this object, sides included
    top: float  # [m] height of its top surface above the table


@dataclass
class Segmentation:
    table_z: float  # [m] world height of the table surface, as measured
    heights: np.ndarray  # (height, width) [m] each pixel's height above the table
    points: np.ndarray  # (height, width, 3) [m] each pixel back-projected to the world
    regions: list[Region]

    def of_kind(self, kind: str) -> list[Region]:
        return [r for r in self.regions if r.kind == kind]


def table_height(z: np.ndarray, bin_size: float = 0.001) -> float:
    """The table is the surface most pixels see: the peak of a histogram of world heights, refined.

    The peak bin is found first (robust to the floor at the image edges and to objects), then the median of
    the pixels within 5 mm of it gives a sub-millimetre estimate even under depth noise.
    """
    z = z[np.isfinite(z)]
    counts, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_size, bin_size))
    peak = edges[np.argmax(counts)] + bin_size / 2
    return float(np.median(z[np.abs(z - peak) < 0.005]))


def top_height(heights: np.ndarray) -> float:
    """An object's top surface height from its pixels' heights.

    Its sides (seen in perspective) span everything from 0 up to the top, but the flat top face is the
    biggest group of equal heights near the top. Take the 90th percentile as a first guess, then the median of
    pixels within 5 mm of it.
    """
    guess = np.percentile(heights, 90)
    return float(np.median(heights[np.abs(heights - guess) < 0.005]))


def segment(obs: CameraObservation, cfg: SegmentConfig = SegmentConfig()) -> Segmentation:
    points = obs.points()
    table_z = table_height(points[..., 2])
    heights = points[..., 2] - table_z

    # Anything clearly above the table is part of an object. Opening (erode, then dilate) removes specks of
    # noise smaller than the structuring element, and leaves real objects (tens of pixels across) intact.
    above = (heights > cfg.min_object_height).astype(np.uint8)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.open_radius + 1,) * 2)
    above = cv2.morphologyEx(above, cv2.MORPH_OPEN, disk)

    # Connected components: each group of touching pixels gets its own label (0 = background).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    regions = []
    split = (cfg.peg_height + cfg.block_height) / 2  # 40 mm: halfway between a block's top and a peg's
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < cfg.min_area:
            continue
        mask = labels == label
        top = top_height(heights[mask])
        regions.append(Region("peg" if top > split else "block", mask, top))
    return Segmentation(table_z, heights, points, regions)
