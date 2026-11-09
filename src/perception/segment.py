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
    """An object's top-surface height [m above the table], from the heights of all its pixels.

    The pixels of one object do not share one height. The camera looks down at an angle on anything away
    from the image centre, so it also sees the *sides* of a peg, and side pixels cover every height from
    the table (0) up to the top. Depth noise (σ ≈ 2 mm) then scatters every pixel a little further.

    So the obvious summaries are wrong:
      - mean: dragged down by the side pixels, so a peg can look shorter than it is;
      - max:  picks the single pixel the noise pushed highest, so it reads a few mm too high.

    Instead, two steps:
      1. 90th percentile = sort the heights and take the one 90 % of the way up. The flat top face is the
         highest part of the object and makes up well over 10 % of its pixels, so this lands on the top
         face, while skipping the few noisy pixels above it. That gives a rough estimate.
      2. Median of the pixels within 5 mm of that estimate, which are essentially just the top face.
         5 mm is 2.5 of the noise, wide enough to include almost all of the noisy top-face pixels, yet far
         narrower than the 20 mm gap between a block's top (30 mm) and a peg's (50 mm), so the two can
         never be mixed up.

    Checked over 40 random layouts at depth noise 0, 2 and 3 mm: always exactly two pegs and two blocks.
    """
    guess = np.percentile(heights, 90)                         # step 1: rough top, robust to outliers
    top_face = heights[np.abs(heights - guess) < 0.005]        # step 2: keep pixels within 5 mm of it
    return float(np.median(top_face))                          #         and take their median



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
