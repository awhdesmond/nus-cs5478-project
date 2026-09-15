"""Pose estimation for the assembly task: identify and localise connectors and socket holes.

`PoseEstimator` is the interface controllers use: camera observation in, `SceneEstimate` out.
`GeometricEstimator` is the classical implementation. It fuses depth (detection, heights, regions) with
colour (crisp top-face outlines):

1. Back-project depth to a world height map; estimate the table height as the dominant surface.
2. Connectors: regions standing well above the table (connector tops, not the lower fixture). The top
   face is taken from the colour image (the lit top face is bright and saturated; its edges stay sharp
   when depth is noisy), falling back to the depth top face if the colour mask disagrees in size. Its
   pixels are intersected with the top plane (height from depth), so shapes are metric and free of
   perspective.
3. Sockets: the fixture's top surface is a plane with holes (from depth). Each hole is filled and its
   pixels intersected with that plane to get the opening in world xy; the square socket's yaw comes from
   its block's long outer edges rather than the small opening.
4. Identify each footprint by fitting an ideal circle and an ideal square (yaw from the minimum-area
   rectangle) of the same area and keeping whichever overlaps better (IoU). When exactly two candidates
   of a part type are found, the scene prior (one round, one square) labels the more square-like one
   as square.

Yaw estimators compared on 40 layouts (95th percentile): depth minimum-area rectangle 0.37° / 1.48° /
3.15° at depth noise σ 0 / 1 / 2 mm; colour outline 0.47° at all three.

A model-based or learned estimator can replace it by implementing `estimate`.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from insertion.camera import CameraObservation


@dataclass
class ShapeFit:
    kind: str  # "round" or "square"
    centre_xy: np.ndarray  # [m]
    yaw: float  # [rad], square: in [−π/4, π/4); round: 0
    iou_square: float  # overlap with the best-yaw square of equal area
    iou_round: float  # overlap with the circle of equal area
    size: float  # [m] equivalent diameter (round) or side (square)
    area_px: int

    @property
    def squareness(self) -> float:
        """> 0 looks square, < 0 looks round."""
        return self.iou_square - self.iou_round


@dataclass
class ConnectorEstimate:
    kind: str
    top: np.ndarray  # top-face centre, world [m]
    yaw: float
    fit: ShapeFit


@dataclass
class SocketEstimate:
    kind: str
    entrance: np.ndarray  # hole opening centre on the fixture top plane, world [m]
    yaw: float
    fit: ShapeFit


@dataclass
class SceneEstimate:
    table_height: float
    connectors: list[ConnectorEstimate] = field(default_factory=list)
    sockets: list[SocketEstimate] = field(default_factory=list)

    def connector(self, kind: str) -> ConnectorEstimate | None:
        matches = [c for c in self.connectors if c.kind == kind]
        return matches[0] if len(matches) == 1 else None

    def socket(self, kind: str) -> SocketEstimate | None:
        matches = [s for s in self.sockets if s.kind == kind]
        return matches[0] if len(matches) == 1 else None


class PoseEstimator(Protocol):
    def estimate(self, observation: CameraObservation) -> SceneEstimate: ...


@dataclass
class GeometricEstimatorConfig:
    connector_min_height: float = 0.040  # [m] above table: connector tops (+50 mm) but not the fixture top (+30 mm)
    fixture_band: tuple[float, float] = (0.020, 0.040)  # [m] above table where the fixture top is searched for
    top_face_tolerance: float = 0.0015  # [m] pixels this close to a region's top height form its top face
    plane_tolerance: float = 0.0005  # [m] fixture top plane (keeps the 45° chamfer out of the plane mask)
    min_connector_px: int = 150
    min_hole_px: int = 150
    min_fixture_px: int = 2000
    top_face_min_value: int = 200  # HSV value (0–255) of the lit top face; lower admits brightly lit side faces
    top_face_min_saturation: int = 80  # HSV saturation: the connector colour, not grey fixture or table
    colour_area_ratio: tuple[float, float] = (0.6, 1.8)  # colour top face must be this size relative to depth's
    median_filter_px: int = 5  # depth median filter (odd, ≤ 5 for float images); 0 disables
    noise_sigmas: float = 3.0  # height tolerances widen to this many robust std devs of the measured table noise
    morph_px: int = 3  # opening/closing kernel for cleaning masks; 0 disables


def fit_footprint(xy: np.ndarray, pixel_area: float, cfg: GeometricEstimatorConfig, yaw: float | None = None) -> ShapeFit:
    """Identify a footprint (metric xy of the pixels covering it, each `pixel_area` m²) and fit its pose.

    Area comes from the pixel count, so stray noise pixels change it only slightly (unlike a convex hull).
    The square model's yaw is `yaw` if given, else the minimum-area rectangle of the points.
    """
    area = len(xy) * pixel_area
    centre = xy.mean(axis=0)
    local = xy - centre
    side = math.sqrt(area)
    radius = math.sqrt(area / math.pi)
    if yaw is None:
        (_, _), (_, _), angle = cv2.minAreaRect(xy.astype(np.float32))
        yaw = math.radians(angle)
    yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def iou(inside_count: int, model_area: float) -> float:
        intersection = inside_count * pixel_area
        return intersection / (area + model_area - intersection)

    c, s_ = math.cos(yaw), math.sin(yaw)
    u = local[:, 0] * c + local[:, 1] * s_
    v = -local[:, 0] * s_ + local[:, 1] * c
    iou_square = iou(int(np.count_nonzero((np.abs(u) <= side / 2) & (np.abs(v) <= side / 2))), side * side)
    iou_round = iou(int(np.count_nonzero(np.hypot(local[:, 0], local[:, 1]) <= radius)), area)
    if iou_square > iou_round:
        return ShapeFit("square", centre, yaw, iou_square, iou_round, side, len(xy))
    return ShapeFit("round", centre, 0.0, iou_square, iou_round, 2 * radius, len(xy))


def relabel(fit: ShapeFit, kind: str) -> ShapeFit:
    """Same measurements, identification forced by the scene prior."""
    return ShapeFit(kind, fit.centre_xy, fit.yaw if kind == "square" else 0.0, fit.iou_square, fit.iou_round,
                    fit.size, fit.area_px)


def apply_one_of_each(fits: list[ShapeFit]) -> list[ShapeFit]:
    """With exactly two candidates, the more square-like is the square (the scene has one of each)."""
    if len(fits) != 2:
        return fits
    order = sorted(range(2), key=lambda i: fits[i].squareness)
    labelled = list(fits)
    labelled[order[0]] = relabel(fits[order[0]], "round")
    labelled[order[1]] = relabel(fits[order[1]], "square")
    return labelled


class GeometricEstimator:
    def __init__(self, cfg: GeometricEstimatorConfig | None = None):
        self.cfg = cfg or GeometricEstimatorConfig()

    def estimate(self, observation: CameraObservation) -> SceneEstimate:
        cfg = self.cfg
        calib = observation.calibration
        depth = observation.depth.astype(np.float32)
        if cfg.median_filter_px:
            depth = cv2.medianBlur(depth, cfg.median_filter_px)
        heights = calib.height_map(depth)
        table = self._table_height(heights)
        near_table = heights[np.abs(heights - table) < 0.005]
        self.noise = float(1.4826 * np.median(np.abs(near_table - np.median(near_table)))) if near_table.size else 0.0
        scene = SceneEstimate(table_height=table)
        scene.connectors = self._connectors(heights, depth, table, observation)
        scene.sockets = self._sockets(heights, table, observation)
        return scene

    def _tolerance(self, base: float) -> float:
        """A height tolerance widened to cover the measured (post-filter) depth noise."""
        return max(base, self.cfg.noise_sigmas * self.noise)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        if not self.cfg.morph_px:
            return mask.astype(np.uint8)
        kernel = np.ones((self.cfg.morph_px, self.cfg.morph_px), np.uint8)
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _table_height(heights: np.ndarray) -> float:
        """Dominant surface height: the mode of a 1 mm height histogram."""
        valid = heights[np.isfinite(heights)]
        bins = np.arange(valid.min(), valid.max() + 0.001, 0.001)
        counts, edges = np.histogram(valid, bins=bins)
        peak = int(np.argmax(counts))
        in_peak = valid[(valid >= edges[peak]) & (valid < edges[peak + 1])]
        return float(np.median(in_peak))

    def _connectors(self, heights, depth, table, observation) -> list[ConnectorEstimate]:
        cfg = self.cfg
        calib = observation.calibration
        hsv = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2HSV)
        tall = self._clean(heights > table + cfg.connector_min_height)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(tall, connectivity=8)
        found = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < cfg.min_connector_px:
                continue
            region = labels == label
            top_height = float(np.percentile(heights[region], 95))
            face = region & (np.abs(heights - top_height) < self._tolerance(cfg.top_face_tolerance))
            face = self._largest_component(self._clean(face).astype(bool))
            if face.sum() < cfg.min_connector_px:
                continue
            top_z = float(np.median(calib.back_project(depth, face)[:, 2]))

            # Colour top face near the depth region; keep it only if its size agrees with the depth face.
            near = cv2.dilate(region.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            colour = near & (hsv[..., 2] >= cfg.top_face_min_value) & (hsv[..., 1] >= cfg.top_face_min_saturation)
            colour = self._largest_component(self._clean(colour).astype(bool))
            lo, hi = cfg.colour_area_ratio
            if lo * face.sum() <= colour.sum() <= hi * face.sum():
                face = colour
            rows, cols = np.nonzero(face)
            xy = self._pixels_on_plane(np.stack([cols, rows], axis=1), top_z, calib)
            pixel_area = ((calib.position[2] - top_z) / calib.focal) ** 2
            found.append((fit_footprint(xy, pixel_area, cfg), top_z))
        fits = apply_one_of_each([fit for fit, _ in found])
        return [ConnectorEstimate(fit.kind, np.array([*fit.centre_xy, z]), fit.yaw, fit) for fit, (_, z) in zip(fits, found)]

    def _sockets(self, heights, table, observation) -> list[SocketEstimate]:
        cfg = self.cfg
        calib = observation.calibration
        lo, hi = cfg.fixture_band
        band = (heights > table + lo) & (heights < table + hi)
        if band.sum() < cfg.min_fixture_px:
            return []
        plane = self._table_height(heights[band][:, None])
        plane_mask = self._clean(np.abs(heights - plane) < self._tolerance(cfg.plane_tolerance))
        contours, hierarchy = cv2.findContours(plane_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return []
        pixel_area = ((calib.position[2] - plane) / calib.focal) ** 2
        fits = []
        for index, (_, _, _, parent) in enumerate(hierarchy[0]):
            if parent < 0 or cv2.contourArea(contours[parent]) < cfg.min_fixture_px:
                continue  # only holes inside a sizeable plane region
            if cv2.contourArea(contours[index]) < cfg.min_hole_px:
                continue
            filled = np.zeros_like(plane_mask)
            cv2.drawContours(filled, contours, index, 1, thickness=cv2.FILLED)
            rows, cols = np.nonzero(filled)
            opening = self._pixels_on_plane(np.stack([cols, rows], axis=1), plane, calib)
            # Yaw from the socket block's outer outline (long straight edges), not the small opening.
            block = self._pixels_on_plane(contours[parent][:, 0, :], plane, calib)
            (_, _), (_, _), block_angle = cv2.minAreaRect(block.astype(np.float32))
            fits.append(fit_footprint(opening, pixel_area, cfg, yaw=math.radians(block_angle)))
        fits = apply_one_of_each(fits)
        return [SocketEstimate(fit.kind, np.array([*fit.centre_xy, plane]), fit.yaw, fit) for fit in fits]

    @staticmethod
    def _pixels_on_plane(pixels_uv: np.ndarray, plane_z: float, calib) -> np.ndarray:
        """World xy where the rays through pixel centres meet the horizontal plane z = plane_z."""
        u = pixels_uv[:, 0] + 0.5
        v = pixels_uv[:, 1] + 0.5
        rays_cam = np.stack([(u - calib.width / 2) / calib.focal, -(v - calib.height / 2) / calib.focal, -np.ones_like(u)], axis=1)
        rays = rays_cam @ calib.rotation.T
        t = (plane_z - calib.position[2]) / rays[:, 2]
        return (calib.position[None, :] + t[:, None] * rays)[:, :2]

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask.astype(bool)
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == best
