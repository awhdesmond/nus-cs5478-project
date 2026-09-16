"""Pose estimation for the assembly task: identify and localise connectors and socket holes.

`PoseEstimator` is the interface controllers use: camera observation in, `SceneEstimate` out.
`GeometricEstimator` is the classical implementation. It fuses depth (detection, heights, regions) with
colour (crisp top-face outlines):

1. Back-project depth to a world height map; estimate the table height as the dominant surface.
2. Upright connectors: regions standing well above the table (connector tops, not the lower fixture). The top
   face is taken from the colour image (the lit top face is bright and saturated; its edges stay sharp
   when depth is noisy), falling back to the depth top face if the colour mask disagrees in size. Its
   pixels are intersected with the top plane (height from depth), so shapes are metric and free of
   perspective.
2b. Lying connectors: low regions (a few mm to 15 mm above the local table) away from the fixture and from
   upright connectors. Long axis from the principal direction of the back-projected points; centre from the
   axis extremes. Round vs square from the cross-section across the width: a cylinder's top drops ~1.5 mm
   within ±3.5 mm of the ridge, is wider (10 vs 8.5 mm) and higher (10 vs 8.5 mm); the three normalised
   margins are averaged so no single noisy feature decides. The marked (insertion) end is the end the blue band's
   pixels lie toward along the axis.
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
    score: float | None = None  # squareness from other evidence (lying connectors' cross-section), on a similar scale

    @property
    def squareness(self) -> float:
        """> 0 looks square, < 0 looks round."""
        return self.score if self.score is not None else self.iou_square - self.iou_round


@dataclass
class ConnectorEstimate:
    kind: str
    top: np.ndarray  # top-face centre, world [m]
    yaw: float
    fit: ShapeFit
    outline_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # ordered top-face boundary on the top plane [m]
    lying: bool = False
    axis_xy: np.ndarray | None = None  # lying: unit long-axis direction from the marked end toward the top (yaw = its angle)
    tip_found: bool = True  # lying: the marked end was seen (else axis_xy's sign is arbitrary)
    centre: np.ndarray | None = None  # lying: axis centre in xy, z = local table surface under it [m] (add half width)
    length: float = float("nan")  # lying: measured length [m]


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
    # Lying connectors
    lying_band: tuple[float, float] = (0.003, 0.015)  # [m] above the local table
    min_lying_px: int = 1500  # a lying connector covers ~4,000 px
    fixture_exclusion_px: int = 21  # dilation around the fixture (its holes' floors sit in the lying band)
    upright_exclusion_px: int = 15  # dilation around upright connectors (their side faces span all heights)
    lying_central_half_length: float = 0.018  # [m] cross-section uses |along-axis| below this (away from the ends)
    lying_drop_half_width: float = 0.0035  # [m] top drop measured this far either side of the ridge
    # Measured on 12 lying layouts at depth σ 0 / 1 / 2 mm: drop round 1.21–1.67 mm vs square −0.22–0.74 mm;
    # 2–98 % width round 9.37–9.79 vs square 8.32–9.02 mm; ridge height round ≈ 10 vs square ≈ 8.5 mm.
    lying_drop_split: tuple[float, float] = (0.00073, 0.00073)  # [m] (midpoint, half-span)
    lying_width_split: tuple[float, float] = (0.0092, 0.00075)
    lying_ridge_split: tuple[float, float] = (0.00925, 0.00075)
    lying_score_scale: float = 0.15  # maps the averaged margin onto the upright IoU-difference scale for the prior
    # Tip band (blue): hue 107–112, saturation ≥ 143 when rendered; the table shares the hue but not the saturation
    # (≤ 30), the fixture reaches saturation 64, the amber connector body has hue 21–30.
    band_hue: tuple[int, int] = (95, 125)  # OpenCV hue (0–179)
    band_min_saturation: int = 100
    min_band_px: int = 40


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
                    fit.size, fit.area_px, fit.score)


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
        upright = self._connectors(heights, depth, table, observation)
        lying = self._lying_connectors(heights, depth, table, observation)
        scene.connectors = self._label_connectors(upright + lying)
        scene.sockets = self._sockets(heights, table, observation)
        return scene

    @staticmethod
    def _label_connectors(connectors: list[ConnectorEstimate]) -> list[ConnectorEstimate]:
        """Scene prior over all connector candidates, upright and lying: one round, one square."""
        fits = apply_one_of_each([c.fit for c in connectors])
        for connector, fit in zip(connectors, fits):
            connector.fit = fit
            connector.kind = fit.kind
            if not connector.lying:
                connector.yaw = fit.yaw
        return connectors

    def _lying_connectors(self, heights, depth, table, observation) -> list[ConnectorEstimate]:
        cfg = self.cfg
        calib = observation.calibration
        hsv = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2HSV)
        lo, hi = cfg.fixture_band
        fixture = self._clean((heights > table + lo) & (heights < table + hi))
        fixture = cv2.morphologyEx(fixture, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fixture)
        exclude = np.isin(labels, [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= cfg.min_fixture_px])
        exclude = cv2.dilate(exclude.astype(np.uint8), np.ones((cfg.fixture_exclusion_px,) * 2, np.uint8)).astype(bool)
        tall = (heights > table + cfg.connector_min_height).astype(np.uint8)
        exclude |= cv2.dilate(tall, np.ones((cfg.upright_exclusion_px,) * 2, np.uint8)).astype(bool)
        band_lo, band_hi = cfg.lying_band
        low = (heights > table + band_lo) & (heights < table + band_hi) & ~exclude
        low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
        found = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < cfg.min_lying_px:
                continue
            mask = labels == label
            points = calib.back_project(depth, mask)
            xy = points[:, :2]
            mean = xy.mean(axis=0)
            _, vectors = np.linalg.eigh(np.cov((xy - mean).T))
            axis = vectors[:, 1]
            across = np.array([-axis[1], axis[0]])
            u, v = (xy - mean) @ axis, (xy - mean) @ across
            u_lo, u_hi = np.percentile(u, [0.5, 99.5])
            v_lo, v_hi = np.percentile(v, [2, 98])
            centre_xy = mean + axis * (u_lo + u_hi) / 2 + across * (v_lo + v_hi) / 2

            # Marked end: blue band pixels on or just around the part; orient the axis from them toward the top.
            near = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            hue, sat = hsv[..., 0], hsv[..., 1]
            band = near & (hue >= cfg.band_hue[0]) & (hue <= cfg.band_hue[1]) & (sat >= cfg.band_min_saturation)
            band &= np.isfinite(depth)
            tip_found = int(band.sum()) >= cfg.min_band_px
            if tip_found:
                band_u = (calib.back_project(depth, band)[:, :2] - centre_xy) @ axis
                if np.median(band_u) > 0:
                    axis, across = -axis, -across

            # Local table height (a ring around the part) removes calibration tilt from the cross-section heights.
            m8 = mask.astype(np.uint8)
            ring = cv2.dilate(m8, np.ones((31, 31), np.uint8)).astype(bool) & ~cv2.dilate(m8, np.ones((11, 11), np.uint8)).astype(bool)
            ring &= np.isfinite(heights) & (np.abs(heights - table) < 0.005)
            local_table = float(np.median(heights[ring])) if ring.any() else table
            central = np.abs(u - (u_lo + u_hi) / 2) < cfg.lying_central_half_length
            z = points[central, 2] - local_table
            vc = v[central] - (v_lo + v_hi) / 2
            w = cfg.lying_drop_half_width
            inner = np.abs(vc) < w
            if inner.sum() < 50:
                continue
            curvature = np.polyfit(vc[inner], z[inner], 2)[0]
            drop = float(-curvature * w * w)
            ridge = float(np.median(z[np.abs(vc) < w / 2]))
            width = float(v_hi - v_lo)
            margins = [(split[0] - value) / split[1] for value, split in (
                (drop, cfg.lying_drop_split), (width, cfg.lying_width_split), (ridge, cfg.lying_ridge_split))]
            score = cfg.lying_score_scale * float(np.clip(margins, -3, 3).mean())
            kind = "square" if score > 0 else "round"
            fit = ShapeFit(kind, centre_xy, 0.0, float("nan"), float("nan"), width, int(mask.sum()), score=score)

            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boundary = max(contours, key=cv2.contourArea)[:, 0, :]
            outline = self._pixels_on_plane(boundary, local_table + width / 2, calib)
            found.append(ConnectorEstimate(
                kind, np.array([*centre_xy, local_table + ridge]), math.atan2(axis[1], axis[0]), fit, outline,
                lying=True, axis_xy=axis, tip_found=tip_found, centre=np.array([*centre_xy, local_table]), length=float(u_hi - u_lo),
            ))
        return found

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
            contours, _ = cv2.findContours(face.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boundary = max(contours, key=cv2.contourArea)[:, 0, :]
            # Pixel-centre boundary lies half a pixel inside the true edge; the outline keeps that small bias.
            outline = self._pixels_on_plane(boundary, top_z, calib)
            found.append((fit_footprint(xy, pixel_area, cfg), top_z, outline))
        return [ConnectorEstimate(fit.kind, np.array([*fit.centre_xy, z]), fit.yaw, fit, outline) for fit, z, outline in found]

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
