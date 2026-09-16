"""Overhead RGB-D camera: rendering, calibration and back-projection.

Conventions (verified against ray casts): the depth image holds distance along the optical axis
(not ray length); image row 0 is the top of the view; MuJoCo cameras look along their −z with +y up,
so pixel (u, v) at depth z maps to camera-frame ((u + ½ − c_x)·z/f, −(v + ½ − c_y)·z/f, −z).
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class CameraCalibration:
    width: int
    height: int
    focal: float  # [px], square pixels
    position: np.ndarray  # camera origin, world [m]
    rotation: np.ndarray  # camera-to-world rotation (3×3)

    @classmethod
    def from_model(cls, model: mujoco.MjModel, data: mujoco.MjData, camera: str, width: int, height: int) -> "CameraCalibration":
        cam = model.camera(camera)
        focal = (height / 2) / np.tan(np.radians(model.cam_fovy[cam.id]) / 2)
        return cls(width, height, float(focal), data.cam(camera).xpos.copy(), data.cam(camera).xmat.reshape(3, 3).copy())

    def back_project(self, depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """World points (N×3) for the pixels in `mask` (all pixels if None), in row-major order."""
        rows, cols = np.nonzero(np.ones_like(depth, dtype=bool) if mask is None else mask)
        z = depth[rows, cols]
        cam = np.stack([
            (cols + 0.5 - self.width / 2) * z / self.focal,
            -(rows + 0.5 - self.height / 2) * z / self.focal,
            -z,
        ], axis=1)
        return cam @ self.rotation.T + self.position

    def height_map(self, depth: np.ndarray) -> np.ndarray:
        """World z of every pixel (H×W)."""
        return self.back_project(depth)[:, 2].reshape(depth.shape)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel coordinates (N×2, as u, v) of world points."""
        cam = (np.atleast_2d(points) - self.position) @ self.rotation
        z = -cam[:, 2]
        return np.stack([cam[:, 0] * self.focal / z + self.width / 2 - 0.5, -cam[:, 1] * self.focal / z + self.height / 2 - 0.5], axis=1)


@dataclass
class CameraObservation:
    """What a controller receives from the camera: images plus the (known) calibration."""

    rgb: np.ndarray  # H×W×3 uint8
    depth: np.ndarray  # H×W float, distance along the optical axis [m]
    calibration: CameraCalibration


@dataclass
class DepthNoise:
    """Sensor noise model (name kept for compatibility; covers both RGB and depth)."""

    sigma: float = 0.0  # [m] per-pixel Gaussian depth noise
    quantization: float = 0.0  # [m] depth resolution; 0 disables
    rgb_sigma: float = 0.0  # [intensity levels, 0–255] per-pixel, per-channel Gaussian colour noise
    # Extrinsic calibration error: the calibration handed to perception is off from the true camera pose by a
    # fixed (per camera instance) random offset, uniform in ±calib_translation per axis and a rotation of up to
    # calib_rotation_deg about a random axis. Biases every estimate the same way, like a real hand-eye calibration.
    calib_translation: float = 0.0  # [m]
    calib_rotation_deg: float = 0.0


class OverheadCamera:
    def __init__(self, model: mujoco.MjModel, width: int, height: int, name: str = "overhead",
                 noise: DepthNoise | None = None, rng: np.random.Generator | None = None):
        self.model = model
        self.name = name
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        self.calib_offset = (np.zeros(3), np.eye(3))
        if noise is not None and (noise.calib_translation > 0 or noise.calib_rotation_deg > 0):
            rng = rng or np.random.default_rng()
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = np.radians(rng.uniform(-noise.calib_rotation_deg, noise.calib_rotation_deg))
            quat = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            self.calib_offset = (rng.uniform(-noise.calib_translation, noise.calib_translation, size=3), rot.reshape(3, 3))

    def capture(self, data: mujoco.MjData, noise: DepthNoise | None = None, rng: np.random.Generator | None = None) -> CameraObservation:
        self.renderer.update_scene(data, camera=self.name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.name)
        depth = self.renderer.render().astype(np.float64)
        self.renderer.disable_depth_rendering()
        if noise is not None and noise.sigma > 0:
            rng = rng or np.random.default_rng()
            depth = depth + rng.normal(0.0, noise.sigma, size=depth.shape)
        if noise is not None and noise.quantization > 0:
            depth = np.round(depth / noise.quantization) * noise.quantization
        if noise is not None and noise.rgb_sigma > 0:
            rng = rng or np.random.default_rng()
            rgb = np.clip(rgb.astype(np.float64) + rng.normal(0.0, noise.rgb_sigma, size=rgb.shape), 0, 255).astype(np.uint8)
        calibration = CameraCalibration.from_model(self.model, data, self.name, self.width, self.height)
        offset, rot = self.calib_offset
        calibration.position = calibration.position + offset
        calibration.rotation = rot @ calibration.rotation
        return CameraObservation(rgb=rgb, depth=depth, calibration=calibration)

    def segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """Ground-truth geom id per pixel (for evaluation only, never for controllers)."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(data, camera=self.name)
        seg = self.renderer.render()[..., 0].copy()
        self.renderer.disable_segmentation_rendering()
        return seg
