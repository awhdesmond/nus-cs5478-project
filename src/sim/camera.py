"""Renders the overhead camera into a CameraObservation"""

import mujoco
import numpy as np

from perception.camera import CameraModel, CameraObservation
from sim.config import SceneConfig

CAMERA = "overhead"


class OverheadCamera:
    """Renders RGB and depth from the `overhead` camera, adding depth noise if cfg.depth_noise > 0.

    Holds two MuJoCo renderers (one draws colour, one depth).
    """

    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig):
        self.cfg = cfg
        self._rgb = mujoco.Renderer(model, cfg.image_height, cfg.image_width)
        self._depth = mujoco.Renderer(model, cfg.image_height, cfg.image_width)
        self._depth.enable_depth_rendering()  # render() now returns depth in metres instead of colour

    def model(self, data: mujoco.MjData) -> CameraModel:
        """The calibration. The camera is fixed, so this is what a one-off real calibration would measure."""
        cam = data.camera(CAMERA)
        fovy = self._rgb.model.camera(CAMERA).fovy[0]
        return CameraModel.from_fovy(self.cfg.image_width, self.cfg.image_height, fovy, cam.xpos.copy(),
                                     cam.xmat.reshape(3, 3).copy())

    def capture(self, data: mujoco.MjData, rng: np.random.Generator | None = None) -> CameraObservation:
        """One RGB-D frame of the current state. `rng` draws the depth noise (needed if depth_noise > 0)."""
        self._rgb.update_scene(data, camera=CAMERA)
        rgb = self._rgb.render()
        self._depth.update_scene(data, camera=CAMERA)
        depth = self._depth.render().astype(np.float32)
        if self.cfg.depth_noise > 0:
            depth += rng.normal(0.0, self.cfg.depth_noise, depth.shape).astype(np.float32)
        return CameraObservation(rgb, depth, self.model(data))

    def close(self) -> None:
        self._rgb.close()
        self._depth.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
