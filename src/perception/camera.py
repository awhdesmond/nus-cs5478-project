from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """A calibrated pinhole camera: intrinsics (focal length, image centre) and its pose in the world.

    Camera frame (MuJoCo's convention): x right, y up in the image, looking along -z.
    Pixels: u to the right (column), v downward (row), pixel (0, 0) is the top-left one, and a pixel's
    centre is at +0.5.
    """

    width: int
    height: int
    focal: float  # [px] focal length; square pixels, so the same in u and v
    position: np.ndarray  # [m] camera centre in the world, (3,)
    rotation: np.ndarray  # camera-to-world rotation, (3, 3): its columns are the camera's x, y, z axes

    @staticmethod
    def from_fovy(width: int, height: int, fovy_deg: float, position, rotation) -> "CameraModel":
        # The vertical field of view spans height/2 pixels either side of the centre:
        # tan(fovy / 2) = (height / 2) / focal.
        focal = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        return CameraModel(width, height, float(focal), np.asarray(position, float), np.asarray(rotation, float))

    def backproject(self, u, v, depth) -> np.ndarray:
        """Pixels (u, v) with depths [m, along the viewing axis] -> world points, shape (..., 3)."""
        u, v, depth = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(depth, float))
        # Similar triangles: a pixel `du` from the centre at depth d is du * d / focal metres off the axis.
        x = (u + 0.5 - self.width / 2) * depth / self.focal
        y = -(v + 0.5 - self.height / 2) * depth / self.focal  # v grows downward, camera y upward
        local = np.stack([x, y, -depth], axis=-1)  # in front of the camera is -z
        return local @ self.rotation.T + self.position  # camera frame -> world

    def project(self, points) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points (..., 3) -> pixel coordinates u, v (floats) and depth [m]. Inverse of backproject."""
        local = (np.asarray(points, float) - self.position) @ self.rotation  # world -> camera frame
        depth = -local[..., 2]
        u = local[..., 0] * self.focal / depth + self.width / 2 - 0.5
        v = -local[..., 1] * self.focal / depth + self.height / 2 - 0.5
        return u, v, depth

    def on_plane(self, u, v, z: float) -> np.ndarray:
        """Where the rays through pixels (u, v) hit the horizontal plane at world height z, shape (..., 3).

        Unlike backproject, this needs no depth: for a surface whose height is known (a peg's top face),
        the pixel alone fixes the point, so depth noise can't move it sideways.
        """
        ray = self.backproject(u, v, 1.0) - self.position  # direction through each pixel
        t = (z - self.position[2]) / ray[..., 2]  # how far along the ray to reach height z
        return self.position + t[..., None] * ray


    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """u and v for every pixel, each (height, width), to back-project a whole depth image at once."""
        return np.meshgrid(np.arange(self.width), np.arange(self.height))


@dataclass(frozen=True)
class CameraObservation:
    rgb: np.ndarray  # (height, width, 3) uint8
    depth: np.ndarray  # (height, width) float32 [m] along the viewing axis, noise included
    camera: CameraModel

    def points(self) -> np.ndarray:
        """Every pixel back-projected to the world, (height, width, 3) [m]."""
        u, v = self.camera.pixel_grid()
        return self.camera.backproject(u, v, self.depth)
