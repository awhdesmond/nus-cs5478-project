"""M2 perception check: estimator output vs ground truth over random connector layouts.

For each seed: sample a two-connector layout, render the overhead camera from the ready arm
pose (optionally with depth noise), run the estimator, and score identification and pose errors for
both connectors and both sockets.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from insertion.assembly_scene import CONNECTORS, build_assembly_model, sample_layout
from insertion.camera import DepthNoise, OverheadCamera
from insertion.config import AssemblyConfig
from insertion.perception import GeometricEstimator, PoseEstimator


def yaw_error_mod90(estimate: float, truth: float) -> float:
    """Smallest angle between two square orientations (90° symmetry) [rad]."""
    diff = (estimate - truth + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return abs(diff)


def socket_yaw(data: mujoco.MjData, kind: str) -> float:
    rot = data.body(f"{kind}_socket").xmat.reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


@dataclass
class PerceptionRow:
    seed: int
    noise_sigma: float
    part: str  # "connector" or "socket"
    kind: str  # true kind
    detected: bool  # exactly one estimate of this kind
    n_estimates: int  # total estimates of this part type (should be 2)
    xy_error: float  # [m]
    z_error: float  # [m]
    yaw_error: float  # [rad], square only (nan for round)
    squareness: float  # IoU(square) − IoU(circle) of the fitted footprint


def check_layout(seed: int, noise: DepthNoise, estimator: PoseEstimator | None = None, base: AssemblyConfig | None = None) -> list[PerceptionRow]:
    cfg = sample_layout(base or AssemblyConfig(), seed)
    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    camera = OverheadCamera(model, cfg.image_width, cfg.image_height)
    observation = camera.capture(data, noise, rng=np.random.default_rng([seed, 4]))
    scene = (estimator or GeometricEstimator()).estimate(observation)

    rows = []
    for kind in CONNECTORS:
        est = scene.connector(kind)
        true_top = data.site(f"{kind}_top").xpos
        rot = data.body(f"{kind}_connector").xmat.reshape(3, 3)
        true_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        rows.append(PerceptionRow(
            seed, noise.sigma, "connector", kind, est is not None, len(scene.connectors),
            float(np.linalg.norm(est.top[:2] - true_top[:2])) if est else float("nan"),
            float(est.top[2] - true_top[2]) if est else float("nan"),
            yaw_error_mod90(est.yaw, true_yaw) if est and kind == "square" else float("nan"),
            est.fit.squareness if est else float("nan"),
        ))
    for kind in CONNECTORS:
        est = scene.socket(kind)
        true_entrance = data.site(f"{kind}_socket_entrance").xpos
        rows.append(PerceptionRow(
            seed, noise.sigma, "socket", kind, est is not None, len(scene.sockets),
            float(np.linalg.norm(est.entrance[:2] - true_entrance[:2])) if est else float("nan"),
            float(est.entrance[2] - true_entrance[2]) if est else float("nan"),
            yaw_error_mod90(est.yaw, socket_yaw(data, kind)) if est and kind == "square" else float("nan"),
            est.fit.squareness if est else float("nan"),
        ))
    return rows
