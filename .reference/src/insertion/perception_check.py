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
    squareness: float  # IoU(square) − IoU(circle) of the fitted footprint (lying: cross-section score)
    lying: bool = False  # true orientation (connectors)
    orientation_correct: bool = True  # estimate's upright/lying matches
    axis_error: float = float("nan")  # [rad] lying: long-axis direction error (mod 180°)
    end_error: float = float("nan")  # [m] lying: worse of the two end positions (xy), marked end matched to marked end
    tip_correct: bool = True  # lying: the marked end was found and is the true one


def check_layout(seed: int, noise: DepthNoise, estimator: PoseEstimator | None = None, base: AssemblyConfig | None = None,
                 orientation: str = "upright") -> list[PerceptionRow]:
    cfg = sample_layout(base or AssemblyConfig(), seed, orientation=orientation)
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
        lying = (cfg.round_pose if kind == "round" else cfg.square_pose).lying
        body = data.body(f"{kind}_connector")
        rot = body.xmat.reshape(3, 3)
        if not lying:
            true_top = data.site(f"{kind}_top").xpos
            true_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
            rows.append(PerceptionRow(
                seed, noise.sigma, "connector", kind, est is not None, len(scene.connectors),
                float(np.linalg.norm(est.top[:2] - true_top[:2])) if est else float("nan"),
                float(est.top[2] - true_top[2]) if est else float("nan"),
                yaw_error_mod90(est.yaw, true_yaw) if est and kind == "square" and not est.lying else float("nan"),
                est.fit.squareness if est else float("nan"),
                lying=False, orientation_correct=est is not None and not est.lying,
            ))
            continue
        ends = [data.site(f"{kind}_tip").xpos[:2], data.site(f"{kind}_top").xpos[:2]]
        true_axis = rot[:2, 2]
        true_top_z = cfg.table_height + 2 * (cfg.round_radius if kind == "round" else cfg.square_half)
        row = PerceptionRow(seed, noise.sigma, "connector", kind, est is not None, len(scene.connectors),
                            float("nan"), float("nan"), float("nan"), est.fit.squareness if est else float("nan"),
                            lying=True, orientation_correct=est is not None and est.lying)
        if est is not None and est.lying:
            row.xy_error = float(np.linalg.norm(est.centre[:2] - body.xpos[:2]))
            row.z_error = float(est.top[2] - true_top_z)
            row.axis_error = float(np.arccos(min(1.0, abs(float(est.axis_xy @ true_axis)))))
            row.tip_correct = bool(est.tip_found and est.axis_xy @ true_axis > 0)
            est_ends = [est.centre[:2] + s * est.axis_xy * est.length / 2 for s in (-1, 1)]  # [marked end, top]
            row.end_error = max(np.linalg.norm(est_ends[i] - ends[i]) for i in (0, 1))
        else:
            row.tip_correct = False
        rows.append(row)
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
