import math

import mujoco
import numpy as np
import pytest

from insertion.assembly_scene import build_assembly_model
from insertion.camera import DepthNoise, OverheadCamera
from insertion.config import AssemblyConfig
from insertion.perception import GeometricEstimatorConfig, apply_one_of_each, fit_footprint
from insertion.perception_check import check_layout, yaw_error_mod90

CFG = AssemblyConfig()
PIXEL = 0.00034  # [m] roughly the camera's resolution on the table


def grid_points(inside) -> np.ndarray:
    xs = np.arange(-0.008, 0.008, PIXEL)
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return pts[inside(pts)] + [0.12, 0.47]


def test_camera_back_projection_matches_ray_casts():
    model = build_assembly_model(CFG)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("camera_clear").id)
    mujoco.mj_forward(model, data)
    obs = OverheadCamera(model, 320, 240).capture(data)
    calib = obs.calibration
    for u, v in ((160, 120), (10, 15), (300, 220)):
        mask = np.zeros_like(obs.depth, dtype=bool)
        mask[v, u] = True
        point = calib.back_project(obs.depth, mask)[0]
        direction = point - calib.position
        geom = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, calib.position, direction / np.linalg.norm(direction), None, 1, -1, geom)
        np.testing.assert_allclose(distance, np.linalg.norm(direction), atol=1e-4)
        np.testing.assert_allclose(calib.project(point)[0], [u, v], atol=1e-6)


@pytest.mark.parametrize("yaw_deg", [0.0, 17.0, 38.0, -30.0])
def test_fit_footprint_identifies_a_square_and_its_yaw(yaw_deg):
    yaw = math.radians(yaw_deg)
    half = 0.00425
    c, s = math.cos(yaw), math.sin(yaw)
    pts = grid_points(lambda p: (np.abs(p[:, 0] * c + p[:, 1] * s) <= half) & (np.abs(-p[:, 0] * s + p[:, 1] * c) <= half))
    fit = fit_footprint(pts, PIXEL**2, GeometricEstimatorConfig())
    assert fit.kind == "square"
    assert math.degrees(yaw_error_mod90(fit.yaw, yaw)) < 1.5
    np.testing.assert_allclose(fit.centre_xy, [0.12, 0.47], atol=PIXEL)
    assert fit.size == pytest.approx(2 * half, rel=0.05)


def test_fit_footprint_identifies_a_circle():
    fit = fit_footprint(grid_points(lambda p: np.hypot(p[:, 0], p[:, 1]) <= 0.005), PIXEL**2, GeometricEstimatorConfig())
    assert fit.kind == "round" and fit.squareness < 0
    assert fit.size == pytest.approx(0.010, rel=0.05)


def test_one_of_each_prior_labels_the_more_square_like_candidate_square():
    cfg = GeometricEstimatorConfig()
    a = fit_footprint(grid_points(lambda p: np.hypot(p[:, 0], p[:, 1]) <= 0.005), PIXEL**2, cfg)
    b = fit_footprint(grid_points(lambda p: (np.abs(p[:, 0]) <= 0.00425) & (np.abs(p[:, 1]) <= 0.00425)), PIXEL**2, cfg)
    forced = apply_one_of_each([a, b])
    assert [f.kind for f in forced] == ["round", "square"]
    same = apply_one_of_each([b, b])  # two squares measured: still exactly one of each, by squareness order
    assert sorted(f.kind for f in same) == ["round", "square"]


@pytest.mark.parametrize("noise", [DepthNoise(), DepthNoise(sigma=0.001, rgb_sigma=3.0)], ids=["clean", "noisy"])
def test_estimator_meets_m2_targets_on_random_layouts(noise):
    rows = [row for seed in range(8) for row in check_layout(seed, noise)]
    assert all(row.detected and row.n_estimates == 2 for row in rows)
    assert max(row.xy_error for row in rows) < 0.001
    square_yaw = [row.yaw_error for row in rows if row.kind == "square"]
    assert math.degrees(max(square_yaw)) < 2.0
