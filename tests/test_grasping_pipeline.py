import math

import mujoco
import numpy as np
import pytest

from insertion.assembly_scene import build_assembly_model, sample_layout
from insertion.assembly_skills import AssemblyRun, visual_seating
from insertion.camera import DepthNoise, OverheadCamera
from insertion.config import AssemblyConfig
from insertion.grasping import GraspConfig, propose_grasps
from insertion.perception import GeometricEstimator

CFG = AssemblyConfig()
STEP = 0.00034  # outline sampling ≈ camera resolution


def square_outline(half: float, yaw: float, centre=(0.1, 0.5)) -> np.ndarray:
    ts = np.arange(-half, half, STEP)
    sides = [np.stack([np.full_like(ts, half), ts], 1), np.stack([-ts, np.full_like(ts, half)], 1),
             np.stack([np.full_like(ts, -half), -ts], 1), np.stack([ts, np.full_like(ts, -half)], 1)]
    pts = np.concatenate(sides)
    c, s = math.cos(yaw), math.sin(yaw)
    return pts @ np.array([[c, -s], [s, c]]).T + centre


def circle_outline(radius: float, centre=(0.1, 0.5)) -> np.ndarray:
    angles = np.arange(0, 2 * math.pi, STEP / radius)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], 1) + centre


@pytest.mark.parametrize("yaw_deg", [0.0, 20.0, 41.0])
def test_square_grasps_close_across_flats(yaw_deg):
    yaw = math.radians(yaw_deg)
    proposals = propose_grasps(square_outline(CFG.square_half, yaw), 0.22, CFG)
    assert proposals
    for p in proposals[:3]:
        face_error = abs((p.closing_angle - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        assert math.degrees(face_error) < 1.0
        assert p.width == pytest.approx(2 * CFG.square_half, abs=0.0005)


def test_square_corner_grasps_are_rejected():
    proposals = propose_grasps(square_outline(CFG.square_half, 0.0), 0.22, CFG, GraspConfig(direction_step_deg=1.0))
    angles = {round(math.degrees(p.closing_angle)) % 90 for p in proposals}
    assert not any(30 <= a <= 60 for a in angles)


def test_round_connector_accepts_many_diameters():
    proposals = propose_grasps(circle_outline(CFG.round_radius), 0.22, CFG)
    assert len(proposals) > 30
    assert all(p.friction_margin_deg >= 0 for p in proposals)


def test_too_wide_objects_are_rejected():
    assert propose_grasps(circle_outline(0.02), 0.22, CFG) == []


def test_finger_keepout_around_other_objects():
    outline = circle_outline(CFG.round_radius, centre=(0.1, 0.5))
    obstacle = np.array([0.1 + 0.021, 0.5])  # right where a finger closing along x would land
    for p in propose_grasps(outline, 0.22, CFG, obstacles_xy=[obstacle]):
        assert abs(math.cos(p.closing_angle)) < 0.95


def test_tool_tip_site_orientation_changes_take_effect():
    model = build_assembly_model(CFG)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    site = model.site("tool_tip").id
    quat = np.array([math.cos(0.3), 0, 0, math.sin(0.3)])
    model.site_quat[site] = quat
    model.site_sameframe[site] = 0
    mujoco.mj_kinematics(model, data)
    gripper_rot = data.body("gripper").xmat.reshape(3, 3)
    expected = np.zeros(9)
    mujoco.mju_quat2Mat(expected, quat)
    np.testing.assert_allclose(data.site("tool_tip").xmat.reshape(3, 3), gripper_rot @ expected.reshape(3, 3), atol=1e-9)


def test_calibration_error_is_fixed_per_camera_and_applied():
    model = build_assembly_model(CFG)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    noise = DepthNoise(calib_translation=0.002, calib_rotation_deg=0.5)
    camera = OverheadCamera(model, 64, 48, noise=noise, rng=np.random.default_rng(3))
    a, b = camera.capture(data), camera.capture(data)
    np.testing.assert_array_equal(a.calibration.position, b.calibration.position)
    offset = a.calibration.position - data.cam("overhead").xpos
    assert 0 < np.abs(offset).max() <= 0.002


@pytest.mark.parametrize("condition", ["ideal", "realistic"])
def test_perception_driven_assembly_seats_both_connectors(condition):
    noise = DepthNoise() if condition == "ideal" else DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5)
    result = AssemblyRun(sample_layout(CFG, 5), seed=5, noise=noise).run()
    for log in result.connectors:
        assert log.identified and log.grasped and log.lifted, log
        assert log.other_disturbed_mm < 0.5
        assert log.inserted_belief and log.seated and log.stayed_seated and not log.wrong_socket, log
        assert log.peak_force_10ms < 30.0
        assert log.seated_visual and log.verified, log
    assert result.both_seated_at_end
    square = next(log for log in result.connectors if log.kind == "square")
    assert square.grasp_face_error_deg < 2.0
    if condition == "ideal":
        assert all(log.in_hand_tip_error_mm < 0.5 for log in result.connectors)


REALISTIC = DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5)


@pytest.mark.parametrize("orientation, noise, seed", [("lying", DepthNoise(), 0), ("mixed", REALISTIC, 4)])
def test_lying_connectors_are_turned_sideways_and_inserted_marked_end_first(orientation, noise, seed):
    cfg = sample_layout(CFG, seed, orientation=orientation)
    assert any(pose.lying for pose in (cfg.round_pose, cfg.square_pose))
    result = AssemblyRun(cfg, seed=seed, noise=noise).run()
    for log in result.connectors:
        assert log.identified and log.grasped and log.lifted, log
        assert log.seated and log.stayed_seated and log.verified and not log.wrong_end and not log.wrong_socket, log
        assert log.peak_force_10ms < 30.0
        if log.lying:
            assert log.sideways_clearance_mm >= 5.0
    assert result.both_seated_at_end


def test_marked_end_decides_which_way_a_lying_connector_points():
    cfg = sample_layout(CFG, 0, orientation="lying")
    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    scene = GeometricEstimator().estimate(OverheadCamera(model, cfg.image_width, cfg.image_height).capture(data))
    for kind in ("round", "square"):
        est = scene.connector(kind)
        assert est.lying and est.tip_found
        true_tip = data.site(f"{kind}_tip").xpos[:2]
        est_tip = est.centre[:2] - est.axis_xy * est.length / 2
        assert np.linalg.norm(est_tip - true_tip) < 0.001


def seated_scene(kind_states: dict, noise: DepthNoise, seed: int = 2):
    """Two captures by one camera from the ready pose: the layout as sampled (holes empty), then with each connector
    placed by hand "seated" in its socket or resting "on_chamfer"."""
    cfg = sample_layout(CFG, seed)
    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    camera = OverheadCamera(model, cfg.image_width, cfg.image_height, noise=noise, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    before = GeometricEstimator().estimate(camera.capture(data, noise, rng))
    for kind, state in kind_states.items():
        bottom = data.site(f"{kind}_socket_bottom").xpos
        lift = 0.0 if state == "seated" else CFG.socket_depth - CFG.socket_chamfer / 2
        adr = model.joint(f"{kind}_connector").qposadr[0]
        data.qpos[adr:adr + 3] = bottom + [0, 0, lift + CFG.connector_length / 2]
        data.qpos[adr + 3:adr + 7] = data.body(f"{kind}_socket").xquat
    mujoco.mj_forward(model, data)
    return cfg, before, GeometricEstimator().estimate(camera.capture(data, noise, rng))


@pytest.mark.parametrize("noise", [DepthNoise(), DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5)])
def test_visual_seating_check_tells_seated_from_resting_on_the_chamfer(noise):
    cfg, before, after = seated_scene({"round": "seated", "square": "on_chamfer"}, noise)
    seated, height, offset = visual_seating(after.connector("round"), before.socket("round"), cfg)
    assert seated and abs(height - 0.030) < 0.0015 and offset < 0.001
    seated, height, _ = visual_seating(after.connector("square"), before.socket("square"), cfg)
    assert not seated and height > 0.040
