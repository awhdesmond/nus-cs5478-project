"""Step 2: perception. Tests are grouped by sub-step."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from perception.camera import CameraModel
from sim.camera import OverheadCamera
from sim.config import PEGS, SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim


# --- 2a: camera observation --------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene():
    """One layout with the arm at ready, and a noise-free frame of it."""
    cfg = sample_layout(SceneConfig(), 0)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    with OverheadCamera(sim.model, cfg) as camera:
        obs = camera.capture(sim.data)
    return sim, obs


def test_project_and_backproject_are_inverses():
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # some rotation (90° about z)
    cam = CameraModel.from_fovy(640, 480, 30.0, [0.1, 0.2, 1.0], rot)
    points = np.random.default_rng(0).uniform([-0.2, 0, 0], [0.3, 0.4, 0.5], (50, 3))
    u, v, depth = cam.project(points)
    np.testing.assert_allclose(cam.backproject(u, v, depth), points, atol=1e-12)


def test_camera_model_matches_the_config(scene):
    sim, obs = scene
    cam, cfg = obs.camera, sim.cfg
    assert obs.rgb.shape == (cfg.image_height, cfg.image_width, 3) and obs.rgb.dtype == np.uint8
    assert obs.depth.shape == (cfg.image_height, cfg.image_width)
    np.testing.assert_allclose(cam.position, cfg.camera_pos)
    # The top edge of the image sees fovy/2 off the axis: here 12.5°.
    u, v, _ = cam.project(np.array(cfg.camera_pos) + [0, np.tan(np.radians(12.5)), -1])
    assert v == pytest.approx(-0.5) and u == pytest.approx(cfg.image_width / 2 - 0.5)


def geom_ids(sim):
    """Ground truth for tests: which geom each overhead-camera pixel shows (-1 = none)."""
    cfg = sim.cfg
    with mujoco.Renderer(sim.model, cfg.image_height, cfg.image_width) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="overhead")
        seg = renderer.render()
    return np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)


def test_depth_puts_the_table_at_table_height(scene):
    sim, obs = scene
    table = geom_ids(sim) == sim.model.geom("table").id   # the pixels that show the table
    assert table.mean() > 0.8                             # most of the image
    z = obs.points()[table, 2]
    assert np.abs(z - sim.cfg.table_height).max() < 1e-5  # exact, to float32 rounding (~2 µm)


def test_peg_tops_back_project_onto_their_true_position(scene):
    sim, obs = scene
    for kind in PEGS:
        top = sim.data.site(f"{kind}_top").xpos           # ground truth: centre of the peg's top face
        u, v, _ = obs.camera.project(top)
        iu, iv = round(u), round(v)                       # the pixel it lands in
        seen = obs.camera.backproject(iu, iv, obs.depth[iv, iu])
        # Same pixel, so within half a pixel (~0.17 mm) sideways, and the top face's height exactly.
        assert np.linalg.norm(seen[:2] - top[:2]) < 0.0003 and abs(seen[2] - top[2]) < 1e-4


def test_depth_noise_is_seeded_and_has_the_configured_spread(scene):
    sim, clean = scene
    cfg = replace(sim.cfg, depth_noise=0.002)
    with OverheadCamera(sim.model, cfg) as camera:
        a = camera.capture(sim.data, np.random.default_rng(7))
        b = camera.capture(sim.data, np.random.default_rng(7))
    np.testing.assert_array_equal(a.depth, b.depth)       # same seed, same noise
    np.testing.assert_array_equal(a.rgb, clean.rgb)       # colour is untouched
    assert np.std(a.depth - clean.depth) == pytest.approx(0.002, rel=0.02)

# --- 2b: segmentation ----------------------------------------------------------------------------

from perception.segment import segment


def body_ids(sim):
    """Ground truth for tests: which body each pixel shows (-1 = none)."""
    geoms = geom_ids(sim)
    return np.where(geoms >= 0, sim.model.geom_bodyid[np.maximum(geoms, 0)], -1)


@pytest.mark.parametrize("noise", [0.0, 0.002])
def test_segmentation_finds_two_pegs_and_two_blocks(noise):
    for seed in range(5):
        cfg = replace(sample_layout(SceneConfig(), seed), depth_noise=noise)
        sim = Sim(build_model(cfg), cfg)
        sim.reset("ready")
        with OverheadCamera(sim.model, cfg) as camera:
            seg = segment(camera.capture(sim.data, np.random.default_rng(seed)))
        assert abs(seg.table_z - cfg.table_height) < 0.0003       # table found to 0.3 mm
        assert len(seg.of_kind("peg")) == 2 and len(seg.of_kind("block")) == 2, (seed, seg.regions)

        bodies = body_ids(sim)
        for region in seg.regions:
            # The ground-truth body this region mostly covers, and how well they agree.
            ids, n = np.unique(bodies[region.mask], return_counts=True)
            body = ids[np.argmax(n)]
            name = sim.model.body(body).name
            assert name.endswith("_peg" if region.kind == "peg" else "_port"), (seed, name, region.kind)
            truth = bodies == body
            # Almost all of the object's pixels are found; the bottom 6 mm of a peg's side is below threshold.
            assert (region.mask & truth).sum() / truth.sum() > 0.9, (seed, name)
            assert (region.mask & ~truth).sum() / region.mask.sum() < 0.02, (seed, name)  # few strays
            nominal = sim.cfg.peg_length if region.kind == "peg" else sim.cfg.port_base + sim.cfg.port_depth
            assert abs(region.top - nominal) < 0.0005, (seed, name, region.top)        # top within 0.5 mm

# --- 2c: pegs --------------------------------------------------------------------------------------

import math

from perception.pegs import estimate_pegs, wrap_quarter_turn


def test_on_plane_finds_points_of_known_height():
    cam = CameraModel.from_fovy(640, 480, 30.0, [0.1, 0.2, 1.0], np.eye(3))  # looking straight down
    points = np.random.default_rng(1).uniform([-0.1, 0, 0.3], [0.3, 0.4, 0.3], (20, 3))  # all at z = 0.3
    u, v, _ = cam.project(points)
    np.testing.assert_allclose(cam.on_plane(u, v, 0.3), points, atol=1e-12)


def test_wrap_quarter_turn():
    for angle, wrapped in [(0.0, 0.0), (math.pi / 2, 0.0), (math.radians(50), math.radians(-40)),
                           (math.radians(-46), math.radians(44)), (math.pi / 4, -math.pi / 4)]:
        assert wrap_quarter_turn(angle) == pytest.approx(wrapped)


@pytest.mark.parametrize("noise", [0.0, 0.002])
def test_pegs_are_identified_and_located(noise):
    for seed in range(8):
        cfg = replace(sample_layout(SceneConfig(), seed), depth_noise=noise)
        sim = Sim(build_model(cfg), cfg)
        sim.reset("ready")
        with OverheadCamera(sim.model, cfg) as camera:
            obs = camera.capture(sim.data, np.random.default_rng(seed))
        pegs = estimate_pegs(obs, segment(obs))
        for kind in PEGS:
            est, top = pegs[kind], sim.data.site(f"{kind}_top").xpos   # estimate vs ground truth
            # Right kind (else this would be the other peg, centimetres away) and where it really is.
            assert np.linalg.norm(est.top[:2] - top[:2]) < 0.0003, (seed, kind)        # 0.3 mm
            assert abs(est.top[2] - top[2]) < 0.0005, (seed, kind)                     # 0.5 mm
            true_size = 2 * cfg.round_radius if kind == "round" else 2 * cfg.square_half
            assert abs(est.size - true_size) < 0.0003, (seed, kind, est.size)
        error = wrap_quarter_turn(pegs["square"].yaw - cfg.peg_poses["square"].yaw)
        assert abs(error) < math.radians(1.5), (seed, math.degrees(error))
        # Clear of the threshold, so the "one of each" fallback isn't what got it right.
        assert pegs["round"].fill < 0.84 and pegs["square"].fill > 0.88, seed

# --- 2d: ports -------------------------------------------------------------------------------------

from perception.pegs import square_yaw
from perception.ports import estimate_ports


def test_square_yaw_of_an_ideal_square():
    for yaw in np.radians([-44, -20, 0, 10, 30, 44]):
        corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * 0.004            # 8 mm square, yaw 0
        t = np.linspace(0, 1, 50, endpoint=False)[:, None]
        outline = np.vstack([a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        centre = np.array([0.1, 0.5])
        assert square_yaw(outline @ turn.T + centre, centre) == pytest.approx(yaw, abs=1e-9)


@pytest.mark.parametrize("noise", [0.0, 0.002])
def test_ports_are_identified_and_located(noise):
    for seed in range(8):
        cfg = replace(sample_layout(SceneConfig(), seed), depth_noise=noise)
        sim = Sim(build_model(cfg), cfg)
        sim.reset("ready")
        with OverheadCamera(sim.model, cfg) as camera:
            obs = camera.capture(sim.data, np.random.default_rng(seed))
        ports = estimate_ports(obs, segment(obs))
        for kind in PEGS:
            est, entrance = ports[kind], sim.data.site(f"{kind}_port_entrance").xpos   # vs ground truth
            assert np.linalg.norm(est.entrance[:2] - entrance[:2]) < 0.0003, (seed, kind)   # 0.3 mm
            assert abs(est.entrance[2] - entrance[2]) < 0.0005, (seed, kind)                # 0.5 mm
        error = wrap_quarter_turn(ports["square"].yaw - cfg.port_poses["square"].yaw)
        assert abs(error) < math.radians(1.0), (seed, math.degrees(error))
        assert ports["round"].fill < 0.85 and ports["square"].fill > 0.93, seed       # far from the 0.89 split

# --- 2e: perceive and the perception check ---------------------------------------------------------

import pandas as pd

from harness.perception_check import check_layout, summarise


def test_check_layout_logs_small_errors_for_every_object():
    cfg = replace(sample_layout(SceneConfig(), 11), depth_noise=0.002)
    rows = check_layout(cfg, 11)
    assert {(r["object"], r["kind"]) for r in rows} == {(o, k) for o in ("peg", "port") for k in PEGS}
    for r in rows:
        assert r["found"] and r["xy_error"] < 0.0003 and abs(r["z_error"]) < 0.0005, r
        assert (r["kind"] == "round") == math.isnan(r["yaw_error"]), r   # yaw only for square objects
    table = summarise(pd.DataFrame(rows))
    assert set(table.index) == {(0.002, "peg"), (0.002, "port")} and (table.found == 1).all()
