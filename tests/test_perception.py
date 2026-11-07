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
