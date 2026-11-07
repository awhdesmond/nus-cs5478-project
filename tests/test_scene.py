"""Step 1: the scene."""

import math
import mujoco
import pytest
import numpy as np

from sim.config import SceneConfig
from sim.scene import build_model
from sim.ur5e import UR5E_JOINTS, UR5E_TORQUE_LIMITS
from sim.scene import port_hole_half, port_top_height


@pytest.fixture
def model():
    return build_model(SceneConfig())


def start_at_home(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    return data


def test_arm_has_six_torque_motors_at_the_real_limits(model):
    assert model.nu == 6
    for i, limit in enumerate(UR5E_TORQUE_LIMITS):
        np.testing.assert_allclose(model.actuator_ctrlrange[i], [-limit, limit])
        # A pure motor: gain 1 on ctrl, and no position/velocity terms (a servo would have them).
        assert model.actuator_gainprm[i, 0] == 1.0
        assert not model.actuator_biasprm[i].any()
    assert model.opt.timestep == 0.001


def test_arm_falls_without_torque(model):
    """Zero torque → nothing holds the arm up. Proves the servos are really gone."""
    data = start_at_home(model)
    home = data.qpos.copy()
    for _ in range(500):                  # 0.5 s
        mujoco.mj_step(model, data)
    assert np.abs(data.qpos - home).max() > 0.1   # sagged by more than ~6°


def test_gravity_compensation_plus_pd_holds_home(model):
    """τ = qfrc_bias + Kp·(q* − q) − Kd·q̇ holds the pose: the controller pattern used from now on."""
    data = start_at_home(model)
    q_home = data.qpos[:6].copy()         # only arm joints exist so far; later we look up by name
    kp = np.array([400, 400, 400, 100, 100, 50.0])   # stiffer on the big joints
    kd = np.array([40, 40, 40, 10, 10, 5.0])         # damping ≈ kp/10 stops oscillation
    for _ in range(1000):                 # 1 s
        q, qd = data.qpos[:6], data.qvel[:6]
        # qfrc_bias = torque needed to cancel gravity (+ Coriolis) at the current state.
        # MuJoCo recomputes it on every step; we just add it.
        data.ctrl[:] = data.qfrc_bias[:6] + kp * (q_home - q) - kd * qd
        mujoco.mj_step(model, data)
    assert np.abs(data.qpos[:6] - q_home).max() < 1e-3   # within 1 mrad: effectively still


def height_below(model, data, x, y):
    """Height of the first surface straight down from (x, y, 0.5 m): a laser rangefinder.

    geomgroup keeps only group-0 geoms (table, ports, floor), so the arm's visual meshes (group 2)
    can't block the ray when the arm happens to be overhead.
    """
    geomid = np.zeros(1, dtype=np.int32)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    distance = mujoco.mj_ray(model, data, np.array([x, y, 0.5]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
    return 0.5 - distance


@pytest.mark.parametrize("clearance", [0.001, 0.003])
def test_each_hole_is_open_to_its_floor_and_its_rim_is_flat(clearance):
    cfg = SceneConfig(clearance=clearance)
    model = build_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    top = port_top_height(cfg)
    floor = cfg.table_height + cfg.port_base
    for kind, pose in cfg.port_poses.items():
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)

        def world(u, v):
            """Port-frame (u, v) → world (x, y)."""
            return pose.x + c * u - s * v, pose.y + s * u + c * v

        h = port_hole_half(cfg, kind)
        # Inside the hole: the ray falls to the floor (centre, and 90% of the way to the wall).
        assert height_below(model, data, *world(0, 0)) == pytest.approx(floor, abs=1e-6)
        assert height_below(model, data, *world(0.9 * h, 0)) == pytest.approx(floor, abs=1e-6)
        assert height_below(model, data, *world(0, -0.9 * h)) == pytest.approx(floor, abs=1e-6)
        # On the rim, halfway between the chamfer and the block's outer edge, all the way round: flat top.
        # Square hole: the 8 points around it (sides and corners). Round hole: one point per polygon face.
        m = h + cfg.port_chamfer + (cfg.port_wall - cfg.port_chamfer) / 2
        if kind == "square":
            rim = [(u, v) for u in (-m, 0, m) for v in (-m, 0, m) if (u, v) != (0, 0)]
        else:
            phis = 2 * math.pi * np.arange(cfg.round_sides) / cfg.round_sides
            rim = [(m * math.cos(phi), m * math.sin(phi)) for phi in phis]
        for u, v in rim:
            assert height_below(model, data, *world(u, v)) == pytest.approx(top, abs=1e-6)


def test_ports_sit_at_their_configured_poses():
    cfg = SceneConfig()
    model = build_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for kind, pose in cfg.port_poses.items():
        np.testing.assert_allclose(data.site(f"{kind}_port_entrance").xpos, [pose.x, pose.y, port_top_height(cfg)],
                                   atol=1e-12)
        rot = data.site(f"{kind}_port_bottom").xmat.reshape(3, 3)
        assert math.atan2(rot[1, 0], rot[0, 0]) == pytest.approx(pose.yaw)

# --- 1c: pegs and seeded layouts -------------------------------------------------------------

from sim.config import PEGS
from sim.layout import ORIENTATIONS, peg_axis_segment, sample_layout, segment_distance


def test_segment_distance_known_cases():
    p = lambda x, y: np.array([x, y], dtype=float)
    assert segment_distance(p(0, 0), p(1, 0), p(0, 1), p(1, 1)) == pytest.approx(1.0)   # parallel, 1 apart
    assert segment_distance(p(0, 0), p(2, 2), p(0, 2), p(2, 0)) == 0.0                 # crossing
    assert segment_distance(p(0, 0), p(1, 0), p(2, 0), p(2, 0)) == pytest.approx(1.0)   # segment to a point
    assert segment_distance(p(0, 0), p(0, 0), p(3, 4), p(3, 4)) == pytest.approx(5.0)   # point to point


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_layouts_are_seeded_in_bounds_and_spaced(orientation):
    cfg = SceneConfig()
    pick_lo = np.array(cfg.pick_area_center) - cfg.pick_area_half
    pick_hi = np.array(cfg.pick_area_center) + cfg.pick_area_half
    port_lo = np.array(cfg.port_area_center) - cfg.port_area_half
    port_hi = np.array(cfg.port_area_center) + cfg.port_area_half
    lying_seen = set()
    for seed in range(40):
        layout = sample_layout(cfg, seed, orientation)
        assert layout == sample_layout(cfg, seed, orientation)  # same seed, same layout
        segments = [peg_axis_segment(cfg, layout.peg_poses[k]) for k in PEGS]
        for seg in segments:
            for end in seg:
                assert np.all(end >= pick_lo - 1e-12) and np.all(end <= pick_hi + 1e-12)
        assert segment_distance(*segments[0], *segments[1]) >= cfg.min_peg_spacing
        ports = [layout.port_poses[k] for k in PEGS]
        for pose in ports:
            assert np.all(np.array([pose.x, pose.y]) >= port_lo) and np.all(np.array([pose.x, pose.y]) <= port_hi)
        assert math.dist((ports[0].x, ports[0].y), (ports[1].x, ports[1].y)) >= cfg.min_port_spacing
        lying_seen |= {layout.peg_poses[k].lying for k in PEGS}
    assert lying_seen == {"upright": {False}, "mixed": {False, True}, "lying": {True}}[orientation]


def test_peg_tips_start_where_the_layout_says():
    cfg = sample_layout(SceneConfig(), 3, "lying")
    model = build_model(cfg)
    data = start_at_home(model)
    mujoco.mj_forward(model, data)
    for kind in PEGS:
        pose = cfg.peg_poses[kind]
        tip = data.site(f"{kind}_tip").xpos
        # Lying: the marked end is half a length *behind* the centre along the yaw direction.
        expected = np.array([pose.x, pose.y]) - cfg.peg_length / 2 * np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
        np.testing.assert_allclose(tip[:2], expected, atol=1e-9)


def test_home_keyframe_still_puts_the_arm_at_home(model):
    """The rebuilt keyframe keeps the menagerie's arm angles."""
    from sim.ur5e import load_stock_ur5e
    stock = load_stock_ur5e()
    np.testing.assert_allclose(model.key("home").qpos[:6], stock.key("home").qpos)


@pytest.mark.parametrize("orientation", ["upright", "mixed", "lying"])
def test_pegs_rest_still(orientation):
    """Pegs dropped into the scene settle and stay put for 2 s, touching nothing but the table."""
    cfg = sample_layout(SceneConfig(), 1, orientation)
    model = build_model(cfg)
    data = start_at_home(model)
    q_home = data.qpos[:6].copy()
    kp = np.array([400, 400, 400, 100, 100, 50.0])
    kd = np.array([40, 40, 40, 10, 10, 5.0])
    mujoco.mj_forward(model, data)
    start = {k: data.body(f"{k}_peg").xpos.copy() for k in PEGS}
    for _ in range(2000):
        data.ctrl[:] = data.qfrc_bias[:6] + kp * (q_home - data.qpos[:6]) - kd * data.qvel[:6]
        mujoco.mj_step(model, data)
    for kind in PEGS:
        assert np.linalg.norm(data.body(f"{kind}_peg").xpos - start[kind]) < 1e-4   # moved less than 0.1 mm
    touching = {tuple(sorted((model.geom(c.geom1).name, model.geom(c.geom2).name))) for c in data.contact[:data.ncon]}
    assert touching <= {("round_peg", "table"), ("square_peg", "table")}, touching
