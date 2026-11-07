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
    assert model.nu == 7 and model.actuator(6).name == "gripper"  # 6 arm motors + gripper

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
    """τ = qfrc_bias + Kp·(q* - q) - Kd·q̇ holds the pose: the controller pattern used from now on."""
    data = start_at_home(model)
    q_home = data.qpos[:6].copy()         # only arm joints exist so far; later we look up by name
    kp = np.array([400, 400, 400, 100, 100, 50.0])   # stiffer on the big joints
    kd = np.array([40, 40, 40, 10, 10, 5.0])         # damping ≈ kp/10 stops oscillation
    for _ in range(1000):                 # 1 s
        q, qd = data.qpos[:6], data.qvel[:6]
        # qfrc_bias = torque needed to cancel gravity (+ Coriolis) at the current state.
        # MuJoCo recomputes it on every step; we just add it.
        data.ctrl[:6] = data.qfrc_bias[:6] + kp * (q_home - q) - kd * qd
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
from sim.layout import sample_layout


def test_layouts_are_seeded_in_bounds_and_spaced():
    cfg = SceneConfig()
    areas = {"peg": (cfg.pick_area_center, cfg.pick_area_half, cfg.min_peg_spacing),
             "port": (cfg.port_area_center, cfg.port_area_half, cfg.min_port_spacing)}
    for seed in range(40):
        layout = sample_layout(cfg, seed)
        assert layout == sample_layout(cfg, seed)  # same seed, same layout
        for what, poses in (("peg", layout.peg_poses), ("port", layout.port_poses)):
            center, half, spacing = areas[what]
            for pose in poses.values():
                assert np.all(np.abs(np.array([pose.x, pose.y]) - center) <= half)   # inside its area
            a, b = poses.values()
            assert math.dist((a.x, a.y), (b.x, b.y)) >= spacing                      # far enough apart


def test_pegs_stand_upright_where_the_layout_says():
    cfg = sample_layout(SceneConfig(), 3)
    model = build_model(cfg)
    data = start_at_home(model)
    mujoco.mj_forward(model, data)
    for kind in PEGS:
        pose = cfg.peg_poses[kind]
        np.testing.assert_allclose(data.site(f"{kind}_tip").xpos, [pose.x, pose.y, cfg.table_height], atol=1e-9)
        np.testing.assert_allclose(data.site(f"{kind}_top").xpos[2], cfg.table_height + cfg.peg_length, atol=1e-9)


def test_home_keyframe_still_puts_the_arm_at_home(model):
    """The rebuilt keyframe keeps the menagerie's arm angles."""
    from sim.ur5e import load_stock_ur5e
    stock = load_stock_ur5e()
    np.testing.assert_allclose(model.key("home").qpos[:6], stock.key("home").qpos)


def test_pegs_rest_still():
    """Pegs dropped into the scene settle and stay put for 2 s, touching nothing but the table."""
    cfg = sample_layout(SceneConfig(), 1)
    model = build_model(cfg)
    data = start_at_home(model)
    q_home = data.qpos[:6].copy()
    kp = np.array([400, 400, 400, 100, 100, 50.0])
    kd = np.array([40, 40, 40, 10, 10, 5.0])
    mujoco.mj_forward(model, data)
    start = {k: data.body(f"{k}_peg").xpos.copy() for k in PEGS}
    for _ in range(2000):
        data.ctrl[:6] = data.qfrc_bias[:6] + kp * (q_home - data.qpos[:6]) - kd * data.qvel[:6]
        mujoco.mj_step(model, data)
    for kind in PEGS:
        assert np.linalg.norm(data.body(f"{kind}_peg").xpos - start[kind]) < 1e-4   # moved less than 0.1 mm
    touching = {tuple(sorted((model.geom(c.geom1).name, model.geom(c.geom2).name))) for c in data.contact[:data.ncon]}
    assert touching <= {("round_peg", "table"), ("square_peg", "table")}, touching


# --- 1d: gripper, attach-on-grasp, wrist F/T -------------------------------------------------

from sim.sim import Sim

KP = np.array([400, 400, 400, 100, 100, 50.0])
KD = np.array([40, 40, 40, 10, 10, 5.0])


def hold(sim, q_hold, seconds):
    """Keep the arm still at q_hold (gravity compensation + PD) while the world runs."""
    for _ in range(round(seconds / sim.model.opt.timestep)):
        sim.step(sim.gravity_comp() + KP * (q_hold - sim.q()) - KD * sim.qd())


def peg_in_gripper_frame(sim, kind):
    """Where the peg's centre is, seen from the gripper: constant while the peg is held rigidly."""
    g = sim.data.body("gripper")
    return g.xmat.reshape(3, 3).T @ (sim.data.body(f"{kind}_peg").xpos - g.xpos)


def put_peg_between_open_fingers(sim, kind):
    """Teleport a peg so it sits centred between the pads, lined up with the gripper (tests only: no IK yet)."""
    joint = sim.model.joint(f"{kind}_peg")
    tcp = sim.data.site("tcp")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, tcp.xmat)
    sim.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [*tcp.xpos, *quat]
    sim.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0
    # Start the fingers 0.1 mm short of the peg, so it can't fall far before they close on it.
    width = 2 * (sim.cfg.round_radius if kind == "round" else sim.cfg.square_half)
    almost = (sim.cfg.finger_open_gap - width) / 2 - 0.0001
    for adr in sim.finger_qpos:
        sim.data.qpos[adr] = almost
    sim.data.ctrl[sim.gripper_act] = almost
    mujoco.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize("kind", PEGS)
def test_gripper_grasps_holds_and_releases_a_peg(kind):
    sim = Sim(build_model(SceneConfig()), SceneConfig())
    sim.reset()
    q_home = sim.q()
    put_peg_between_open_fingers(sim, kind)
    assert sim.attached() is None

    sim.command_gripper(True)
    hold(sim, q_home, 0.1)
    assert sim.attached() == kind                       # both pads touched → welded
    width = 2 * (sim.cfg.round_radius if kind == "round" else sim.cfg.square_half)
    assert sim.finger_gap() == pytest.approx(width, abs=0.3e-3)   # pads stopped on the peg's sides

    held = peg_in_gripper_frame(sim, kind)
    hold(sim, q_home, 1.0)
    assert np.linalg.norm(peg_in_gripper_frame(sim, kind) - held) < 1e-4   # no slip over 1 s: < 0.1 mm

    sim.command_gripper(False)
    z = sim.data.body(f"{kind}_peg").xpos[2]
    hold(sim, q_home, 0.3)
    assert sim.attached() is None
    assert sim.data.body(f"{kind}_peg").xpos[2] < z - 0.05   # released: it fell


def test_ft_sensor_reads_the_gripper_weight_at_rest():
    """At rest the wrist carries only the gripper, so the sensor reads its weight (mass x g)."""
    sim = Sim(build_model(SceneConfig()), SceneConfig())
    sim.reset()
    hold(sim, sim.q(), 0.5)
    weight = sim.model.body_subtreemass[sim.gripper_body] * 9.81   # gripper + fingers [N]
    assert np.linalg.norm(sim.ft_raw()[:3]) == pytest.approx(weight, rel=0.01)


# --- 1e: overhead camera and ready pose ------------------------------------------------------


def test_ready_pose_puts_the_tool_down_at_ready_tcp():
    cfg = SceneConfig()
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    tcp = sim.data.site("tcp")
    np.testing.assert_allclose(tcp.xpos, cfg.ready_tcp, atol=1e-4)            # within 0.1 mm
    # "Tool down": the tcp's z axis (out of the flange) points along world -z.
    np.testing.assert_allclose(tcp.xmat.reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-3)


def test_camera_matches_the_proposal():
    cfg = SceneConfig()
    model = build_model(cfg)
    cam = model.camera("overhead")
    assert model.cam_fovy[cam.id] == 25.0
    assert cfg.camera_pos[2] - cfg.table_height == pytest.approx(0.75)
    assert model.vis.global_.offwidth >= 1280 and model.vis.global_.offheight >= 960
    assert not model.light_castshadow.any()


def robot_geoms(model):
    """Every geom on the robot: bodies from the arm's base down to the fingertips."""
    base = model.body("base").id

    def on_robot(body):
        while body > 0:  # walk up the body tree; body 0 is the world
            if body == base:
                return True
            body = model.body_parentid[body]
        return False

    return [g for g in range(model.ngeom) if on_robot(model.geom_bodyid[g])]


@pytest.mark.parametrize("seed", range(6))
def test_from_ready_the_camera_sees_both_pegs_and_ports_whole_and_no_arm(seed):
    cfg = sample_layout(SceneConfig(), seed)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    model = sim.model
    # Segmentation rendering: each pixel holds (object id, object type) instead of a colour.
    with mujoco.Renderer(model, 240, 320) as renderer:   # low resolution keeps the test fast
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="overhead")
        seg = renderer.render()
    geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)  # geom id per pixel, -1 if none
    border = np.concatenate([geom[0], geom[-1], geom[:, 0], geom[:, -1]])

    assert not np.isin(geom, robot_geoms(model)).any()   # not a single arm or gripper pixel
    for kind in PEGS:
        peg = [model.geom(f"{kind}_peg").id]
        port = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(f"{kind}_port").id]
        for obj in (peg, port):
            assert np.isin(geom, obj).any()               # visible...
            assert not np.isin(border, obj).any()         # ...and not cut off by the image edge


# --- 1f: ground truth and the week-1 checks --------------------------------------------------

from dataclasses import replace

from harness.scene_check import keying_check, rest_check


def teleport_peg(sim, kind, tip_at):
    """Stand a peg upright with its insertion end at `tip_at`."""
    joint = sim.model.joint(f"{kind}_peg")
    centre = np.array(tip_at) + [0, 0, sim.cfg.peg_length / 2]
    sim.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [*centre, 1, 0, 0, 0]
    mujoco.mj_forward(sim.model, sim.data)


def test_ground_truth_tells_seated_and_wrong_port_apart():
    cfg = replace(SceneConfig(), clearance=0.003)   # at 3 mm the round peg fits the square hole
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")
    floor = lambda port: sim.data.site(f"{port}_port_bottom").xpos + [0, 0, 0.0005]
    teleport_peg(sim, "round", floor("round"))
    assert sim.seated("round") and sim.in_wrong_port("round") is None
    teleport_peg(sim, "round", floor("square"))
    assert not sim.seated("round") and sim.in_wrong_port("round") == "square"
    teleport_peg(sim, "round", floor("round") + [0, 0, 0.005])   # 5 mm short of the floor
    assert not sim.seated("round") and sim.in_wrong_port("round") is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scene_rests_stably(seed):
    result = rest_check(sample_layout(SceneConfig(), seed))
    assert result.passed, result


def test_each_peg_fits_only_its_own_port_at_1mm():
    """The week-1 deliverable: at 1 mm clearance, own port seats; the other blocks it at the rim."""
    for r in keying_check(SceneConfig()):
        assert r.passed, r
        assert r.entered == (r.peg == r.port), r
