import math

import mujoco
import numpy as np
import pytest

from insertion.assembly_check import scripted_assembly, tool_down
from insertion.assembly_scene import (
    CONNECTORS,
    build_assembly_model,
    build_assembly_spec,
    sample_layout,
    socket_apothem,
)
from insertion.assembly_sim import AssemblySim
from insertion.config import AssemblyConfig, JointPDConfig
from insertion.control.ik import JacobianIK
from insertion.control.joint_pd import joint_pd_torque
from insertion.scene import UR5E_JOINTS, contact_params

CFG = AssemblyConfig()


@pytest.fixture(scope="module")
def model():
    return build_assembly_model(CFG)


# --- scene structure -------------------------------------------------------------------------------


def test_scene_has_expected_parts(model):
    assert model.nu == 7 and model.actuator(6).name == "gripper"
    assert {model.key(i).name for i in range(model.nkey)} == {"home", "ready"}
    assert model.camera("overhead").id >= 0
    for kind in CONNECTORS:
        assert model.body(f"{kind}_connector").jntnum == 1
        assert model.eq_active0[model.equality(f"attach_{kind}").id] == 0
        np.testing.assert_allclose(model.eq_data[model.equality(f"attach_{kind}").id, :3], 0.0)


def test_nothing_moves_or_collides_at_rest(model):
    sim = AssemblySim(model, CFG)
    sim.reset()
    start = {k: sim.data.body(f"{k}_connector").xpos.copy() for k in CONNECTORS}
    q0 = sim.q()
    for _ in range(1000):
        sim.step(joint_pd_torque(sim, q0, JointPDConfig()))
    for k in CONNECTORS:
        assert np.linalg.norm(sim.data.body(f"{k}_connector").xpos - start[k]) < 1e-4
    pairs = {(model.geom(c.geom1).name, model.geom(c.geom2).name) for c in sim.data.contact[: sim.data.ncon]}
    assert all("connector" in a + b and "table" in a + b for a, b in pairs), pairs


def test_ready_pose_is_tool_down_at_the_configured_tcp(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site("tcp").xpos, CFG.ready_tcp, atol=1e-4)
    np.testing.assert_allclose(data.site("tcp").xmat.reshape(3, 3), tool_down(0.0), atol=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_camera_sees_no_arm_and_every_whole_object_from_the_ready_pose(seed):
    cfg = sample_layout(CFG, seed)
    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 240, 320)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="overhead")
    seg = renderer.render()[..., 0]
    arm = [g for g in range(model.ngeom) if model.body("base").id <= model.geom_bodyid[g] <= model.body("finger_right").id]
    assert not np.isin(seg, arm).any()
    border = np.concatenate([seg[0], seg[-1], seg[:, 0], seg[:, -1]])
    for kind in CONNECTORS:
        assert (seg == model.geom(f"{kind}_connector").id).sum() > 0
        socket_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(f"{kind}_socket").id]
        assert np.isin(seg, socket_geoms).any()
        assert not np.isin(border, socket_geoms + [model.geom(f"{kind}_connector").id]).any()  # not cut off


def test_layout_sampler_is_seeded_spaced_and_in_bounds():
    a, b = sample_layout(CFG, 4), sample_layout(CFG, 4)
    assert a.round_pose == b.round_pose and a.square_pose == b.square_pose
    for seed in range(50):
        layout = sample_layout(CFG, seed)
        r, s = layout.round_pose, layout.square_pose
        assert math.dist((r.x, r.y), (s.x, s.y)) >= CFG.min_connector_spacing
        for pose in (r, s):
            assert abs(pose.x - CFG.pick_area_center[0]) <= CFG.pick_area_half[0]
            assert abs(pose.y - CFG.pick_area_center[1]) <= CFG.pick_area_half[1]
        r, s = layout.round_socket_pose, layout.square_socket_pose
        assert math.dist((r.x, r.y), (s.x, s.y)) >= CFG.min_socket_spacing
        for pose in (r, s):
            assert abs(pose.x - CFG.socket_area_center[0]) <= CFG.socket_area_half[0]
            assert abs(pose.y - CFG.socket_area_center[1]) <= CFG.socket_area_half[1]
    fixed = sample_layout(CFG, 4, sockets=False)
    assert fixed.round_pose == a.round_pose and fixed.round_socket_pose == CFG.round_socket_pose


# --- round (N-gon) socket geometry ---------------------------------------------------------------


def surface_height(model, data, x, y, bottom_z):
    origin = np.array([x, y, bottom_z + 0.1])
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, origin, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    assert dist >= 0
    return 0.1 - dist


@pytest.mark.parametrize("face", [0, 3, 7, 12])
def test_round_socket_profile_across_a_face(model, face):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottom = data.site("round_socket_bottom").xpos
    R, d, c = socket_apothem(CFG, "round"), CFG.socket_depth, CFG.socket_chamfer
    phi = 2 * math.pi * face / CFG.round_sides
    direction = np.array([math.cos(phi), math.sin(phi)])
    for radius, expected in ((0.9 * R, 0.0), (R + 0.25 * c, d - 0.75 * c), (R + 0.75 * c, d - 0.25 * c), (R + c + 0.002, d)):
        x, y = bottom[:2] + radius * direction
        assert surface_height(model, data, x, y, bottom[2]) == pytest.approx(expected, abs=1e-6)


def test_round_socket_hole_is_open_out_to_the_polygon_vertices(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottom = data.site("round_socket_bottom").xpos
    R = socket_apothem(CFG, "round")
    vertex_radius = R / math.cos(math.pi / CFG.round_sides)
    for i in range(CFG.round_sides):
        phi = 2 * math.pi * (i + 0.5) / CFG.round_sides
        x, y = bottom[:2] + 0.98 * vertex_radius * np.array([math.cos(phi), math.sin(phi)])
        assert surface_height(model, data, x, y, bottom[2]) == pytest.approx(0.0, abs=1e-6)


# --- keyed fit and contact validity --------------------------------------------------------------


def push_into_socket(socket: str, shape: str, half: float, yaw: float = 0.0, push: float = 5.0, mass: float = 2.0,
                     seconds: float = 1.5, start_height: float | None = None):
    """Free test peg (effective mass `mass`) centred over a socket and pushed down.

    Starts with its tip at `start_height` above the hole bottom (default: 2 mm above the entrance).
    Returns (lowest, final) tip heights in the socket frame.
    """
    spec = build_assembly_spec(CFG)
    for key in list(spec.keys):
        spec.delete(key)
    base = build_assembly_model(CFG)
    base_data = mujoco.MjData(base)
    mujoco.mj_forward(base, base_data)
    entrance = base_data.site(f"{socket}_socket_entrance").xpos
    bottom_z = base_data.site(f"{socket}_socket_bottom").xpos[2]
    half_len = 0.015
    start = CFG.socket_depth + 0.002 if start_height is None else start_height
    body = spec.worldbody.add_body(
        name="probe", pos=[entrance[0], entrance[1], bottom_z + start + half_len],
        quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
    )
    body.add_freejoint()
    geom = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[half, half_len]) if shape == "round" else dict(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, half_len])
    body.add_geom(mass=mass, **geom, **contact_params(CFG))
    m = spec.compile()
    d = mujoco.MjData(m)
    arm_qpos = [m.joint(j).qposadr[0] for j in UR5E_JOINTS]
    ready = base.key("ready").qpos[[base.joint(j).qposadr[0] for j in UR5E_JOINTS]]
    d.qpos[arm_qpos] = ready
    arm_dofs = [m.joint(j).dofadr[0] for j in UR5E_JOINTS]
    probe_id = m.body("probe").id
    lowest = np.inf
    for _ in range(int(seconds / m.opt.timestep)):
        d.qvel[arm_dofs] = 0.0
        d.qpos[arm_qpos] = ready
        d.xfrc_applied[probe_id, 2] = -push
        mujoco.mj_step(m, d)
        lowest = min(lowest, d.body("probe").xpos[2] - half_len - bottom_z)
    return lowest, d.body("probe").xpos[2] - half_len - bottom_z


@pytest.mark.parametrize("kind", CONNECTORS)
def test_each_connector_seats_in_its_own_socket(kind):
    half = CFG.round_radius if kind == "round" else CFG.square_half
    _, final = push_into_socket(kind, kind, half)
    assert final == pytest.approx(0.0, abs=2e-4)


def test_round_connector_cannot_enter_square_socket():
    lowest, _ = push_into_socket("square", "round", CFG.round_radius, push=30.0)
    assert lowest > CFG.socket_depth - CFG.socket_chamfer - 0.001


def test_square_connector_cannot_enter_round_socket_even_corners_on_vertices():
    # Square corners at 45° + yaw; polygon vertices at (i + ½)·360°/N — align one corner (and so all four) with vertices.
    yaw = math.pi / CFG.round_sides - math.pi / 4
    lowest, _ = push_into_socket("round", "square", CFG.square_half, yaw=yaw, push=30.0)
    assert lowest > CFG.socket_depth - CFG.socket_chamfer - 0.001


def test_round_socket_chamfer_lip_penetration_stays_within_the_per_side_gap():
    R, c = socket_apothem(CFG, "round"), CFG.socket_chamfer
    radius = R + 0.001
    rest = CFG.socket_depth - c + (radius - R)
    lowest, _ = push_into_socket("round", "round", radius, push=30.0, start_height=rest + 0.0005)
    assert rest - lowest < 5e-4


# --- gripper and attach-on-grasp -----------------------------------------------------------------


def test_attach_on_grasp_holds_without_a_jump_and_releases(model):
    sim = AssemblySim(model, CFG)
    sim.reset()
    ik = JacobianIK(model, "tcp")
    conn = sim.data.body("round_connector").xpos.copy()
    grasp = conn + [0, 0, CFG.connector_length / 2 - CFG.grasp_below_top]
    q_grasp, ok = ik.solve_with_restarts(model.key("home").qpos[ik.qpos_adr], grasp, tool_down(0.0), rng=np.random.default_rng(0))
    assert ok
    sim.data.qpos[sim.qpos_adr] = q_grasp
    sim.data.qvel[:] = 0
    mujoco.mj_forward(model, sim.data)

    pd = JointPDConfig()
    sim.command_gripper(close=True)
    for _ in range(1200):  # fingers close at the 20 mm/s speed limit
        sim.step(joint_pd_torque(sim, q_grasp, pd))
    assert sim.attached() == "round"

    def conn_in_gripper():
        g = sim.data.body("gripper")
        return g.xmat.reshape(3, 3).T @ (sim.data.body("round_connector").xpos - g.xpos)

    before = conn_in_gripper()
    q_lift, _ = ik.solve(q_grasp, grasp + [0, 0, 0.08], tool_down(0.0))
    for _ in range(1500):
        sim.step(joint_pd_torque(sim, q_lift, pd))
    assert sim.data.body("round_connector").xpos[2] > conn[2] + 0.07
    np.testing.assert_allclose(conn_in_gripper(), before, atol=2e-4)

    sim.command_gripper(close=False)
    for _ in range(1200):
        sim.step(joint_pd_torque(sim, q_lift, pd))
    assert sim.attached() is None and sim.finger_gap() > CFG.finger_open_gap - 1e-3


# --- end to end ------------------------------------------------------------------------------------


def test_scripted_ground_truth_assembly_seats_both_connectors_round_first():
    result = scripted_assembly()
    assert [s.connector for s in result.stages] == ["round", "square"]
    assert all(s.grasped and s.seated and s.released and not s.wrong_socket for s in result.stages)
    assert result.both_seated_at_end
    assert result.max_other_connector_drift < 1e-3
