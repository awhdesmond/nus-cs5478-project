import mujoco
import numpy as np
import pytest

from insertion.config import SceneConfig, TaskConfig
from insertion.scene import build_spec, contact_params, hole_width

CFG = SceneConfig(clearance=0.004, socket_chamfer=0.002)
H = hole_width(CFG) / 2
D = CFG.socket_depth
C = CFG.socket_chamfer


def surface_height(model, data, x, y):
    """Height (socket frame) of the first static surface hit by a downward ray at (x, y)."""
    origin = np.array(CFG.socket_pos) + np.array([x, y, 0.1])
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, origin, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    assert dist >= 0, f"ray at ({x}, {y}) hit nothing"
    return 0.1 - dist


@pytest.fixture(scope="module")
def scene():
    model = build_spec(CFG).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.0),  # hole centre → hole bottom
        (H - 0.0005, 0.0),  # just inside the hole edge
        (H + C * 0.25, D - C * 0.75),  # on the chamfer slope
        (H + C * 0.75, D - C * 0.25),
        (H + C + 0.002, D),  # top face beyond the chamfer
    ],
)
def test_surface_profile_across_wall(scene, x, expected):
    for sx, sy in ((x, 0.0), (-x, 0.0), (0.0, x), (0.0, -x)):
        assert surface_height(*scene, sx, sy) == pytest.approx(expected, abs=1e-6)


def test_chamfer_ring_has_no_corner_notch(scene):
    # Inside the corner square the surface should sit at or above the chamfer slope.
    for f in (0.25, 0.5, 0.75):
        z = surface_height(*scene, H + f * C, H + f * C)
        assert z >= D - C + f * C - 1e-6


def test_zero_chamfer_gives_square_edge():
    cfg = SceneConfig(clearance=0.004, socket_chamfer=0.0)
    model = build_spec(cfg).compile()
    assert not any(model.geom(i).name.startswith("socket_chamfer") for i in range(model.ngeom))


def test_rejects_chamfer_larger_than_wall():
    with pytest.raises(ValueError):
        build_spec(SceneConfig(socket_chamfer=0.010))


def drop_peg(radius, start_height, mass=None, push=0.0, seconds=1.0):
    """Release a free upright cylinder with its bottom at `start_height` (socket frame).

    Returns (lowest, final) bottom heights. `push` is a constant downward force [N].
    """
    spec = build_spec(CFG)
    half_len = 0.015
    start = np.array(CFG.socket_pos) + [0.0, 0.0, start_height + half_len]
    peg = spec.worldbody.add_body(name="test_peg", pos=start.tolist())
    peg.add_freejoint()
    geom_kwargs = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, half_len], **contact_params(CFG))
    if mass is not None:
        geom_kwargs["mass"] = mass
    peg.add_geom(**geom_kwargs)
    model = spec.compile()
    data = mujoco.MjData(model)
    body_id = model.body("test_peg").id
    lowest = np.inf
    for _ in range(int(seconds / model.opt.timestep)):
        data.xfrc_applied[body_id, 2] = -push
        mujoco.mj_step(model, data)
        lowest = min(lowest, data.body("test_peg").xpos[2])
    final = data.body("test_peg").xpos[2]
    assert np.isfinite(final)
    offset = half_len + CFG.socket_pos[2]
    return lowest - offset, final - offset


def test_fitting_peg_drops_to_bottom():
    _, final = drop_peg(CFG.peg_radius, start_height=D + 0.01)
    assert final == pytest.approx(0.0, abs=1e-4)


def test_oversized_peg_does_not_enter():
    _, final = drop_peg(H + 0.001, start_height=D + 0.01)
    assert final > D - C


def test_force_cap_push_on_chamfer_lip_stays_within_per_side_gap():
    # Effective end-effector mass ~2 kg pressing at the force cap onto the chamfer lip.
    # Sinking must stay below the per-side gap at the tightest target clearance (1 mm → 0.5 mm),
    # otherwise the sim could "insert" by penetration rather than alignment.
    radius = H + 0.001
    rest = D - C + (radius - H)
    lowest, _ = drop_peg(radius, start_height=rest + 0.0005, mass=2.0, push=TaskConfig().force_cap, seconds=1.5)
    assert rest - lowest < 0.0005
