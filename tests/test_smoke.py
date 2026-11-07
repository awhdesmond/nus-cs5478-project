"""MuJoCo loads and steps the stock UR5e, and offscreen rendering
works (the overhead camera depends on it)."""

import mujoco
import numpy as np

from sim.ur5e import UR5E_JOINTS, load_stock_ur5e


def test_stock_ur5e_loads_with_six_joints_and_a_home_pose():
    model = load_stock_ur5e()
    assert [model.joint(i).name for i in range(model.njnt)] == list(UR5E_JOINTS)
    assert model.nu == 6
    assert model.key("home").id >= 0


def test_stock_ur5e_holds_home_for_a_second():
    model = load_stock_ur5e()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.qpos.copy()
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    # The menagerie servos sag slightly under gravity (~8 mrad), nothing more.
    assert np.abs(data.qpos - home).max() < 0.02


def test_offscreen_rendering_works():
    model = load_stock_ur5e()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, 64, 64) as renderer:
        renderer.update_scene(data)
        image = renderer.render()
    assert image.shape == (64, 64, 3) and image.std() > 0
