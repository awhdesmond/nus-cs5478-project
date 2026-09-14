import mujoco
import numpy as np

from insertion.scene import load_stock_ur5e


def test_stock_ur5e_holds_home_without_nans():
    model = load_stock_ur5e()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
