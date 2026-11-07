"""Open the scene in MuJoCo's interactive viewer, with the arm holding its home pose.

    uv run python scripts/view.py

The arm is torque-controlled, so without a controller it would fall. A control callback runs
gravity compensation + PD every physics step, the same law as in tests/test_scene.py.
"""

import mujoco
import mujoco.viewer
import numpy as np

from sim.config import SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model

KP = np.array([400, 400, 400, 100, 100, 50.0])
KD = np.array([40, 40, 40, 10, 10, 5.0])


def main():
    model = build_model(sample_layout(SceneConfig(), seed=4))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    q_home = data.qpos[:6].copy()

    def hold_home(m, d):
        # MuJoCo calls this before each step; whatever we put in d.ctrl is applied this step.
        d.ctrl[:6] = d.qfrc_bias[:6] + KP * (q_home - d.qpos[:6]) - KD * d.qvel[:6]

    mujoco.set_mjcb_control(hold_home)
    mujoco.viewer.launch(model, data)  # blocks until you close the window
    mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
