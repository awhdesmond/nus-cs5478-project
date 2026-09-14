"""Open the interactive viewer on the stock UR5e (swap in build_model once the scene exists)."""

import mujoco
import mujoco.viewer

from insertion.scene import load_stock_ur5e


def main():
    model = load_stock_ur5e()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
