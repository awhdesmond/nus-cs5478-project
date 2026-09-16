"""Check the install: load the stock UR5e, hold the home pose for 2 s, report sim speed."""

import time

import mujoco
import numpy as np

from insertion.scene import load_stock_ur5e


def main():
    model = load_stock_ur5e()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    q_home = data.qpos.copy()

    n_steps = int(2.0 / model.opt.timestep)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    wall = time.perf_counter() - t0

    drift = np.abs(data.qpos - q_home).max()
    print(f"MuJoCo {mujoco.__version__}: nq={model.nq}, nu={model.nu}, timestep={model.opt.timestep}")
    print(f"Held home for {data.time:.2f} s sim, max joint drift {drift:.2e} rad")
    print(f"{n_steps / wall:,.0f} steps/s ({data.time / wall:.0f}× real time)")


if __name__ == "__main__":
    main()
