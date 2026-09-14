"""Thin wrapper around MjModel/MjData: stepping, state and sensor readout."""

import mujoco
import numpy as np


class Sim:
    """TODO (week 1):
      - reset(keyframe="home")
      - step(ctrl) → mj_step
      - ee_pose() from the peg tip site
      - ft() tared + low-pass filtered wrist force/torque
      - contact_force_total() summed over peg–socket contacts (mj_contactForce)
      - gravity_comp() → data.qfrc_bias for joint PD hold
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)

    @property
    def time(self) -> float:
        return self.data.time

    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()
