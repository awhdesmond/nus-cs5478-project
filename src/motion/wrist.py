"""The wrist force/torque sensor, as a controller uses it: tared, in the world frame, lightly filtered."""

import numpy as np

from motion.kinematics import Kinematics
from sim.sim import Sim


class WristFT:
    """Force the environment exerts on the tool, in world coordinates [N].

    Raw readings include the weight of everything below the sensor (gripper, fingers, a held peg), so they're
    tared (zeroed) in free space just before contact is expected. MuJoCo's sensor reports what the gripper
    exerts on the flange, in the sensor's frame; the environment's push on the tool is minus that, rotated to
    the world. A first-order low-pass filter takes off the step-to-step contact jitter.
    """

    def __init__(self, sim: Sim, kin: Kinematics, time_constant: float = 0.005):
        self.sim, self.kin = sim, kin
        self.alpha = sim.model.opt.timestep / (time_constant + sim.model.opt.timestep)  # filter gain per step
        self.offset = np.zeros(3)
        self.filtered = np.zeros(3)
        self.raw = np.zeros(3)  # this step's tared, unfiltered reading


    def _raw_world(self) -> np.ndarray:
        _, R = self.kin.fk(self.sim.q())  # the sensor's frame turns with the tool (same orientation as the tcp)
        return -R @ self.sim.ft_raw()[:3]

    def tare(self) -> None:
        """Zero the reading here: call in free space, after holding still for a moment. (The sensor reads 0
        until the first physics step, so a tare straight after reset would be off by the gripper's weight.)"""
        self.offset = self._raw_world()
        self.filtered = np.zeros(3)

    def update(self) -> np.ndarray:
        """Call once per control step; returns the filtered, tared force [N]."""
        self.raw = self._raw_world() - self.offset
        self.filtered += self.alpha * (self.raw - self.filtered)
        return self.filtered.copy()

