"""Wrist force/torque signal conditioning: tare and first-order low-pass filter.

This is what a controller sees. It mirrors a real sensor driver: bias is removed with a
tare taken while still and out of contact, and high-frequency contact chatter is filtered.
"""

import math

import numpy as np

from insertion.config import SensorConfig
from insertion.sim import Sim


class FTSensor:
    def __init__(self, sim: Sim, cfg: SensorConfig):
        self.sim = sim
        self.cfg = cfg
        dt = sim.model.opt.timestep
        rc = 1.0 / (2 * math.pi * cfg.ft_cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.bias = np.zeros(6)
        # Tared + filtered EXTERNAL wrench on the peg, sensor frame. MuJoCo's force/torque sensors
        # report what the wrist exerts on the peg, which is the negative of what the environment
        # exerts once tared — so the sign is flipped: +z world = peg pushed up by the socket.
        self.wrench = np.zeros(6)

    def tare(self, samples: list[np.ndarray]) -> None:
        """Set bias to the mean of raw readings taken while still and out of contact."""
        self.bias = np.mean(samples, axis=0)
        self.wrench = np.zeros(6)

    def update(self) -> np.ndarray:
        """Call once per sim step after mj_step."""
        external = -(self.sim.ft_raw() - self.bias)
        self.wrench += self.alpha * (external - self.wrench)
        return self.wrench

    def force_world(self) -> np.ndarray:
        """Filtered external force on the peg, world frame."""
        _, rot = self.sim.site_pose("ft_site")
        return rot @ self.wrench[:3]


class ContactDetector:
    """Hysteresis + debounce on filtered F/T force magnitude, so contact chatter doesn't toggle it."""

    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.in_contact = False
        self._count = 0

    def update(self, wrench: np.ndarray) -> bool:
        force = float(np.linalg.norm(wrench[:3]))
        if self.in_contact:
            self.in_contact = force > self.cfg.contact_off
            self._count = 0
        else:
            self._count = self._count + 1 if force > self.cfg.contact_on else 0
            self.in_contact = self._count >= self.cfg.contact_debounce
        return self.in_contact
