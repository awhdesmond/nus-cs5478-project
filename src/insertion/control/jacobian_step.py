"""Week 1 Cartesian motion: iterate Δq = J⁺·Δx toward a target EE position.

Grows into full damped-least-squares IK in week 2 (orientation error, damping, joint limits).
"""

import mujoco
import numpy as np


def jacobian_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    """Return a joint-position increment moving `site_id` toward `target_pos`.

    TODO (week 1): mj_jacSite → jacp (3×nv); dq = pinv(jacp) @ (gain * (target - site_xpos)).
    """
    raise NotImplementedError
