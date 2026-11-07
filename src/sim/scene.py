"""Build the MuJoCo model for a trial.

The scene is generated in code (MjSpec) rather than written as a static XML file, because
clearance, friction and the layout change from trial to trial.
"""

import mujoco

from sim.config import SceneConfig
from sim.ur5e import load_ur5e_spec


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    """The scene before compiling, so later parts (and tests) can keep adding to it."""
    spec = load_ur5e_spec()
    spec.option.timestep = cfg.timestep
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    return build_spec(cfg).compile()
