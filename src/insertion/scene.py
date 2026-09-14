"""Build the MuJoCo model for a trial.

The scene is generated in code (via `mujoco.MjSpec`) rather than a static XML, because
clearance, socket pose and friction are swept by the evaluation harness.
"""

import mujoco

from insertion.config import UR5E_SCENE_XML, SceneConfig


def load_stock_ur5e() -> mujoco.MjModel:
    """Unmodified menagerie UR5e + floor. Used by the smoke test."""
    return mujoco.MjModel.from_xml_path(str(UR5E_SCENE_XML))


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    """Week 1 scene: UR5e with torque actuators, peg welded to flange, box-geom socket.

    TODO (week 1):
      - Load UR5E_SCENE_XML into an MjSpec
      - Replace the position-servo `general` actuators with torque `motor`s
        (ctrlrange = forcerange: ±150 Nm size3 joints, ±28 Nm wrist joints)
      - Attach a round peg body at `attachment_site` on wrist_3_link
      - Add force/torque sensors on a site between flange and peg
      - Build the socket from box geoms with hole width = 2 * peg_radius + clearance
      - Apply cfg.timestep, cfg.friction, elliptic cone
    """
    raise NotImplementedError
