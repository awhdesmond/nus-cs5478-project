from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
UR5E_DIR = ROOT / "third_party" / "mujoco_menagerie" / "universal_robots_ur5e"
UR5E_SCENE_XML = UR5E_DIR / "scene.xml"  # arm + floor + lights

# Joint names in the model's own order (base → wrist). Later code looks joints up by these
# names rather than by index, so adding bodies (board, parts) can't shift them.
UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def load_stock_ur5e() -> mujoco.MjModel:
    """The unmodified menagerie scene: UR5e with its position servos, on a floor."""
    if not UR5E_SCENE_XML.exists():
        raise FileNotFoundError(f"{UR5E_SCENE_XML} is missing: run ./scripts/fetch_assets.sh")
    return mujoco.MjModel.from_xml_path(str(UR5E_SCENE_XML))
