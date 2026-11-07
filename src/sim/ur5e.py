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


"""
The menagerie arm uses position servos: you command a joint angle
and a built-in spring pulls the joint there. The proposal needs
joint-torque control, because impedance control (step 5) has to decide
the force at the tool itself. So we delete the servos and add plain motors,
where ctrl is the joint torque in N·m. The catch is that the arm now falls
under gravity unless we add the torque that cancels it.
"""

# Real UR5e joint torque limits [N·m]: the three big joints, then the three wrist joints.
# These match the menagerie's own `forcerange`, so we don't invent numbers.
UR5E_TORQUE_LIMITS = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)


def use_torque_actuators(spec: mujoco.MjSpec) -> None:
    """Replace the menagerie position servos with direct joint-torque motors.

    Afterwards ctrl[i] is joint i's torque [N·m], so the controller owns gravity compensation
    and compliance (needed for impedance control in step 5).
    """
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    for joint, limit in zip(UR5E_JOINTS, UR5E_TORQUE_LIMITS):
        spec.add_actuator(
            name=joint.removesuffix("_joint"),     # e.g. "elbow"; same names the servos had
            target=joint,                          # which joint this motor drives
            trntype=mujoco.mjtTrn.mjTRN_JOINT,     # acts on a joint (not a tendon or site)
            ctrlrange=[-limit, limit],             # MuJoCo clips commands to the real limit
        )

    for key in spec.keys:
        key.ctrl = [0.0] * len(UR5E_JOINTS)


def load_ur5e_spec() -> mujoco.MjSpec:
    """The menagerie scene (arm + floor + lights) as an editable spec, with torque motors."""
    if not UR5E_SCENE_XML.exists():
        raise FileNotFoundError(f"{UR5E_SCENE_XML} is missing: run ./scripts/fetch_assets.sh")
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE_XML))
    use_torque_actuators(spec)
    return spec
