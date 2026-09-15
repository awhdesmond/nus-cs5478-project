"""Two-connector assembly scene (weeks 6–7 extension).

UR5e (torque motors) + parallel-jaw gripper with a wrist F/T sensor, a table, a fixture with a
round (N-gon) and a square socket, a round and a square connector as free bodies, a fixed
overhead RGB-D camera, and one inactive weld per connector for attach-on-grasp.

Both connectors share one colour, so perception has to tell them apart by shape.
"""

import math

import mujoco
import numpy as np

from insertion.config import UR5E_SCENE_XML, AssemblyConfig, ConnectorPose
from insertion.scene import (
    BOX,
    CYLINDER,
    UR5E_JOINTS,
    add_ngon_hole_geoms,
    add_square_hole_geoms,
    add_wrist_ft_sensor,
    contact_params,
    use_torque_actuators,
)

CONNECTORS = ("round", "square")
CONNECTOR_RGBA = [0.85, 0.65, 0.15, 1.0]
GRIPPER_RGBA = [0.3, 0.3, 0.35, 1.0]
FINGER_RGBA = [0.55, 0.55, 0.6, 1.0]
TABLE_RGBA = [0.35, 0.37, 0.40, 1.0]
HIDDEN_SITE_GROUP = 4  # sites stay out of camera images (the default renderer shows groups 0–2)
FINGER_JOINTS = ("finger_left", "finger_right")


def socket_apothem(cfg: AssemblyConfig, kind: str) -> float:
    """Half-width (square) or apothem (round N-gon) of a socket hole."""
    if kind == "round":
        return cfg.round_radius + cfg.clearance / 2
    return cfg.square_half + cfg.clearance / 2


def connector_half_width(cfg: AssemblyConfig, kind: str) -> float:
    return cfg.round_radius if kind == "round" else cfg.square_half


def add_gripper(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    """Parallel-jaw gripper at the flange (+z out of the flange, fingers close along local y).

    Each finger is a slide joint measured as closing travel from fully open; the right finger is
    coupled to the left by a joint equality, and one force-limited position actuator drives the
    left finger (ctrl = closing travel [m]). Sites: `ft_site` at the gripper base (wrist F/T sensor)
    and `tcp` midway between the finger pads.
    """
    flange = spec.site("attachment_site")
    gripper = spec.body("wrist_3_link").add_body(name="gripper", pos=list(flange.pos), quat=list(flange.quat))
    palm_half = (0.02, cfg.finger_open_gap / 2 + cfg.finger_thickness + 0.004, cfg.palm_offset / 2)
    gripper.add_geom(name="palm", type=BOX, size=list(palm_half), pos=[0, 0, cfg.palm_offset / 2], rgba=GRIPPER_RGBA)
    gripper.add_site(name="ft_site", pos=[0, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)

    travel = cfg.finger_open_gap / 2
    for side, sign in (("left", 1.0), ("right", -1.0)):
        body = gripper.add_body(
            name=f"finger_{side}",
            pos=[0, sign * (cfg.finger_open_gap / 2 + cfg.finger_thickness / 2), cfg.palm_offset],
        )
        body.add_joint(
            name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -sign, 0],
            range=[0, travel], damping=2.0, armature=0.01,
        )
        params = contact_params(cfg)
        params["friction"] = [cfg.finger_friction, 0.005, 0.0001]
        body.add_geom(
            name=f"finger_{side}", type=BOX, pos=[0, 0, cfg.finger_length / 2],
            size=[cfg.finger_width / 2, cfg.finger_thickness / 2, cfg.finger_length / 2],
            rgba=FINGER_RGBA, mass=0.02, **params,
        )
    pad_depth = 0.008  # tcp (grasp contact centre) this far above the fingertips
    gripper.add_site(name="tcp", pos=[0, 0, cfg.palm_offset + cfg.finger_length - pad_depth], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)

    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_right", name2="finger_left",
        objtype=mujoco.mjtObj.mjOBJ_JOINT, data=[0, 1, 0, 0, 0] + [0] * 6,
    )
    spec.add_actuator(
        name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT,
        ctrlrange=[0, travel], forcerange=[-cfg.grip_force, cfg.grip_force],
    ).set_to_position(kp=2000.0, kv=40.0)


def add_table_and_fixture(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    tx, ty = cfg.table_center
    hx, hy = cfg.table_half
    spec.worldbody.add_geom(
        name="table", type=BOX, pos=[tx, ty, cfg.table_height / 2],
        size=[hx, hy, cfg.table_height / 2], rgba=TABLE_RGBA, **contact_params(cfg),
    )
    hole_bottom_z = cfg.table_height + cfg.socket_base
    for kind, xy in (("round", cfg.round_socket_xy), ("square", cfg.square_socket_xy)):
        body = spec.worldbody.add_body(name=f"{kind}_socket", pos=[xy[0], xy[1], hole_bottom_z])
        if kind == "round":
            add_ngon_hole_geoms(body, socket_apothem(cfg, kind), cfg.round_sides, cfg, prefix=f"{kind}_socket")
        else:
            add_square_hole_geoms(body, socket_apothem(cfg, kind), cfg, prefix=f"{kind}_socket")
        body.add_site(name=f"{kind}_socket_bottom", pos=[0, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        body.add_site(name=f"{kind}_socket_entrance", pos=[0, 0, cfg.socket_depth], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)


def add_connectors(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    L = cfg.connector_length
    for kind, pose in (("round", cfg.round_pose), ("square", cfg.square_pose)):
        body = spec.worldbody.add_body(
            name=f"{kind}_connector",
            pos=[pose.x, pose.y, cfg.table_height + L / 2],
            quat=[math.cos(pose.yaw / 2), 0, 0, math.sin(pose.yaw / 2)],
        )
        body.add_freejoint(name=f"{kind}_connector")
        if kind == "round":
            geom = dict(type=CYLINDER, size=[cfg.round_radius, L / 2])
        else:
            geom = dict(type=BOX, size=[cfg.square_half, cfg.square_half, L / 2])
        body.add_geom(name=f"{kind}_connector", rgba=CONNECTOR_RGBA, **geom, **contact_params(cfg))
        body.add_site(name=f"{kind}_tip", pos=[0, 0, -L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        body.add_site(name=f"{kind}_top", pos=[0, 0, L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        # Attach-on-grasp weld; relative pose is set at activation (see AssemblySim.attach).
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD, name=f"attach_{kind}", name1="gripper", name2=f"{kind}_connector",
            objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
            # anchor(3) relpose(7) torquescale(1). Set explicitly: the spec's default data is the joint
            # equality's polycoef [0, 1, 0, ...], which would put the weld anchor 1 m off.
            data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        )


def add_camera(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    """Overhead camera: MuJoCo cameras look along their −z, so identity orientation looks straight down."""
    spec.worldbody.add_camera(name="overhead", pos=list(cfg.camera_pos), quat=[1, 0, 0, 0], fovy=cfg.camera_fovy)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, cfg.image_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, cfg.image_height)


def build_assembly_spec(cfg: AssemblyConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE_XML))
    spec.option.timestep = cfg.timestep
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    use_torque_actuators(spec)
    add_gripper(spec, cfg)
    add_wrist_ft_sensor(spec)
    add_table_and_fixture(spec, cfg)
    add_connectors(spec, cfg)
    add_camera(spec, cfg)
    _set_home_keyframe(spec)
    return spec


# Arm parked behind the base (home mirrored about the base, shoulder pan +π/2): nothing above the
# workspace, so the overhead camera sees the table, fixture and connectors unobstructed.
CAMERA_CLEAR_ARM = (1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)


def _set_home_keyframe(spec: mujoco.MjSpec) -> None:
    """Rebuild keyframes for the enlarged qpos (gripper open, connectors at their initial poses).

    `home`: menagerie arm home, above the fixture. `camera_clear`: arm parked out of the camera's view.
    """
    arm_home = list(spec.key("home").qpos)
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    for name, arm in (("home", arm_home), ("camera_clear", CAMERA_CLEAR_ARM)):
        qpos = model.qpos0.copy()
        for joint, value in zip(UR5E_JOINTS, arm):
            qpos[model.joint(joint).qposadr[0]] = value
        spec.add_key(name=name, qpos=qpos.tolist(), ctrl=[0.0] * model.nu)


def build_assembly_model(cfg: AssemblyConfig) -> mujoco.MjModel:
    return build_assembly_spec(cfg).compile()


def sample_layout(cfg: AssemblyConfig, seed: int, max_tries: int = 1000) -> AssemblyConfig:
    """Random non-overlapping connector poses in the pick area; which connector goes where is random too."""
    from dataclasses import replace

    rng = np.random.default_rng([seed, 3])
    cx, cy = cfg.pick_area_center
    hx, hy = cfg.pick_area_half
    for _ in range(max_tries):
        a = rng.uniform([cx - hx, cy - hy], [cx + hx, cy + hy])
        b = rng.uniform([cx - hx, cy - hy], [cx + hx, cy + hy])
        if np.linalg.norm(a - b) >= cfg.min_connector_spacing:
            yaws = rng.uniform(0, np.pi / 2, size=2)
            return replace(cfg, round_pose=ConnectorPose(*a, yaws[0]), square_pose=ConnectorPose(*b, yaws[1]))
    raise RuntimeError(f"no layout with spacing {cfg.min_connector_spacing} m in the pick area")
