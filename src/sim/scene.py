"""Build the MuJoCo model for a trial.

The scene is generated in code (MjSpec) rather than written as a static XML file, because
clearance, friction and the layout change from trial to trial.
"""

import math
import mujoco
import numpy as np

from sim.config import SceneConfig, PegPose, PEGS
from sim.ur5e import UR5E_JOINTS, load_ur5e_spec
from sim.holes import BOX, WALL_RGBA, add_ngon_hole, add_square_hole
from sim.utils import quat_mul, quat_z


TABLE_RGBA = [0.35, 0.37, 0.40, 1.0]
PEG_RGBA = [0.85, 0.65, 0.15, 1.0]  # both pegs the same amber: the camera must tell them apart by shape
BAND_RGBA = [0.15, 0.35, 0.90, 1.0]  # blue: marks the insertion end
BAND_PROUD = 0.0001  # [m] the band sits 0.1 mm outside the peg's surface so it's drawn on top of it
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER

GRIPPER_RGBA = [0.3, 0.3, 0.35, 1.0]
FINGER_RGBA = [0.55, 0.55, 0.6, 1.0]
FINGER_JOINTS = ("finger_left", "finger_right")
# Equality constraints are soft springs too. These make one effectively rigid: with MuJoCo's soft default, a
# 5 N push slid the gripper 4 mm along a welded peg in the first attempt.
RIGID = dict(solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])


HIDDEN_SITE_GROUP = 4  # sites are markers for code, not objects; group 4 keeps them out of camera images


def contact_params(cfg: SceneConfig) -> dict:
    """Friction and stiffness for every geom a peg can touch.

    MuJoCo mixes the two geoms' settings in a contact, so pegs, port blocks and table all get the same ones.
    friction = (sliding, torsional, rolling).
    """
    return dict(friction=[cfg.friction, 0.005, 0.0001], solref=list(cfg.contact_solref),
                solimp=list(cfg.contact_solimp))


def peg_half(cfg: SceneConfig, kind: str) -> float:
    """A peg's half-width: the round peg's radius, or half the square peg's width across its flats."""
    return cfg.round_radius if kind == "round" else cfg.square_half


def port_hole_half(cfg: SceneConfig, kind: str) -> float:
    """Half-width of a port's hole (for the round hole: the polygon's apothem, centre to a flat)."""
    return peg_half(cfg, kind) + cfg.clearance / 2  # clearance is total, so half of it on each side


def port_block_half(cfg: SceneConfig, kind: str) -> float:
    """Half-width of a port block's square base plate: wide enough to sit under all of its walls."""
    outer = port_hole_half(cfg, kind) + cfg.port_wall  # to the walls' outside (round: to a flat)
    if kind == "round":
        outer /= math.cos(math.pi / cfg.round_sides)  # out to the polygon's corners
    return outer


def port_top_height(cfg: SceneConfig) -> float:
    """Height of every port block's top surface (= the hole entrance) above the floor."""
    return cfg.table_height + cfg.port_base + cfg.port_depth


def add_table(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    # A fixed box sitting on the floor: a geom on the worldbody never moves.
    (tx, ty), (hx, hy) = cfg.table_center, cfg.table_half
    spec.worldbody.add_geom(name="table", type=BOX, pos=[tx, ty, cfg.table_height / 2],
                            size=[hx, hy, cfg.table_height / 2], rgba=TABLE_RGBA, **contact_params(cfg))


def add_port(spec: mujoco.MjSpec, cfg: SceneConfig, kind: str) -> None:
    """One port: a block on the table holding one chamfered hole (`kind` = "round" or "square").

    The body `<kind>_port` sits at the hole-bottom centre, turned by the port's yaw. Below it is a base
    plate (the hole's floor); around it, the hole's walls. A body with no joint is welded in place: it never
    moves, but each trial can put it somewhere new.
    """
    pose, b, d = cfg.port_poses[kind], cfg.port_base, cfg.port_depth
    contact = contact_params(cfg)
    port = spec.worldbody.add_body(name=f"{kind}_port", pos=[pose.x, pose.y, cfg.table_height + b],
                                   quat=quat_z(pose.yaw))
    plate = port_block_half(cfg, kind)
    port.add_geom(name=f"{kind}_port_base", type=BOX, pos=[0, 0, -b / 2], size=[plate, plate, b / 2],
                  rgba=WALL_RGBA, **contact)
    hole = port_hole_half(cfg, kind)
    if kind == "round":
        add_ngon_hole(port, hole, cfg.round_sides, cfg.port_wall, d, cfg.port_chamfer, contact, f"{kind}_port")
    else:
        add_square_hole(port, hole, cfg.port_wall, d, cfg.port_chamfer, contact, f"{kind}_port")
    # Sites are massless markers. These are ground truth for scoring; a controller must never read them.
    port.add_site(name=f"{kind}_port_bottom", pos=[0, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
    port.add_site(name=f"{kind}_port_entrance", pos=[0, 0, d], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)


def peg_body_pose(cfg: SceneConfig, kind: str, pose: PegPose) -> tuple[np.ndarray, np.ndarray]:
    """Where a peg's body (its centre) sits at rest on the table, and its orientation quaternion.

    The peg's own +z axis runs from its marked end (bottom when upright) to its top.
    """
    q_yaw = quat_z(pose.yaw)
    if not pose.lying:
        # Standing on the table: centre half a length up.
        return np.array([pose.x, pose.y, cfg.table_height + cfg.peg_length / 2]), np.array(q_yaw)
    # Lying: first tip it over by +90° about y (its +z now points along world +x, and the square peg's
    # local x face is underneath), then turn it by yaw. Its centre is half a width up.
    tip_over = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
    return np.array([pose.x, pose.y, cfg.table_height + peg_half(cfg, kind)]), np.array(quat_mul(q_yaw, tip_over))


def add_peg(spec: mujoco.MjSpec, cfg: SceneConfig, kind: str) -> None:
    """A peg: a free body (it falls, slides, can be grasped) with a blue band on its marked end.

    Body `<kind>_peg`; geoms `<kind>_peg` (the solid) and `<kind>_band` (visual only); sites `<kind>_tip`
    (marked end) and `<kind>_top`, used later for ground truth.
    """
    L, band, w = cfg.peg_length, cfg.tip_band_length, peg_half(cfg, kind)
    pos, quat = peg_body_pose(cfg, kind, cfg.peg_poses[kind])
    peg = spec.worldbody.add_body(name=f"{kind}_peg", pos=pos.tolist(), quat=quat.tolist())
    # A free joint gives the body all 6 degrees of freedom: 3 position + 3 rotation (7 numbers in qpos:
    # position, then a quaternion).
    peg.add_freejoint(name=f"{kind}_peg")
    contact = contact_params(cfg)
    if kind == "round":
        solid = dict(type=CYLINDER, size=[w, L / 2])  # cylinder size = (radius, half-length)
        band_shape = dict(type=CYLINDER, size=[w + BAND_PROUD, band / 2 + BAND_PROUD])
        # condim 6 adds torsional and rolling friction to this peg's contacts; rolling friction stops a lying
        # cylinder from rolling away.
        contact["friction"] = [cfg.friction, 0.005, cfg.round_rolling_friction]
        contact["condim"] = 6
    else:
        solid = dict(type=BOX, size=[w, w, L / 2])
        band_shape = dict(type=BOX, size=[w + BAND_PROUD, w + BAND_PROUD, band / 2 + BAND_PROUD])
    # Mass comes from volume × density (MuJoCo's default 1000 kg/m³): about 4 g per peg.
    peg.add_geom(name=f"{kind}_peg", rgba=PEG_RGBA, **solid, **contact)
    # The band: contype/conaffinity 0 = never collides; density 0 = no mass. Only the camera sees it.
    peg.add_geom(name=f"{kind}_band", pos=[0, 0, -L / 2 + band / 2], rgba=BAND_RGBA, contype=0, conaffinity=0,
                 density=0.0, **band_shape)
    peg.add_site(name=f"{kind}_tip", pos=[0, 0, -L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
    peg.add_site(name=f"{kind}_top", pos=[0, 0, L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)


def add_gripper(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    """A parallel-jaw gripper on the arm's flange: a palm and two fingers on slide joints.

    Body `gripper` (palm) with sites `ft_site` (where the wrist force/torque sensor measures) and `tcp` (tool
    centre point: midway between the finger pads). The fingers close along the gripper's local y axis. Each
    finger joint measures how far that finger has closed from fully open [m]; a joint equality makes the right
    finger mirror the left, and one force-limited position actuator `gripper` drives the left (ctrl = closing
    travel [m], 0 = open).
    """
    # The menagerie marks the flange (tool mounting face) with a site; the gripper body goes exactly there.
    flange = spec.site("attachment_site")
    gripper = spec.body("wrist_3_link").add_body(name="gripper", pos=list(flange.pos), quat=list(flange.quat))
    # +z points out of the flange, so the palm, fingers and tcp are all at positive z.
    palm_half = (0.02, cfg.finger_open_gap / 2 + cfg.finger_thickness + 0.004, cfg.palm_offset / 2)
    gripper.add_geom(name="palm", type=BOX, size=list(palm_half), pos=[0, 0, cfg.palm_offset / 2], rgba=GRIPPER_RGBA)
    gripper.add_site(name="ft_site", pos=[0, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)

    travel = cfg.finger_open_gap / 2  # each finger can close by half the gap, until they meet
    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = gripper.add_body(name=f"finger_{side}",
                                  pos=[0, sign * (cfg.finger_open_gap / 2 + cfg.finger_thickness / 2), cfg.palm_offset])
        # A slide joint moves along one axis. Pointing it inward (-sign·y) makes positive = closing for both.
        finger.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -sign, 0],
                         range=[0, travel], damping=2.0, armature=0.01)
        pads = contact_params(cfg)
        pads["friction"] = [cfg.finger_friction, 0.005, 0.0001]
        finger.add_geom(name=f"finger_{side}", type=BOX, pos=[0, 0, cfg.finger_length / 2],
                        size=[cfg.finger_width / 2, cfg.finger_thickness / 2, cfg.finger_length / 2],
                        rgba=FINGER_RGBA, mass=0.02, **pads)
    gripper.add_site(name="tcp", pos=[0, 0, cfg.palm_offset + cfg.finger_length - cfg.pad_depth],
                     size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)

    # right finger = left finger. data = polynomial coefficients (right = 0 + 1·left), padded to MuJoCo's 11.
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_right", name2="finger_left",
                      objtype=mujoco.mjtObj.mjOBJ_JOINT, data=[0, 1, 0, 0, 0] + [0] * 6, **RIGID)
    # A position actuator is a motor with a built-in spring: force = kp·(ctrl - q) - kv·q̇, clipped to
    # ±grip_force. Commanding full travel squeezes a peg with up to grip_force.
    spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[0, travel],
                      forcerange=[-cfg.grip_force, cfg.grip_force]).set_to_position(kp=2000.0, kv=40.0)


def add_ft_sensor(spec: mujoco.MjSpec) -> None:
    """A six-axis force/torque sensor at `ft_site`, between the flange and the gripper.

    MuJoCo reports the force and torque the wrist applies to everything beyond the site (gripper, plus any
    held peg), in the site's frame. So at rest it reads their weight; controllers tare that off (step 5).
    """
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="ft_site")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="ft_site")


def add_attach_weld(spec: mujoco.MjSpec, kind: str) -> None:
    """An inactive weld between the gripper and a peg, switched on at grasp time (Sim.attach).

    The proposal locks a peg to the gripper once both fingers touch it. A weld equality glues two bodies at a
    fixed relative pose; Sim.attach fills in that pose when it activates the weld.
    """
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=f"attach_{kind}", name1="gripper", name2=f"{kind}_peg",
                      objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                      # data = anchor (3), relative pose (3 position + 4 quaternion), torque scale (1). Set
                      # explicitly: an MjSpec equality starts with the joint equality's default numbers,
                      # which put the anchor 1 m off and flung the peg on attach in the first attempt.
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], **RIGID)


def rebuild_keyframes(spec: mujoco.MjSpec) -> None:
    """Re-create the `home` keyframe for the bigger model.

    A keyframe stores the whole qpos vector. The menagerie's `home` was saved for the 6 arm joints only, so
    once the pegs' free joints exist it no longer fits and compiling would fail. Rebuild it: arm at the
    menagerie's home angles, everything else at its default (qpos0: each peg where its body was placed).
    """
    arm_home = list(spec.key("home").qpos)[: len(UR5E_JOINTS)]  # read it before deleting
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()  # compile once to learn the full qpos layout
    qpos = model.qpos0.copy()
    for joint, angle in zip(UR5E_JOINTS, arm_home):
        qpos[model.joint(joint).qposadr[0]] = angle  # look joints up by name, not position
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=[0.0] * model.nu)


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    """The scene before compiling, so later parts (and tests) can keep adding to it."""
    spec = load_ur5e_spec()
    spec.option.timestep = cfg.timestep
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    add_table(spec, cfg)
    add_gripper(spec, cfg)
    add_ft_sensor(spec)
    for kind in PEGS:
        add_port(spec, cfg, kind)
        add_peg(spec, cfg, kind)
        add_attach_weld(spec, kind)
    rebuild_keyframes(spec)  # last: needs every joint in place
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    return build_spec(cfg).compile()
