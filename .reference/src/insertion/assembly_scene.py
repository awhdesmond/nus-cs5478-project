"""Two-connector assembly scene (weeks 6-7 extension).

UR5e (torque motors) + parallel-jaw gripper with a wrist F/T sensor, a table, a fixture with a
round (N-gon) and a square socket (each its own block, at a sampled position and yaw), a round and a
square connector as free bodies, a fixed overhead RGB-D camera, and one inactive weld per connector for
attach-on-grasp.

Both connectors share one colour, so perception has to tell them apart by shape. Each has a blue band on its
insertion end (visual only: no contacts or mass), so a lying connector's insertion direction is perceivable.
"""

import math

import mujoco
import numpy as np

from insertion.config import UR5E_SCENE_XML, AssemblyConfig, ConnectorPose
from insertion.control.ik import JacobianIK
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
TIP_BAND_RGBA = [0.15, 0.35, 0.90, 1.0]  # marks the insertion end
TIP_BAND_PROUD = 0.0001  # [m] the visual band sits this far outside the connector surface (renders over it)
GRIPPER_RGBA = [0.3, 0.3, 0.35, 1.0]
FINGER_RGBA = [0.55, 0.55, 0.6, 1.0]
TABLE_RGBA = [0.35, 0.37, 0.40, 1.0]
HIDDEN_SITE_GROUP = 4  # sites stay out of camera images (the default renderer shows groups 0-2)
PLANNING_GEOM_GROUP = 5  # planning-only obstacle geoms: not rendered, no contacts
N_PLANNING_OBSTACLES = 6
PAD_DEPTH = 0.008  # [m] tcp (grasp contact centre) this far above the fingertips
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
    pad_depth = PAD_DEPTH
    gripper.add_site(name="tcp", pos=[0, 0, cfg.palm_offset + cfg.finger_length - pad_depth], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
    # Estimated tip of the held connector. Controllers move it (model.site_pos/quat) to their in-hand estimate,
    # so IK, impedance control and tip observations all act on the believed tip. Sites don't affect physics.
    # Compiled with a non-identity orientation (an upright connector held by a pointing-down gripper): a site
    # compiled coincident with its body gets site_sameframe set, and MuJoCo then ignores later quat changes.
    gripper.add_site(
        name="tool_tip", pos=[0, 0, cfg.palm_offset + cfg.finger_length - pad_depth + cfg.connector_length - cfg.grasp_below_top],
        quat=[0, 1, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP,
    )

    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_right", name2="finger_left",
        objtype=mujoco.mjtObj.mjOBJ_JOINT, data=[0, 1, 0, 0, 0] + [0] * 6,
        # Stiff so the fingers stay symmetric under grip force (the soft default let them differ by ~2 mm).
        solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
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
    for kind, pose in (("round", cfg.round_socket_pose), ("square", cfg.square_socket_pose)):
        body = spec.worldbody.add_body(
            name=f"{kind}_socket", pos=[pose.x, pose.y, hole_bottom_z],
            quat=[math.cos(pose.yaw / 2), 0, 0, math.sin(pose.yaw / 2)],
        )
        if kind == "round":
            add_ngon_hole_geoms(body, socket_apothem(cfg, kind), cfg.round_sides, cfg, prefix=f"{kind}_socket")
        else:
            add_square_hole_geoms(body, socket_apothem(cfg, kind), cfg, prefix=f"{kind}_socket")
        if cfg.socket_debris is not None and cfg.socket_debris[0] == kind:
            height = cfg.socket_debris[1]
            half = 0.6 * socket_apothem(cfg, kind)  # well inside the hole: blocks depth, not the walls
            body.add_geom(name=f"{kind}_socket_debris", type=BOX, size=[half, half, height / 2], pos=[0, 0, height / 2],
                          rgba=[0.3, 0.25, 0.2, 1], **contact_params(cfg))
        body.add_site(name=f"{kind}_socket_bottom", pos=[0, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        body.add_site(name=f"{kind}_socket_entrance", pos=[0, 0, cfg.socket_depth], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)


def connector_body_pose(cfg: AssemblyConfig, kind: str, pose: ConnectorPose) -> tuple[np.ndarray, np.ndarray]:
    """Resting body position and quaternion for a connector pose (upright on its marked end, or lying).

    The body's +z runs from the marked insertion end (`tip` site) to the top. Lying: +z points along
    (cos yaw, sin yaw), so the marked end is at centre - (cos yaw, sin yaw)·L/2."""
    q_yaw = np.array([math.cos(pose.yaw / 2), 0, 0, math.sin(pose.yaw / 2)])
    if not pose.lying:
        return np.array([pose.x, pose.y, cfg.table_height + cfg.connector_length / 2]), q_yaw
    # Rotate +90° about y (connector +z → world +x; a square's local x face → down), then yaw.
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, q_yaw, np.array([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]))
    return np.array([pose.x, pose.y, cfg.table_height + connector_half_width(cfg, kind)]), quat


def connector_axis_segment(cfg: AssemblyConfig, pose: ConnectorPose) -> tuple[np.ndarray, np.ndarray]:
    """The connector's axis projected on the table (a point if upright): two xy end points [m]."""
    c = np.array([pose.x, pose.y])
    if not pose.lying:
        return c, c
    half = cfg.connector_length / 2 * np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
    return c - half, c + half


def add_connectors(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    L = cfg.connector_length
    for kind, pose in (("round", cfg.round_pose), ("square", cfg.square_pose)):
        pos, quat = connector_body_pose(cfg, kind, pose)
        body = spec.worldbody.add_body(name=f"{kind}_connector", pos=pos.tolist(), quat=quat.tolist())
        body.add_freejoint(name=f"{kind}_connector")
        params = contact_params(cfg)
        if kind == "round":
            geom = dict(type=CYLINDER, size=[cfg.round_radius, L / 2])
            params["friction"] = [cfg.friction, 0.005, cfg.round_rolling_friction]
            params["condim"] = 6
        else:
            geom = dict(type=BOX, size=[cfg.square_half, cfg.square_half, L / 2])
        body.add_geom(name=f"{kind}_connector", rgba=CONNECTOR_RGBA, **geom, **params)
        band = cfg.tip_band_length
        if kind == "round":
            band_geom = dict(type=CYLINDER, size=[cfg.round_radius + TIP_BAND_PROUD, band / 2 + TIP_BAND_PROUD])
        else:
            band_geom = dict(type=BOX, size=[cfg.square_half + TIP_BAND_PROUD] * 2 + [band / 2 + TIP_BAND_PROUD])
        body.add_geom(name=f"{kind}_tip_band", pos=[0, 0, -L / 2 + band / 2], rgba=TIP_BAND_RGBA, contype=0,
                      conaffinity=0, density=0.0, **band_geom)
        body.add_site(name=f"{kind}_tip", pos=[0, 0, -L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        body.add_site(name=f"{kind}_top", pos=[0, 0, L / 2], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
        # Attach-on-grasp weld; relative pose is set at activation (see AssemblySim.attach).
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD, name=f"attach_{kind}", name1="gripper", name2=f"{kind}_connector",
            objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
            # anchor(3) relpose(7) torquescale(1). Set explicitly: the spec's default data is the joint
            # equality's polycoef [0, 1, 0, ...], which would put the weld anchor 1 m off.
            data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            # Stiff: MuJoCo's default soft equality let a 5 N insertion push slide the gripper 4 mm down the
            # connector (effectively in-hand slip), and the wrist F/T then saw only ~0.8 N of the push.
            solref=[0.004, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
        )


CABLE_RGBA = "0.08 0.08 0.1 1"


def _add_cable_plug(parent: mujoco.MjsBody, name: str, pos, quat, cfg: AssemblyConfig) -> mujoco.MjsBody:
    """Strain-relief boot + square plug with a blue marked tip, extending along the new body's +x from `pos`."""
    boot, plug, band = cfg.cable_boot_length, cfg.cable_plug_length, cfg.tip_band_length
    half = cfg.square_half
    body = parent.add_body(name=name, pos=list(pos), quat=list(quat))
    along_x = [0.7071068, 0, 0.7071068, 0]  # a cylinder's own z → the body's x
    body.add_geom(name=f"{name}_boot", type=CYLINDER, size=[cfg.cable_radius * 1.6, boot / 2], pos=[boot / 2, 0, 0],
                  quat=along_x, rgba=[0.08, 0.08, 0.1, 1], density=1200.0, **contact_params(cfg))
    body.add_geom(name=name, type=BOX, size=[plug / 2, half, half], pos=[boot + plug / 2, 0, 0],
                  rgba=CONNECTOR_RGBA, density=1200.0, **contact_params(cfg))
    body.add_geom(name=f"{name}_tip_band", type=BOX, size=[band / 2 + TIP_BAND_PROUD, half + TIP_BAND_PROUD, half + TIP_BAND_PROUD],
                  pos=[boot + plug - band / 2, 0, 0], rgba=TIP_BAND_RGBA, contype=0, conaffinity=0, density=0.0)
    body.add_site(name=f"{name}_tip", pos=[boot + plug, 0, 0], size=[0.002, 0, 0], group=HIDDEN_SITE_GROUP)
    return body


def add_cable(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    """Dual-ended deformable cable: MuJoCo's `cable` composite (a chain of capsules with ball joints) with the
    elasticity plugin supplying bending and twisting stiffness, and a square plug on each end — the square
    connector's cross-section, colour and blue marked tip, behind a strain-relief boot — so either end looks like it
    belongs in the square socket.

    End A (`cable_plug_a`) is docked horizontally in a port block at the table's left edge; the composite's first
    segment is fixed there (`initial="none"`), so the port holds the cable. The rest points off the table and drapes
    over the edge, with end B (`cable_plug_b`) hanging free.

    Composites only exist in MJCF, so the cable is built in its own spec from XML and attached with a prefix.
    Bodies: `cable_clamp` (port) → `cable_plug_a`, and `cable_B_first`, `cable_B_1`, … `cable_B_last` → `cable_plug_b`.
    """
    n = cfg.cable_segments + 1
    xml = f"""
    <mujoco>
      <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
      <worldbody>
        <body name="clamp">
          <composite prefix="" type="cable" curve="s" count="{n} 1 1" size="{cfg.cable_length}" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <config key="twist" value="{cfg.cable_twist_stiffness}"/>
              <config key="bend" value="{cfg.cable_bend_stiffness}"/>
              <config key="vmax" value="0"/>
            </plugin>
            <joint kind="main" damping="0.002" armature="1e-5"/>
            <geom type="capsule" size="{cfg.cable_radius}" rgba="{CABLE_RGBA}" density="1500" friction="{cfg.friction} 0.005 0.0001"/>
          </composite>
        </body>
      </worldbody>
    </mujoco>"""
    cable = mujoco.MjSpec.from_string(xml)
    x, y = cfg.cable_anchor
    axis_height = cfg.square_half + 0.0005  # plug A rests just above the table
    # The composite runs along the clamp's +x; turn it 180° about z so the cable points off the table's left edge (-x).
    frame = spec.worldbody.add_frame(pos=[x, y, cfg.table_height + axis_height], quat=[0, 0, 0, 1])
    frame.attach_body(cable.body("clamp"), "cable_", "")

    # End A: plug pointing back onto the table (clamp -x), inserted socket-depth deep into a port block.
    clamp = spec.body("cable_clamp")
    _add_cable_plug(clamp, "cable_plug_a", [0, 0, 0], [0, 0, 0, 1], cfg)
    exposed = cfg.cable_boot_length + cfg.cable_plug_length - cfg.socket_depth  # boot + plug length left outside the port
    depth = cfg.socket_depth + 0.002
    port_half = cfg.square_half + 0.005
    port_top = axis_height + port_half
    clamp.add_geom(name="cable_port", type=BOX, size=[depth / 2, port_half, port_top / 2],
                   pos=[-(exposed + depth / 2), 0, port_top / 2 - axis_height],
                   rgba=[0.25, 0.25, 0.3, 1], contype=0, conaffinity=0, density=0.0)

    # End B: each cable segment's capsule runs along its body's +x from the origin, so the free end is one segment
    # length along the last body's x.
    _add_cable_plug(spec.body("cable_B_last"), "cable_plug_b", [cfg.cable_length / cfg.cable_segments, 0, 0], [1, 0, 0, 0], cfg)


def add_camera(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    """Overhead camera: MuJoCo cameras look along their -z, so identity orientation looks straight down."""
    spec.worldbody.add_camera(name="overhead", pos=list(cfg.camera_pos), quat=[1, 0, 0, 0], fovy=cfg.camera_fovy)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, cfg.image_width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, cfg.image_height)


def add_planning_obstacles(spec: mujoco.MjSpec) -> None:
    """Mocap boxes the controller places at *perceived* obstacle poses for its own clearance checks.

    They never collide (contype/conaffinity 0) and aren't rendered, so they don't affect physics or images.
    Geometry distances to them come from `mujoco.mj_geomDistance` on a separate MjData (see assembly_planning).
    """
    for i in range(N_PLANNING_OBSTACLES):
        body = spec.worldbody.add_body(name=f"plan_obstacle_{i}", mocap=True, pos=[0, 0, -1])
        body.add_geom(name=f"plan_obstacle_{i}", type=BOX, size=[0.01, 0.01, 0.01], contype=0, conaffinity=0,
                      group=PLANNING_GEOM_GROUP, rgba=[1, 0, 0, 0.3], mass=0.0)


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
    add_planning_obstacles(spec)
    if cfg.cable:
        add_cable(spec, cfg)
    _set_home_keyframe(spec, cfg)
    return spec


def ready_arm(model: mujoco.MjModel, cfg: AssemblyConfig, q_home) -> np.ndarray:
    """Arm joints for the `ready` pose: tcp at cfg.ready_tcp pointing down, same elbow-up branch as home."""
    ik = JacobianIK(model, "tcp")
    q, ok = ik.solve(np.asarray(q_home, dtype=float), np.asarray(cfg.ready_tcp), np.diag([1.0, -1.0, -1.0]), iters=300)
    if not ok:
        raise RuntimeError(f"IK failed for the ready pose {cfg.ready_tcp}")
    return q


def _set_home_keyframe(spec: mujoco.MjSpec, cfg: AssemblyConfig) -> None:
    """Rebuild keyframes for the enlarged qpos (gripper open, connectors at their initial poses).

    `home`: menagerie arm home, above the fixture. `ready`: tool down above the table's near edge, outside the
    overhead camera's view; runs start and take every image here.
    """
    arm_home = list(spec.key("home").qpos)[: len(UR5E_JOINTS)]
    for key in list(spec.keys):
        spec.delete(key)
    model = spec.compile()
    for name, arm in (("home", arm_home), ("ready", ready_arm(model, cfg, arm_home))):
        qpos = model.qpos0.copy()
        for joint, value in zip(UR5E_JOINTS, arm):
            qpos[model.joint(joint).qposadr[0]] = value
        spec.add_key(name=name, qpos=qpos.tolist(), ctrl=[0.0] * model.nu)


def build_assembly_model(cfg: AssemblyConfig) -> mujoco.MjModel:
    return build_assembly_spec(cfg).compile()


def segment_distance(a0, a1, b0, b1) -> float:
    """Minimum distance between 2D segments a0-a1 and b0-b1 (points allowed)."""

    def point_to_segment(p, s0, s1):
        d = s1 - s0
        t = 0.0 if not d.any() else float(np.clip(np.dot(p - s0, d) / np.dot(d, d), 0.0, 1.0))
        return float(np.linalg.norm(p - (s0 + t * d)))

    def crosses():
        def orient(p, q, r):
            return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return orient(a0, a1, b0) * orient(a0, a1, b1) < 0 and orient(b0, b1, a0) * orient(b0, b1, a1) < 0

    if crosses():
        return 0.0
    return min(point_to_segment(a0, b0, b1), point_to_segment(a1, b0, b1), point_to_segment(b0, a0, a1), point_to_segment(b1, a0, a1))


def _sample_pair(rng, center, half, spacing, max_tries, what, lying=(False, False), cfg=None) -> tuple[ConnectorPose, ConnectorPose]:
    """Two poses in the area, axes (or centres) ≥ `spacing` apart; lying connectors must lie wholly in the area."""
    cx, cy = center
    hx, hy = half
    lo, hi = np.array([cx - hx, cy - hy]), np.array([cx + hx, cy + hy])
    for _ in range(max_tries):
        a = rng.uniform(lo, hi)
        b = rng.uniform(lo, hi)
        # Lying: a full turn, so the marked end points anywhere. (All upright: the original draw, keeping old layouts.)
        yaws = rng.uniform(0, np.pi / 2, size=2) if not any(lying) else rng.uniform(0, 2 * np.pi, size=2)
        poses = ConnectorPose(*a, yaws[0], lying[0]), ConnectorPose(*b, yaws[1], lying[1])
        if not any(lying):
            if np.linalg.norm(a - b) >= spacing:
                return poses
            continue
        segments = [connector_axis_segment(cfg, pose) for pose in poses]
        inside = all(np.all(end >= lo) and np.all(end <= hi) for seg in segments for end in seg)
        if inside and segment_distance(*segments[0], *segments[1]) >= spacing:
            return poses
    raise RuntimeError(f"no {what} layout with spacing {spacing} m")


ORIENTATIONS = ("upright", "lying", "mixed")


def sample_layout(cfg: AssemblyConfig, seed: int, sockets: bool = True, orientation: str = "upright", max_tries: int = 1000) -> AssemblyConfig:
    """Random non-overlapping poses: connectors in the pick area and (if `sockets`) sockets in the socket area.

    Positions and yaws are independent per object, so which one sits where is random too. `orientation`: all
    connectors upright, all lying, or "mixed" (each independently upright or lying, 50/50). Sockets and the mixed
    orientation draw use their own random streams, so a seed's upright connector layout is the same as before
    either option existed.
    """
    from dataclasses import replace

    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}")
    lying = {"upright": (False, False), "lying": (True, True)}.get(orientation)
    if lying is None:
        lying = tuple(bool(v) for v in np.random.default_rng([seed, 7]).random(2) < 0.5)
    rng = np.random.default_rng([seed, 3])
    round_pose, square_pose = _sample_pair(rng, cfg.pick_area_center, cfg.pick_area_half, cfg.min_connector_spacing, max_tries, "connector", lying, cfg)
    cfg = replace(cfg, round_pose=round_pose, square_pose=square_pose)
    if sockets:
        rng = np.random.default_rng([seed, 6])
        round_socket, square_socket = _sample_pair(rng, cfg.socket_area_center, cfg.socket_area_half, cfg.min_socket_spacing, max_tries, "socket")
        cfg = replace(cfg, round_socket_pose=round_socket, square_socket_pose=square_socket)
    return cfg
