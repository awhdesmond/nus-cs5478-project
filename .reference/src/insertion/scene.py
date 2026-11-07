"""Build the MuJoCo model for a trial.

The scene is generated in code (via `mujoco.MjSpec`) rather than a static XML, because
clearance, socket pose and friction are swept by the evaluation harness.
"""

import math

import mujoco
import numpy as np

from insertion.config import UR5E_SCENE_XML, SceneConfig

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
PEG_RGBA = [0.8, 0.65, 0.2, 1.0]
SOCKET_RGBA = [0.25, 0.25, 0.3, 1.0]
CHAMFER_RGBA = [0.45, 0.35, 0.2, 1.0]
PEDESTAL_RGBA = [0.6, 0.6, 0.6, 1.0]


UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR5E_TORQUE_LIMITS = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)  # [Nm], menagerie forcerange


def load_stock_ur5e() -> mujoco.MjModel:
    """Unmodified menagerie UR5e + floor. Used by the smoke test."""
    return mujoco.MjModel.from_xml_path(str(UR5E_SCENE_XML))


def hole_width(cfg: SceneConfig) -> float:
    """Side length of the square hole: peg diameter plus total clearance."""
    return 2 * cfg.peg_radius + cfg.clearance


def contact_params(cfg: SceneConfig) -> dict:
    """Friction and solver params for any geom in the peg-socket contact pair."""
    return dict(
        friction=[cfg.friction, 0.005, 0.0001],
        solref=list(cfg.contact_solref),
        solimp=list(cfg.contact_solimp),
    )


def _check_chamfer(geom_cfg) -> None:
    c, t, d = geom_cfg.socket_chamfer, geom_cfg.socket_wall, geom_cfg.socket_depth
    if c < 0 or 2 * c > min(t, d):
        raise ValueError(f"chamfer {c} must satisfy 0 <= 2*chamfer <= min(wall {t}, depth {d})")


def add_square_hole_geoms(body: mujoco.MjsBody, h: float, geom_cfg, prefix: str) -> float:
    """Box geoms forming a chamfered square hole of half-width `h` in `body`'s frame.

    Frame: origin at the hole-bottom centre, +z up. The hole spans |x|, |y| < h and 0 < z < d.
    Mesh geoms are avoided because MuJoCo collides with their convex hull, which would fill the hole.

    Each wall is split so a 45° chamfer can be formed from boxes alone (cross-section of
    the +x wall shown; c = chamfer, t = wall thickness):

        z=d      ╱██████      upper box:  x ∈ [h+c, h+t],  z ∈ [d-c, d]
               ╱ ◇ █████      diamond:    box rotated 45° centred at (h+c, d-c); its
        z=d-c  ██◇███████                 top-left face is the chamfer surface
               ██████████     lower box:  x ∈ [h, h+t],    z ∈ [0, d-c]
        z=0    ██████████
              x=h

    Diamonds extend c past the hole edge at both ends, so the chamfer ring is continuous
    at the corners (no notches for the peg to catch on). Returns the outer half-width.
    `geom_cfg` provides socket_depth/wall/base/chamfer and the contact parameters.
    """
    _check_chamfer(geom_cfg)
    d, t, b, c = geom_cfg.socket_depth, geom_cfg.socket_wall, geom_cfg.socket_base, geom_cfg.socket_chamfer
    H = h + t

    def box(name, center, half, quat=(1, 0, 0, 0), rgba=SOCKET_RGBA):
        body.add_geom(
            name=f"{prefix}_{name}", type=BOX, pos=list(center), size=list(half),
            quat=list(quat), rgba=rgba, **contact_params(geom_cfg),
        )

    box("base", (0, 0, -b / 2), (H, H, b / 2))

    # Walls. ±x walls span the full outer length in y; ±y walls fit between them.
    lower_top = d - c
    for sign, axis in ((1, "px"), (-1, "nx")):
        box(f"wall_{axis}", (sign * (h + t / 2), 0, lower_top / 2), (t / 2, H, lower_top / 2))
    for sign, axis in ((1, "py"), (-1, "ny")):
        box(f"wall_{axis}", (0, sign * (h + t / 2), lower_top / 2), (h, t / 2, lower_top / 2))

    if c > 0:
        upper_half_t = (t - c) / 2
        for sign, axis in ((1, "px"), (-1, "nx")):
            box(f"upper_{axis}", (sign * (h + c + upper_half_t), 0, d - c / 2), (upper_half_t, H, c / 2))
        for sign, axis in ((1, "py"), (-1, "ny")):
            box(f"upper_{axis}", (0, sign * (h + c + upper_half_t), d - c / 2), (H, upper_half_t, c / 2))

        # Square cross-section of side c·√2 rotated 45°: vertices lie at ±c from its centre.
        s = c * math.sqrt(2) / 2
        q = math.sin(math.pi / 8)
        w = math.cos(math.pi / 8)
        along = h + c
        for sign, axis in ((1, "px"), (-1, "nx")):
            box(f"chamfer_{axis}", (sign * (h + c), 0, d - c), (s, along, s), (w, 0, q, 0), CHAMFER_RGBA)
        for sign, axis in ((1, "py"), (-1, "ny")):
            box(f"chamfer_{axis}", (0, sign * (h + c), d - c), (along, s, s), (w, q, 0, 0), CHAMFER_RGBA)
    return H


def add_ngon_hole_geoms(body: mujoco.MjsBody, apothem: float, sides: int, geom_cfg, prefix: str) -> float:
    """Box geoms forming a chamfered regular N-gon hole (a round hole from primitives).

    Each of the N faces is the square-hole wall section above, rotated about z to angle
    φ_i = 2πi/N, with its inner face at distance `apothem` from the axis. The hole is the
    intersection of those half-spaces; every box lies at radial distance ≥ apothem along its
    own face normal, so overlaps between neighbours only ever add wall material. Vertices sit
    at apothem/cos(π/N) (+0.11 mm at N = 16, apothem 5.5 mm). Returns the outer half-width
    of the square base plate.
    """
    _check_chamfer(geom_cfg)
    d, t, b, c = geom_cfg.socket_depth, geom_cfg.socket_wall, geom_cfg.socket_base, geom_cfg.socket_chamfer
    R = apothem
    tan_half = math.tan(math.pi / sides)
    H = (R + t) / math.cos(math.pi / sides)

    def quat_z(angle):
        return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

    def mul(a, bq):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = bq
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def box(name, radial, z, half, quat, rgba=SOCKET_RGBA):
        body.add_geom(
            name=f"{prefix}_{name}", type=BOX, pos=list(radial) + [z], size=list(half),
            quat=list(quat), rgba=rgba, **contact_params(geom_cfg),
        )

    body.add_geom(
        name=f"{prefix}_base", type=BOX, pos=[0, 0, -b / 2], size=[H, H, b / 2],
        rgba=SOCKET_RGBA, **contact_params(geom_cfg),
    )
    lower_top = d - c
    q45_about_y = (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0)
    s = c * math.sqrt(2) / 2
    for i in range(sides):
        phi = 2 * math.pi * i / sides
        er = (math.cos(phi), math.sin(phi))
        qz = quat_z(phi)
        r_wall = R + t / 2
        box(f"wall_{i}", (er[0] * r_wall, er[1] * r_wall), lower_top / 2,
            (t / 2, (R + t) * tan_half + 1e-4, lower_top / 2), qz)
        if c > 0:
            r_upper = R + c + (t - c) / 2
            box(f"upper_{i}", (er[0] * r_upper, er[1] * r_upper), d - c / 2,
                ((t - c) / 2, (R + t) * tan_half + 1e-4, c / 2), qz)
            r_diamond = R + c
            box(f"chamfer_{i}", (er[0] * r_diamond, er[1] * r_diamond), d - c,
                (s, (R + 2 * c) * tan_half + 1e-4, s), mul(qz, q45_about_y), CHAMFER_RGBA)
    return H


def add_socket(spec: mujoco.MjSpec, cfg: SceneConfig) -> mujoco.MjsBody:
    """Insertion-only scene: one square-hole socket (see `add_square_hole_geoms`) on a pedestal.

    Sites: `socket_bottom` (hole-bottom centre) and `socket_entrance` (hole-top centre)
    for success checks — the controller must not read these (true pose).
    """
    socket = spec.worldbody.add_body(name="socket", pos=list(cfg.socket_pos))
    H = add_square_hole_geoms(socket, hole_width(cfg) / 2, cfg, prefix="socket")
    socket.add_site(name="socket_bottom", pos=[0, 0, 0], size=[0.002, 0, 0])
    socket.add_site(name="socket_entrance", pos=[0, 0, cfg.socket_depth], size=[0.002, 0, 0])

    pedestal_height = cfg.socket_pos[2] - cfg.socket_base
    if pedestal_height > 0:
        x, y, _ = cfg.socket_pos
        spec.worldbody.add_geom(
            name="pedestal", type=BOX, pos=[x, y, pedestal_height / 2],
            size=[H, H, pedestal_height / 2], rgba=PEDESTAL_RGBA,
        )
    return socket


def arm_ctrlrange(model: mujoco.MjModel) -> np.ndarray:
    """Control range of the six arm torque motors (added first, so actuators 0-5)."""
    names = [model.actuator(i).name for i in range(len(UR5E_JOINTS))]
    assert names == [j.removesuffix("_joint") for j in UR5E_JOINTS], names
    return model.actuator_ctrlrange[: len(UR5E_JOINTS)]


def use_torque_actuators(spec: mujoco.MjSpec) -> None:
    """Replace the menagerie position servos with direct joint-torque motors.

    ctrl is then joint torque [Nm], so the controller owns gravity compensation and
    compliance. The `home` keyframe's ctrl (servo targets) is zeroed to match.
    """
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for joint, limit in zip(UR5E_JOINTS, UR5E_TORQUE_LIMITS):
        spec.add_actuator(
            name=joint.removesuffix("_joint"), target=joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-limit, limit],
        )
    for key in spec.keys:
        key.ctrl = [0.0] * len(UR5E_JOINTS)


def add_peg(spec: mujoco.MjSpec, cfg: SceneConfig) -> mujoco.MjsBody:
    """Round peg rigidly welded to the flange (no joint).

    Peg frame = flange tool frame at `attachment_site`: +z points out of the flange, so
    the peg extends along +z. Sites: `ft_site` at the flange (for wrist F/T sensors) and
    `peg_tip` at the free end (the end-effector point for motion and success checks).
    """
    flange = spec.site("attachment_site")
    peg = spec.body("wrist_3_link").add_body(name="peg", pos=list(flange.pos), quat=list(flange.quat))
    peg.add_geom(
        name="peg", type=CYLINDER, size=[cfg.peg_radius, cfg.peg_length / 2],
        pos=[0, 0, cfg.peg_length / 2], rgba=PEG_RGBA, **contact_params(cfg),
    )
    peg.add_site(name="ft_site", pos=[0, 0, 0], size=[0.002, 0, 0])
    peg.add_site(name="peg_tip", pos=[0, 0, cfg.peg_length], size=[0.002, 0, 0])
    return peg


def add_wrist_ft_sensor(spec: mujoco.MjSpec) -> None:
    """Force and torque sensors at `ft_site` (flange/peg interface), in the site frame.

    MuJoCo reports the wrench the parent (wrist) exerts on the child subtree (peg), so the
    reading includes peg weight and inertia — tare it before contact.
    """
    for name, sensor_type in (("ft_force", mujoco.mjtSensor.mjSENS_FORCE), ("ft_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        spec.add_sensor(name=name, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft_site")


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    """Scene spec before compilation, so tests and tools can add bodies to it."""
    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE_XML))
    spec.option.timestep = cfg.timestep
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    use_torque_actuators(spec)
    add_peg(spec, cfg)
    add_wrist_ft_sensor(spec)
    add_socket(spec, cfg)
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    """Week 1 scene: torque-controlled UR5e with welded peg, plus socket."""
    return build_spec(cfg).compile()
