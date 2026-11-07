"""Build the MuJoCo model for a trial.

The scene is generated in code (MjSpec) rather than written as a static XML file, because
clearance, friction and the layout change from trial to trial.
"""

import math
import mujoco

from sim.config import SceneConfig
from sim.ur5e import load_ur5e_spec
from sim.holes import BOX, WALL_RGBA, add_ngon_hole, add_square_hole
from sim.utils import quat_z


TABLE_RGBA = [0.35, 0.37, 0.40, 1.0]
HIDDEN_SITE_GROUP = 4  # sites are markers for code, not objects; group 4 keeps them out of camera images


def contact_params(cfg: SceneConfig) -> dict:
    """Friction and stiffness for every geom a peg can touch.

    MuJoCo mixes the two geoms' settings in a contact, so pegs, port blocks and table all get the same ones.
    friction = (sliding, torsional, rolling).
    """
    return dict(friction=[cfg.friction, 0.005, 0.0001], solref=list(cfg.contact_solref),
                solimp=list(cfg.contact_solimp))


def port_hole_half(cfg: SceneConfig, kind: str) -> float:
    """Half-width of a port's hole (for the round hole: the polygon's apothem, centre to a flat)."""
    peg_half = cfg.round_radius if kind == "round" else cfg.square_half
    return peg_half + cfg.clearance / 2  # clearance is total, so half of it on each side


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


def build_spec(cfg: SceneConfig) -> mujoco.MjSpec:
    """The scene before compiling, so later parts (and tests) can keep adding to it."""
    spec = load_ur5e_spec()
    spec.option.timestep = cfg.timestep
    if cfg.elliptic_cone:
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    add_table(spec, cfg)
    for kind in cfg.port_poses:
        add_port(spec, cfg, kind)
    return spec


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    return build_spec(cfg).compile()
