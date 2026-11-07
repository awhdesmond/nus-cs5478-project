"""Pick a peg from the table and carry it to hover over its port, using only perception and the arm's own
sensing.

Sequence (proposal §4.3): above the peg → straight down to the grasp → close → check the fingers stopped
at the peg's width → lift → estimate where the peg sits in the hand → over to the port → straight down until
the peg's tip hovers just above the entrance, turned to match the port.
"""

import math
from dataclasses import dataclass

import numpy as np

from motion.kinematics import tool_down
from perception.estimate import SceneEstimate
from motion.arm import Arm
from grasp.plan import Grasp, GraspConfig, plan_grasp


@dataclass(frozen=True)
class PickConfig:
    peg_length: float = 0.050  # [m] nominal, from the part's drawing
    above: float = 0.05  # [m] start the straight descent this far above the grasp
    lift: float = 0.10  # [m] how high to lift the peg before moving over
    hover: float = 0.010  # [m] carried tip ends this far above the port's entrance
    width_tolerance: float = 0.0015  # [m] closed gap may differ this much from the expected peg width


@dataclass
class InHand:
    """Where the held peg's insertion end (tip) is, in the tcp's own frame, estimated at the grasp.

    Along the closing axis (tcp y) the mirrored fingers centre the peg, so that offset is 0. Along the pads
    (tcp x) and in height, it's what perception said, relative to where the tool actually was. The tip is then
    one peg length below the top face: along the tcp's +z, which points down.
    """
    kind: str
    tip: np.ndarray  # [m] (3,) in the tcp frame


@dataclass
class PickResult:
    ok: bool
    stage: str  # how far it got: "grasp", "lift", "carry" ("carry" and ok = hovering over the port)
    grasp: Grasp | None = None
    in_hand: InHand | None = None
    gap: float = math.nan  # [m] finger gap after closing


def estimate_in_hand(arm: Arm, peg_top_world: np.ndarray, kind: str, cfg: PickConfig) -> InHand:
    """The peg's tip in the tcp frame, from the perceived top-face centre and the tool's pose at the grasp."""
    p, R = arm.tcp()
    top = R.T @ (np.asarray(peg_top_world) - p)  # perceived top centre, in the tcp frame
    top[1] = 0.0  # the fingers centred it along the closing axis
    return InHand(kind, top + [0.0, 0.0, cfg.peg_length])


def pick(arm: Arm, scene: SceneEstimate, kind: str, cfg: PickConfig = PickConfig(),
         grasp_cfg: GraspConfig = GraspConfig()) -> PickResult:
    """Grasp the `kind` peg as perceived in `scene` and lift it clear of the table."""
    peg = scene.pegs[kind]
    others = [p.outline for k, p in scene.pegs.items() if k != kind]
    _, R_now = arm.tcp()
    grasp = plan_grasp(peg, others, reference_yaw=math.atan2(R_now[1, 0], R_now[0, 0]), cfg=grasp_cfg)
    R = tool_down(grasp.yaw)

    arm.grip(False)
    arm.move_to(grasp.tcp + [0, 0, cfg.above], R)
    arm.move_line(grasp.tcp, R)
    gap = arm.grip(True)
    if abs(gap - grasp.width) > cfg.width_tolerance:
        return PickResult(False, "grasp", grasp, gap=gap)  # closed on nothing, or on something else
    in_hand = estimate_in_hand(arm, peg.top, kind, cfg)
    arm.move_line(grasp.tcp + [0, 0, cfg.lift], R)
    return PickResult(True, "lift", grasp, in_hand, gap)


def port_yaw_for(in_hand: InHand, scene: SceneEstimate, current_yaw: float) -> float:
    """The tcp yaw to insert with. Square: the held square's faces are parallel to the pads (the squeeze
    turned it so), so match the tcp to the port's yaw, picking the multiple of 90° nearest the current yaw.
    Round: any yaw fits, so don't turn."""
    if in_hand.kind == "round":
        return current_yaw
    port = scene.ports["square"].yaw
    return port + (math.pi / 2) * round((current_yaw - port) / (math.pi / 2))


def tcp_for_tip(in_hand: InHand, tip_world, yaw: float) -> np.ndarray:
    """Where the tcp must be for the held peg's tip to be at tip_world, with the tool down at `yaw`."""
    return np.asarray(tip_world) - tool_down(yaw) @ in_hand.tip


def carry_to_port(arm: Arm, scene: SceneEstimate, picked: PickResult, cfg: PickConfig = PickConfig()) -> PickResult:
    """From the lift, carry the held peg until its tip hovers `cfg.hover` above its port's entrance."""
    in_hand = picked.in_hand
    entrance = scene.ports[in_hand.kind].entrance
    _, R_now = arm.tcp()
    yaw = port_yaw_for(in_hand, scene, math.atan2(R_now[1, 0], R_now[0, 0]))
    R = tool_down(yaw)
    hover = entrance + [0, 0, cfg.hover]
    arm.move_to(tcp_for_tip(in_hand, hover + [0, 0, cfg.lift], yaw), R)  # high over the port
    arm.move_line(tcp_for_tip(in_hand, hover, yaw), R)  # straight down to hover
    return PickResult(True, "carry", picked.grasp, in_hand, picked.gap)
