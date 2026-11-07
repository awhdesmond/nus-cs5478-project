from dataclasses import dataclass, field


@dataclass
class PlanarPose:
    """A pose on the table: position x, y [m] and rotation about the vertical axis, yaw [rad]."""

    x: float
    y: float
    yaw: float = 0.0


PEGS = ("round", "square")

@dataclass
class PegPose(PlanarPose):
    """A peg's pose on the table.

    Upright: standing on its marked (insertion) end, turned by yaw. Lying: on its side (the square peg on a
    face) with its long axis horizontal along (cos yaw, sin yaw), pointing from the marked end to the top.
    """

    lying: bool = False   # inherits x, y, yaw from PlanarPose and adds this one field


@dataclass
class SceneConfig:
    # --- Physics ---------------------------------------------------------------------------
    # 1 ms: small enough for stiff mm-scale contacts. The menagerie default is 2 ms.
    timestep: float = 0.001

    # Elliptic friction cones model Coulomb friction more faithfully than MuJoCo's default
    # pyramid approximation, which matters when a peg slides on a chamfer.
    elliptic_cone: bool = True


    # Sliding friction on every peg/board/table surface. 0.3 ≈ tin-plated pins in plastic (main setting);
    # 0.8 (dry steel on steel) is the proposal's high-friction condition.
    friction: float = 0.3

    # Contact stiffness. MuJoCo contacts are soft springs;
    # solref = (time constant [s], damping ratio),
    # solimp = how hard the contact gets as it penetrates.
    contact_solref: tuple[float, float] = (0.002, 1.0)
    contact_solimp: tuple[float, float, float, float, float] = (0.99, 0.999, 0.001, 0.5, 2.0)

    # --- Pegs (their cross-sections size the ports) -----------------------------------------
    # Keyed at 1 mm clearance: the 10 mm round peg is wider than the 9.5 mm square hole, and the square
    # peg's 12.0 mm diagonal is wider than the round hole's widest point.
    round_radius: float = 0.005  # [m]
    square_half: float = 0.00425  # [m] half of 8.5 mm across the flats
    peg_length: float = 0.050  # [m] marked end to top

    # A blue band (visual only: no mass, no contact) marks each peg's insertion end, so the camera can tell
    # which way a lying peg points.
    tip_band_length: float = 0.008  # [m]

    # A lying cylinder rocks and rolls away in MuJoCo with sliding friction alone.
    # A little rolling friction, like a real plastic part's, holds it still.
    round_rolling_friction: float = 0.001  # [m]

    # Default poses; sample_layout (sim/layout.py) draws new ones per trial.
    peg_poses: dict[str, PegPose] = field(default_factory=lambda: {
        "round": PegPose(0.05, 0.45),
        "square": PegPose(0.12, 0.55, 0.3),
    })


    # --- Ports -----------------------------------------------------------------------------
    clearance: float = 0.001  # [m] hole width minus peg width (proposal grid: 1-3 mm)

    # MuJoCo collides with a mesh's convex hull, which would fill a hole in, so holes are built from
    # boxes; the round hole is a 16-sided polygon.
    round_sides: int = 16
    port_depth: float = 0.020  # [m] hole floor to board top
    port_chamfer: float = 0.002  # [m] 45° lead-in around each hole's rim
    port_wall: float = 0.006  # [m] solid material between a hole's rim and the next box (≥ 2 × chamfer)
    port_base: float = 0.010  # [m] solid plate under each hole (it's the hole's floor)

    # Each port is its own block on the table. Pose = hole centre (x, y) and the block's yaw;
    # randomised per trial.
    port_poses: dict[str, PlanarPose] = field(default_factory=lambda: {
        "round": PlanarPose(-0.16, 0.47, 0.0),
        "square": PlanarPose(-0.10, 0.53, 0.4),
    })

    # --- Gripper (a parallel jaw on the flange; the fingers close along the gripper's own y axis) --------
    finger_open_gap: float = 0.030  # [m] between the finger pads when fully open: room for a 10 mm peg ± 10 mm
    finger_length: float = 0.050  # [m]
    finger_width: float = 0.016  # [m] along the gripper's x
    finger_thickness: float = 0.006  # [m] along the closing direction
    palm_offset: float = 0.030  # [m] flange to the finger roots
    pad_depth: float = 0.008  # [m] the tool centre point (tcp) sits this far above the fingertips
    grip_force: float = 20.0  # [N] the finger motor's force limit

    # Closing speed limit, as on a real gripper. Without it the fingers slammed shut in ~27 ms in the first
    # attempt, bit into the peg unevenly and knocked it off-centre in the hand.
    finger_speed: float = 0.02  # [m/s] per finger
    finger_friction: float = 1.0  # rubber-like pads


    # --- Layout areas --------
    # Pick zone: where the pegs start. Lying pegs must lie wholly inside it.
    pick_area_center: tuple[float, float] = (0.085, 0.50)
    pick_area_half: tuple[float, float] = (0.075, 0.10)

    # Between peg axes (centres if upright): open fingers reach 21 mm either side of a peg being grasped,
    # so this keeps them off the other peg.
    min_peg_spacing: float = 0.06  # [m]

    # Port area: where the port blocks' centres go, left of the pick zone, with room for a block at any yaw
    # to stay in the camera's view.
    port_area_center: tuple[float, float] = (-0.15, 0.50)
    port_area_half: tuple[float, float] = (0.05, 0.08)

    # Between port centres: fingers inserting the second peg (±10 mm when closed) stay clear of the first peg,
    # which stands 30 mm out of its port.
    min_port_spacing: float = 0.06  # [m]

    # --- Table -----------------------------------------------------------------------------
    table_height: float = 0.17  # [m] table top above the floor
    table_center: tuple[float, float] = (-0.05, 0.50)  # [m] in front of the arm's base (at the origin)
    table_half: tuple[float, float] = (0.25, 0.15)  # [m] a 500 × 300 mm table