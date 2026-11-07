from dataclasses import dataclass, field


@dataclass
class PlanarPose:
    """A pose on the table: position x, y [m] and rotation about the vertical axis, yaw [rad]."""

    x: float
    y: float
    yaw: float = 0.0


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

    # --- Ports -----------------------------------------------------------------------------
    clearance: float = 0.001  # [m] hole width minus peg width (proposal grid: 1–3 mm)

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

    # --- Table -----------------------------------------------------------------------------
    table_height: float = 0.17  # [m] table top above the floor
    table_center: tuple[float, float] = (-0.05, 0.50)  # [m] in front of the arm's base (at the origin)
    table_half: tuple[float, float] = (0.25, 0.15)  # [m] a 500 × 300 mm table