"""Scene, task and evaluation parameters.

Every tunable number lives here so the harness can sweep it and each logged trial
records the exact config it ran with. Values marked PLACEHOLDER are starting guesses
to revisit during week 1.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE_DIR = ROOT / "third_party" / "mujoco_menagerie"
UR5E_XML = MENAGERIE_DIR / "universal_robots_ur5e" / "ur5e.xml"
UR5E_SCENE_XML = MENAGERIE_DIR / "universal_robots_ur5e" / "scene.xml"
RESULTS_DIR = ROOT / "results"


@dataclass
class SceneConfig:
    # Geometry [m]. Clearance is total (diameter vs. hole width), ratcheted 10 → 5 → 3 → 1 mm.
    clearance: float = 0.010
    peg_radius: float = 0.005
    peg_length: float = 0.040
    socket_depth: float = 0.020  # hole depth, bottom to top surface
    socket_wall: float = 0.015  # wall thickness around the hole
    socket_base: float = 0.010  # base plate thickness under the hole
    socket_chamfer: float = 0.002  # 45° lead-in leg length; 0 disables
    # Hole-bottom centre in world frame. A pedestal fills the gap down to the floor.
    # Chosen below the home pose with the peg pointing down: along pre-insert → seated the elbow stays
    # ≈2.0 rad, wrist_2 at −π/2 and the wrist 0.43 m off the base axis (clear of all UR singularities).
    socket_pos: tuple[float, float, float] = (-0.15, 0.50, 0.18)

    # Physics
    timestep: float = 0.001  # try 0.0005 if contact jitters
    # Sliding friction. 0.3 ≈ brass/tin-plated pins in plastic housings (main setting); 0.8 (dry steel-on-steel)
    # is used as a high-friction benchmark. Dominant factor for chamfer-guided insertion: see weeks 2–3 findings.
    friction: float = 0.3
    elliptic_cone: bool = True
    # Stiffer-than-default contact, applied to socket AND peg (MuJoCo averages the pair).
    # 2 kg effective mass + 30 N push on the chamfer lip sinks ≤0.21 mm (default params: 1.16 mm,
    # which exceeds the 0.5 mm per-side gap at 1 mm clearance). solref[0] must stay ≥ 2·timestep.
    contact_solref: tuple[float, float] = (0.005, 1.0)
    contact_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)


@dataclass
class JointPDConfig:
    # Per joint: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3.
    # From home: 0.05 rad step on all joints settles to 2% in 0.46 s, ≤7% overshoot, ≤18% torque limit.
    kp: tuple[float, ...] = (400.0, 400.0, 400.0, 100.0, 100.0, 50.0)  # [Nm/rad]
    kd: tuple[float, ...] = (40.0, 40.0, 40.0, 10.0, 10.0, 5.0)  # [Nm·s/rad]


@dataclass
class MotionConfig:
    joint_vmax: float = 1.0  # [rad/s] joint-space moves (UR5e hardware limit ~3.14)
    joint_amax: float = 2.0  # [rad/s²]
    cart_vmax: float = 0.05  # [m/s] free-space Cartesian lines near the socket
    cart_amax: float = 0.5  # [m/s²]
    approach_height: float = 0.05  # pre-insertion tip height above the believed entrance [m]
    via_height: float = 0.10  # via pose above pre-insertion; arbitrary starts lift to this height first [m]
    lift_vmax: float = 0.15  # [m/s] lift and via→pre lines (clear of the socket, so faster)


@dataclass
class ImpedanceConfig:
    """Cartesian impedance at the peg tip: τ = Jᵀ[K·e + D·(ẋ* − ẋ)] + τ_bias − D_q·q̇."""

    # Soft laterally, stiff along the insertion axis. On a 45° chamfer with friction μ, a peg pushed with
    # force P slides at most δ = P(1−μ)/(K_xy(1+μ)) before the lateral spring stalls it, so K_xy must be low.
    # Sweep at 1 mm clearance, ±2 mm: K_xy 1000 → 70–80% success (μ=0.3); K_xy ≤ 300 → 100%, no cap violations.
    stiffness_pos: tuple[float, float, float] = (300.0, 300.0, 1000.0)  # [N/m]
    # 4° commanded tilt pressed at 15 N ends at 0.01° for K_rot ≤ 0.5, 0.9° at 1.0, 2.1° at 2.0, 3.8° at 20.
    stiffness_rot: tuple[float, float, float] = (0.5, 0.5, 0.5)  # [Nm/rad]
    damping_ratio: float = 1.0  # D = 2ζ·sqrt(K·Λ_ii), Λ = task-space inertia at the start of the impedance phase
    joint_damping: float = 0.5  # [Nm·s/rad] small extra damping on every joint
    max_spring_force: float = 15.0  # [N] saturate |K_pos·e| so contact force stays bounded whatever the setpoint


@dataclass
class ImpedanceInsertConfig:
    descend_speed: float = 0.01  # setpoint speed outside the touchdown zone [m/s]
    touchdown_speed: float = 0.0025  # setpoint speed within ±touchdown_zone of the entrance [m/s]
    touchdown_zone: float = 0.003  # [m] impact spike ≈ 2.7 N per mm/s of touchdown speed, so slow down here
    press_depth: float = 0.005  # setpoint ends this far below the nominal hole bottom, so the spring keeps pushing [m]
    # Setpoint leash: the path pauses while the tip lags the setpoint by more than this along the axis,
    # so push force stays ≈ K_z·max_lead. Without it the setpoint winds up while the peg is held on the
    # chamfer, and the stored spring force fires the peg into the hole wall once it drops in (130 N spike).
    max_lead: float = 0.005  # [m]


@dataclass
class SpiralSearchConfig:
    """Contact-triggered Archimedean spiral search under impedance control."""

    search_force: float = 3.0  # [N] normal push while searching (setpoint held search_force/K_z below contact height)
    search_stiffness_xy: float = 1500.0  # [N/m] stiffer laterally while searching so friction drag (μ·F/K) doesn't distort the spiral
    pitch: float = 0.0015  # [m] radial spacing between turns; must not exceed the capture width (clearance + chamfer)
    search_speed: float = 0.01  # [m/s] setpoint speed along the spiral
    max_angular_speed: float = 6.28  # [rad/s] caps the spin near the centre, where v/r blows up
    search_radius: float = 0.0075  # [m] restart the spiral beyond this (covers a ±5 mm square)
    partial_drop: float = 0.0005  # [m] new drop since the spiral last resumed that pauses it to let the chamfer act
    hole_drop: float = 0.0025  # [m] tip below contact height that confirms the hole → insert (just past the 2 mm chamfer depth)
    # Lateral leash: the spiral stops advancing while the tip lags the spiral setpoint by more than this, capping
    # sideways force at ≈ search_stiffness_xy·max_lateral_lead. Without it a peg that has dropped into the top of
    # the hole gets dragged into the wall at ~15 N, and wall friction then holds it up (jam).
    max_lateral_lead: float = 0.002  # [m]
    pause_time: float = 0.4  # [s] pause length after a partial drop or stall; extended while the peg keeps dropping
    progress_drop: float = 0.0002  # [m] drop during a pause that counts as progress (extends the pause)
    stall_timeout: float = 0.3  # [s] continuous leash stall that triggers a centring pause
    # Force-guided centring while paused: pressed on the chamfer, the sensed lateral contact force points toward
    # the hole axis, so the xy setpoint moves along it. Unhooks pegs caught on the chamfer near the hole's corners.
    center_gain: float = 0.002  # [m/s per N]
    center_speed_max: float = 0.005  # [m/s]


@dataclass
class GuardedRetryConfig:
    """Guarded-move + random-retry baseline: descend until contact; if blocked, retry elsewhere."""

    hover_height: float = 0.004  # tip height above believed entrance between attempts [m]
    descend_speed: float = 0.01  # guarded descent speed [m/s]
    contact_force: float = 3.0  # |F| from wrist F/T that ends a guarded descent [N]
    search_radius: float = 0.005  # retry xy sampled uniformly in ±radius around the believed entrance [m]
    max_attempts: int = 100
    stop_above_bottom: float = 0.0005  # descent target just above nominal hole bottom [m]


@dataclass
class SensorConfig:
    ft_cutoff_hz: float = 30.0  # first-order low-pass on the wrist F/T signal PLACEHOLDER
    tare_steps: int = 50  # samples averaged while holding still before motion
    contact_on: float = 2.0  # [N] filtered |F| that declares contact
    contact_off: float = 1.0  # [N] hysteresis: contact released below this
    contact_debounce: int = 3  # consecutive samples above `contact_on` before declaring contact


@dataclass
class UncertaintyConfig:
    # Offset between believed and true socket pose; the controller only sees the believed pose.
    translation_max: float = 0.0  # [m], ratchet 0 → small → 0.005–0.010
    rotation_max_deg: float = 0.0  # ratchet after translation, target ±5°


@dataclass
class TaskConfig:
    force_cap: float = 30.0  # [N] PLACEHOLDER
    cycle_time_budget: float = 60.0  # [s sim time] PLACEHOLDER; see spiral budget note in plan
    contact_step_budget: int = 20_000  # PLACEHOLDER
    seated_depth_tol: float = 0.002  # [m] PLACEHOLDER


@dataclass
class StartConfig:
    joint_noise: float = 0.0  # [rad] uniform perturbation of the home pose per joint (collision-free resampled)


@dataclass
class TrialConfig:
    seed: int = 0
    start: StartConfig = field(default_factory=StartConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def to_dict(self) -> dict:
        return asdict(self)
