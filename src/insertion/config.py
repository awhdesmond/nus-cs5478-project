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
    friction: float = 0.8  # PLACEHOLDER
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
