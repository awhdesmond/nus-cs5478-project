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
    socket_depth: float = 0.020
    socket_pos: tuple[float, float, float] = (0.45, 0.10, 0.0)  # PLACEHOLDER: keep arm away from singularities

    # Physics
    timestep: float = 0.001  # try 0.0005 if contact jitters
    friction: float = 0.8  # PLACEHOLDER
    elliptic_cone: bool = True


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
class TrialConfig:
    seed: int = 0
    scene: SceneConfig = field(default_factory=SceneConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def to_dict(self) -> dict:
        return asdict(self)
