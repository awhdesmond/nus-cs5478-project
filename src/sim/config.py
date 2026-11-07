from dataclasses import dataclass


@dataclass
class SceneConfig:
    # --- Physics ---------------------------------------------------------------------------
    # 1 ms: small enough for stiff mm-scale contacts. The menagerie default is 2 ms.
    timestep: float = 0.001

    # Elliptic friction cones model Coulomb friction more faithfully than MuJoCo's default
    # pyramid approximation, which matters when a peg slides on a chamfer.
    elliptic_cone: bool = True
