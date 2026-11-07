"""Perception's single entry point: one RGB-D observation in, everything a controller needs out."""

from dataclasses import dataclass

from perception.camera import CameraObservation
from perception.pegs import PegConfig, PegEstimate, estimate_pegs
from perception.ports import PortConfig, PortEstimate, estimate_ports
from perception.segment import SegmentConfig, segment


@dataclass(frozen=True)
class PerceptionConfig:
    segment: SegmentConfig = SegmentConfig()
    pegs: PegConfig = PegConfig()
    ports: PortConfig = PortConfig()


@dataclass
class SceneEstimate:
    table_z: float  # [m] measured table surface height
    pegs: dict[str, PegEstimate]  # by kind: "round", "square"
    ports: dict[str, PortEstimate]


def perceive(obs: CameraObservation, cfg: PerceptionConfig = PerceptionConfig()) -> SceneEstimate:
    """Segment once, then find the pegs and the ports. Raises RuntimeError if it can't find two of each."""
    seg = segment(obs, cfg.segment)
    return SceneEstimate(seg.table_z, estimate_pegs(obs, seg, cfg.pegs), estimate_ports(obs, seg, cfg.ports))
