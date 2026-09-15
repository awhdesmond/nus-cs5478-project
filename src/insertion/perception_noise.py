"""Independent per-object pose errors on top of any estimator (M5 stress condition).

The overhead camera's calibration error shifts every estimate the same way, so it largely cancels between a
connector and its socket and barely stresses insertion. This wrapper adds a fixed, independent xy offset to each
object (round/square connector, round/square socket), drawn once per run, as a model-based estimator with
per-object registration errors would have. The same offset applies to every capture of that object, including
after it has been inserted, so the visual seating check sees a consistent error.
"""

from dataclasses import replace

import numpy as np

from insertion.camera import CameraObservation
from insertion.perception import PoseEstimator, SceneEstimate


class ObjectOffsetEstimator:
    def __init__(self, base: PoseEstimator, translation: float, rng: np.random.Generator):
        """translation: offsets are uniform in ±translation per horizontal axis [m]."""
        self.base = base
        self.translation = translation
        self.offsets = {
            (part, kind): rng.uniform(-translation, translation, size=2)
            for part in ("connector", "socket") for kind in ("round", "square")
        }

    def estimate(self, observation: CameraObservation) -> SceneEstimate:
        scene = self.base.estimate(observation)
        if self.translation <= 0:
            return scene
        connectors = []
        for c in scene.connectors:
            d = self.offsets[("connector", c.kind)]
            d3 = np.array([*d, 0.0])
            connectors.append(replace(
                c, top=c.top + d3, outline_xy=c.outline_xy + d,
                centre=None if c.centre is None else c.centre + d3,
                fit=replace(c.fit, centre_xy=c.fit.centre_xy + d),
            ))
        sockets = []
        for s in scene.sockets:
            d = self.offsets[("socket", s.kind)]
            sockets.append(replace(s, entrance=s.entrance + np.array([*d, 0.0]), fit=replace(s.fit, centre_xy=s.fit.centre_xy + d)))
        return replace(scene, connectors=connectors, sockets=sockets)
