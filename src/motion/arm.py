"""The arm as a task-level controller sees it: move the tool, work the gripper, read its own joints.

Everything here uses only what a real robot knows about itself (joint encoders, finger positions, its own
kinematic model), never the positions of the pegs or ports in the simulator.
"""

import numpy as np

from motion.control import JointTracker, follow
from motion.kinematics import Kinematics
from motion.trajectory import CartesianMove, JointMove
from sim.sim import Sim


class Arm:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.kin = Kinematics(sim.model)
        self.tracker = JointTracker(sim)

    def tcp(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the tool is (position, orientation), from the joint encoders and forward kinematics."""
        return self.kin.fk(self.sim.q())

    def move_to(self, position, R: np.ndarray, max_joint_speed: float = 1.0) -> None:
        """Joint-space move: fast, but the tool's path curves. For moves away from everything."""
        result = self.kin.ik(position, R, self.sim.q())
        if not result.converged:
            raise ValueError(f"no IK solution for {position}")
        follow(self.sim, self.tracker, JointMove(self.sim.q(), result.q, max_joint_speed=max_joint_speed))

    def move_line(self, position, R: np.ndarray, max_speed: float = 0.05) -> None:
        """Straight-line tool move. For moves near the table, a peg or a port."""
        follow(self.sim, self.tracker, CartesianMove(self.kin, self.sim.q(), position, R, max_speed=max_speed))

    def hold(self, seconds: float) -> None:
        """Keep the arm still (it keeps controlling: gravity doesn't take over) while time passes."""
        q = self.sim.q()
        for _ in range(round(seconds / self.sim.model.opt.timestep)):
            self.sim.step(self.tracker.torque(q, np.zeros(6), np.zeros(6)))

    def grip(self, close: bool, timeout: float = 1.5, still: float = 0.05) -> float:
        """Close or open the fingers and wait until they stop moving. Returns the gap between the pads [m].

        "Stopped" = the gap changed by under 0.05 mm over the last `still` seconds. Closed on a peg, the gap is
        the peg's width; fully closed (gap ~0), the fingers missed.
        """
        sim, dt = self.sim, self.sim.model.opt.timestep
        sim.command_gripper(close)
        q = sim.q()
        window = round(still / dt)
        gaps = []
        for _ in range(round(timeout / dt)):
            sim.step(self.tracker.torque(q, np.zeros(6), np.zeros(6)))
            gaps.append(sim.finger_gap())
            if len(gaps) > 4 * window and abs(gaps[-1] - gaps[-window]) < 5e-5:
                break
        return sim.finger_gap()
