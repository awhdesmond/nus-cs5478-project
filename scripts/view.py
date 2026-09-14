"""Interactive viewer with live physics.

    uv run python scripts/view.py                                            # hold the home pose
    uv run python scripts/view.py --controller straight                      # open-loop insert
    uv run python scripts/view.py --controller guarded --clearance 0.001 --uncertainty 0.005 --seed 1
                                                                              # watch guarded-retry search
    uv run python scripts/view.py --controller guarded --joint-noise 1.0     # start from a perturbed pose
    uv run python scripts/view.py --controller impedance --clearance 0.001 --uncertainty 0.002 --seed 1 --friction 0.3
                                                                              # watch the chamfer guide a compliant peg in

In the viewer: drag to orbit, scroll to zoom, double-click a body to select it, then
Ctrl + right-drag to push it (try shoving the arm mid-insertion). Space pauses.
To restart: Simulation panel → Key → load key 0 ("home") — the controller restarts itself
whenever sim time jumps backwards (restarts begin from home, not the perturbed start).
"""

import argparse

import mujoco
import mujoco.viewer

from insertion.config import JointPDConfig, SceneConfig, SensorConfig, StartConfig, TrialConfig, UncertaintyConfig
from insertion.control.joint_pd import joint_pd_torque
from insertion.controllers import CONTROLLERS
from insertion.harness import InsertionTarget, believed_entrance, observe, perturbed_start, tare_while_holding
from insertion.scene import build_model
from insertion.sensors import FTSensor
from insertion.sim import Sim


def make_controller_callback(sim: Sim, cfg: TrialConfig, controller_name: str):
    """mjcb_control callback that runs a registered controller on the viewer's live data."""
    perturbed_start(sim, cfg)
    ft = FTSensor(sim, SensorConfig())
    tare_while_holding(sim, ft)
    entrance, _ = sim.site_pose("socket_entrance")
    target = InsertionTarget(believed_entrance(cfg, entrance), cfg.scene.socket_depth)
    controller = CONTROLLERS[controller_name]()
    state = {"t_start": sim.time, "last_time": sim.time}

    def restart():
        controller.reset(sim.model, sim.q(), target, cfg.seed)
        state["t_start"] = sim.time

    restart()

    def callback(model, data):
        sim.model, sim.data = model, data  # viewer may hand over new objects (e.g. after reload)
        if data.time < state["last_time"]:
            restart()  # user reset or loaded a keyframe
        state["last_time"] = data.time
        ft.update()
        data.ctrl[:] = controller.act(observe(sim, ft, state["t_start"]))

    return callback


def make_hold_callback(sim: Sim):
    q_home = sim.q()
    pd = JointPDConfig()

    def callback(model, data):
        sim.model, sim.data = model, data
        data.ctrl[:] = joint_pd_torque(sim, q_home, pd)

    return callback


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=CONTROLLERS, help="run this controller live (default: just hold home)")
    parser.add_argument("--clearance", type=float, default=0.010, help="total clearance [m]")
    parser.add_argument("--friction", type=float, default=SceneConfig().friction, help="sliding friction coefficient")
    parser.add_argument("--uncertainty", type=float, default=0.0, help="max xy offset of believed socket pose [m]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint-noise", type=float, default=0.0, help="start-pose perturbation [rad]")
    args = parser.parse_args()

    cfg = TrialConfig(
        seed=args.seed,
        scene=SceneConfig(clearance=args.clearance, friction=args.friction),
        uncertainty=UncertaintyConfig(translation_max=args.uncertainty),
        start=StartConfig(joint_noise=args.joint_noise),
    )
    sim = Sim(build_model(cfg.scene))
    sim.reset()
    callback = make_controller_callback(sim, cfg, args.controller) if args.controller else make_hold_callback(sim)

    mujoco.set_mjcb_control(callback)
    try:
        mujoco.viewer.launch(sim.model, sim.data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
