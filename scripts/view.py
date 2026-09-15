"""Interactive viewer with live physics.

Two-connector assembly (perception → pick → insert, round then square). Uses MuJoCo's passive viewer,
which on macOS must be launched with `mjpython`:

    uv run mjpython scripts/view.py --scene assembly                               # realistic camera, 1 mm clearance
    uv run mjpython scripts/view.py --scene assembly --seed 7 --condition ideal --clearance 0.003
    uv run mjpython scripts/view.py --scene assembly --speed 0.5                   # half real time
    uv run python   scripts/view.py --scene assembly --hold                        # just look at the scene

Insertion-only scene (weeks 1–5 controllers; managed viewer, plain python works):

    uv run python scripts/view.py                                            # hold the home pose
    uv run python scripts/view.py --controller straight                      # open-loop insert
    uv run python scripts/view.py --controller spiral --clearance 0.001 --uncertainty 0.005 --seed 1
    uv run python scripts/view.py --controller guarded --joint-noise 1.0     # start from a perturbed pose
    uv run python scripts/view.py --controller impedance --clearance 0.001 --uncertainty 0.002 --seed 1
                                                                              # watch the chamfer guide a compliant peg in

In the viewer: drag to orbit, right-drag to pan, scroll to zoom. Double-click a body to select it, then
Ctrl + right-drag to push it (try shoving the arm mid-insertion). In the insertion scene, Space pauses and
loading key 0 ("home") restarts the controller. The assembly run narrates its progress in the terminal.
"""

import argparse
import sys
import time
from dataclasses import replace

import mujoco
import mujoco.viewer

from insertion.assembly_scene import build_assembly_model, sample_layout
from insertion.assembly_sim import AssemblySim
from insertion.assembly_skills import AssemblyRun
from insertion.camera import DepthNoise
from insertion.config import AssemblyConfig, JointPDConfig, SceneConfig, SensorConfig, StartConfig, TrialConfig, UncertaintyConfig
from insertion.control.joint_pd import joint_pd_torque
from insertion.controllers import CONTROLLERS
from insertion.harness import InsertionTarget, believed_entrance, observe, perturbed_start, tare_while_holding
from insertion.scene import build_model
from insertion.sensors import FTSensor
from insertion.sim import Sim

CONDITIONS = {
    "ideal": DepthNoise(),
    "realistic": DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5),
}


# --- insertion-only scene (managed viewer + control callback) ---------------------------------------------


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
    q_hold = sim.q()
    pd = JointPDConfig()

    def callback(model, data):
        sim.model, sim.data = model, data
        data.ctrl[: len(q_hold)] = joint_pd_torque(sim, q_hold, pd)

    return callback


def run_managed(sim: Sim, callback) -> None:
    mujoco.set_mjcb_control(callback)
    try:
        mujoco.viewer.launch(sim.model, sim.data)
    finally:
        mujoco.set_mjcb_control(None)


# --- assembly pipeline (passive viewer, the pipeline drives the simulation) ----------------------------------


class ViewerClosed(Exception):
    pass


class ViewerPacer:
    """Physics-step hook: pushes frames to the passive viewer and keeps the run at `speed`× real time."""

    def __init__(self, viewer, sim: AssemblySim, speed: float, fps: float = 60.0):
        self.viewer, self.speed = viewer, speed
        self.frame_dt = 1.0 / fps
        self.wall_start = time.perf_counter()
        self.sim_start = sim.time
        self.last_frame = -1.0

    def __call__(self, sim: AssemblySim) -> None:
        if sim.time - self.last_frame < self.frame_dt * self.speed:
            return
        self.last_frame = sim.time
        if not self.viewer.is_running():
            raise ViewerClosed
        self.viewer.sync()
        lag = (sim.time - self.sim_start) / self.speed - (time.perf_counter() - self.wall_start)
        if lag > 0:
            time.sleep(lag)


class NarratedRun(AssemblyRun):
    """AssemblyRun that prints what each skill is doing and how it went."""

    def _say(self, message: str) -> None:
        print(f"[t = {self.robot.time:6.1f} s] {message}", flush=True)

    def perceive(self):
        self._say("capture overhead image and estimate poses")
        scene = super().perceive()
        found = ", ".join(f"{c.kind} at ({c.top[0]:+.3f}, {c.top[1]:+.3f})" for c in scene.connectors)
        self._say(f"  connectors: {found or 'none'} | sockets: {', '.join(s.kind for s in scene.sockets) or 'none'}")
        return scene

    def pick(self, kind, scene):
        self._say(f"pick the {kind} connector (antipodal grasp)")
        in_hand = super().pick(kind, scene)
        log = self._log(kind)
        self._say(f"  grasped={log.grasped} lifted={bool(log.lifted)} in-hand tip error {log.in_hand_tip_error_mm:.2f} mm {log.failure}")
        return in_hand

    def insert(self, kind, scene, in_hand):
        self._say(f"insert into the {kind} socket (impedance + spiral search)")
        super().insert(kind, scene, in_hand)
        log = self._log(kind)
        searched = "searched" if log.search_info.get("searched") else "no search needed"
        self._say(f"  seated={log.seated} after {log.insert_time:.1f} s, {searched}, peak force {log.peak_force_10ms:.1f} N {log.failure}")

    def release_and_retract(self, kind):
        self._say(f"release the {kind} connector and retract")
        super().release_and_retract(kind)

    def move_to_ready(self):
        self._say("move to the ready pose (out of the camera's view)")
        super().move_to_ready()

    def verify(self, kind, scene):
        super().verify(kind, scene)
        log = self._log(kind)
        self._say(f"  visual check: {kind} top {log.visual_height_mm:.1f} mm above its socket (30 mm when seated), "
                  f"{log.visual_offset_mm:.1f} mm off centre → seated_visual={log.seated_visual}, verified={log.verified}")


def run_assembly(args) -> None:
    cfg = replace(sample_layout(AssemblyConfig(), args.seed), clearance=args.clearance, friction=args.friction)
    if args.hold:
        sim = AssemblySim(build_assembly_model(cfg), cfg)
        sim.reset()
        run_managed(sim, make_hold_callback(sim))
        return

    run = NarratedRun(cfg, seed=args.seed, noise=CONDITIONS[args.condition])
    try:
        viewer = mujoco.viewer.launch_passive(run.sim.model, run.sim.data)
    except RuntimeError as err:
        if "mjpython" in str(err):
            sys.exit("On macOS the assembly viewer needs mjpython:\n  uv run mjpython scripts/view.py " + " ".join(sys.argv[1:]))
        raise
    with viewer:
        viewer.cam.lookat[:] = [-0.04, 0.50, 0.23]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.9, 145, -30
        run.robot.on_step.append(ViewerPacer(viewer, run.sim, args.speed))
        print(f"Assembly: seed {args.seed}, {args.condition} camera, {cfg.clearance * 1e3:.0f} mm clearance, friction {cfg.friction}")
        try:
            result = run.run()
        except ViewerClosed:
            print("Viewer closed.")
            return
        print("\nResult: " + "   ".join(f"{log.kind}: {'seated' if log.stayed_seated else 'FAILED ' + log.failure} (robot says {'done' if log.verified else 'not done'})" for log in result.connectors))
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=["insertion", "assembly"], default="insertion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--friction", type=float, default=SceneConfig().friction, help="sliding friction coefficient")
    parser.add_argument("--clearance", type=float, default=None, help="total clearance [m] (insertion default 10 mm, assembly 1 mm)")
    # assembly
    parser.add_argument("--condition", choices=list(CONDITIONS), default="realistic", help="assembly: camera noise model")
    parser.add_argument("--speed", type=float, default=1.0, help="assembly: playback speed relative to real time")
    parser.add_argument("--hold", action="store_true", help="assembly: just show the scene with the arm holding still")
    # insertion
    parser.add_argument("--controller", choices=CONTROLLERS, help="insertion: run this controller live (default: hold home)")
    parser.add_argument("--uncertainty", type=float, default=0.0, help="insertion: max xy offset of believed socket pose [m]")
    parser.add_argument("--joint-noise", type=float, default=0.0, help="insertion: start-pose perturbation [rad]")
    args = parser.parse_args()

    if args.scene == "assembly":
        args.clearance = 0.001 if args.clearance is None else args.clearance
        run_assembly(args)
        return

    cfg = TrialConfig(
        seed=args.seed,
        scene=SceneConfig(clearance=0.010 if args.clearance is None else args.clearance, friction=args.friction),
        uncertainty=UncertaintyConfig(translation_max=args.uncertainty),
        start=StartConfig(joint_noise=args.joint_noise),
    )
    sim = Sim(build_model(cfg.scene))
    sim.reset()
    run_managed(sim, make_controller_callback(sim, cfg, args.controller) if args.controller else make_hold_callback(sim))


if __name__ == "__main__":
    main()
