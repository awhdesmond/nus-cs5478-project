"""Interactive viewer with live physics.

Two-connector assembly (perception → pick → insert, round then square). Loops over layouts (seed, seed + 1, …)
in one window until you close it; connectors are upright or lying (marked insertion end in blue) at random by
default. Uses MuJoCo's passive viewer, which on macOS must be launched with `mjpython`:

    uv run mjpython scripts/view.py --scene assembly                               # loop: mixed orientations, realistic camera, 1 mm
    uv run mjpython scripts/view.py --scene assembly --orientation lying --speed 2 # all lying, twice real time
    uv run mjpython scripts/view.py --scene assembly --seed 7 --once --condition ideal --clearance 0.003
    uv run mjpython scripts/view.py --scene assembly --orientation upright         # the M3/M4 task
    uv run mjpython scripts/view.py --scene assembly --cable                       # add the deformable cable (scenery)
    uv run python   scripts/view.py --scene assembly --hold --cable                # just look at the scene; poke the cable

Insertion-only scene (weeks 1–5 controllers; managed viewer, plain python works):

    uv run python scripts/view.py                                            # hold the home pose
    uv run python scripts/view.py --controller straight                      # open-loop insert
    uv run python scripts/view.py --controller spiral --clearance 0.001 --uncertainty 0.005 --seed 1
    uv run python scripts/view.py --controller guarded --joint-noise 1.0     # start from a perturbed pose
    uv run python scripts/view.py --controller impedance --clearance 0.001 --uncertainty 0.002 --seed 1
                                                                              # watch the chamfer guide a compliant peg in

The top-right corner shows the overhead camera live and, below it, what perception made of the last capture
(detected outlines labelled round/square, socket centres, the chosen grasp); `--no-camera` hides both.

In the viewer: drag to orbit, right-drag to pan, scroll to zoom. Double-click a body to select it, then
Ctrl + right-drag to push it (try shoving the arm mid-insertion). In the insertion scene, Space pauses and
loading key 0 ("home") restarts the controller. The assembly run narrates its progress in the terminal.
"""

import argparse
import sys
import time
from dataclasses import replace

import cv2
import mujoco
import mujoco.viewer
import numpy as np

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
from pipeline_demo import perception_overlay

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


class CameraPanels:
    """Picture-in-picture overlays: the live overhead camera and the perception result of the last capture."""

    def __init__(self, viewer, model: mujoco.MjModel, live_hz: float = 10.0):
        self.viewer = viewer
        self.renderer = mujoco.Renderer(model, 480, 640)
        self.live_period = 1.0 / live_hz
        self.last_live = -np.inf
        self.live = None
        self.perceived = None
        self.capture_key = None

    def close(self) -> None:
        self.renderer.close()

    @staticmethod
    def _label(image: np.ndarray, text: str) -> np.ndarray:
        image = image.copy()
        cv2.rectangle(image, (0, 0), (image.shape[1], 26), (20, 22, 28), thickness=-1)
        cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        return image

    def update(self, data: mujoco.MjData, run: AssemblyRun | None) -> None:
        viewport = self.viewer.viewport
        if viewport is None or viewport.width < 100:
            return
        width = max(240, viewport.width // 4)
        height = width * 3 // 4
        margin = max(8, viewport.width // 200)
        now = time.perf_counter()
        if self.live is None or now - self.last_live >= self.live_period:
            self.last_live = now
            self.renderer.update_scene(data, camera="overhead")
            self.live = self.renderer.render().copy()
        panels = [(self._label(self.live, "overhead camera (live)"), 0)]
        if run is not None and run.result.captures:
            t, observation, scene, proposals = run.result.captures[-1]
            key = (len(run.result.captures), len(proposals))
            if key != self.capture_key or self.perceived is None or self.perceived.shape[1] != 640:
                self.capture_key = key
                self.perceived = perception_overlay(observation, scene, proposals, (640, 480))
            panels.append((self._label(self.perceived, f"perception at t = {t:.1f} s"), 1))
        images = []
        for image, row in panels:
            rect = mujoco.MjrRect(viewport.left + viewport.width - width - margin,
                                  viewport.bottom + viewport.height - (row + 1) * (height + margin), width, height)
            images.append((rect, cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)))
        self.viewer.set_images(images)


class ViewerPacer:
    """Physics-step hook: pushes frames to the passive viewer and keeps the run at `speed`× real time."""

    def __init__(self, viewer, sim: AssemblySim, speed: float, fps: float = 60.0, panels: CameraPanels | None = None,
                 run: AssemblyRun | None = None):
        self.viewer, self.speed = viewer, speed
        self.panels, self.run = panels, run
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
        if self.panels is not None:
            self.panels.update(sim.data, self.run)
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
        found = ", ".join(f"{c.kind} {'lying' if c.lying else 'upright'} at ({c.top[0]:+.3f}, {c.top[1]:+.3f})" for c in scene.connectors)
        self._say(f"  connectors: {found or 'none'} | sockets: {', '.join(s.kind for s in scene.sockets) or 'none'}")
        return scene

    def pick(self, kind, scene):
        estimate = scene.connector(kind)
        how = "across its width, near the unmarked end" if estimate is not None and estimate.lying else "antipodal grasp"
        self._say(f"pick the {kind} connector ({how})")
        in_hand = super().pick(kind, scene)
        log = self._log(kind)
        self._say(f"  grasped={log.grasped} lifted={bool(log.lifted)} in-hand tip error {log.in_hand_tip_error_mm:.2f} mm {log.failure}")
        return in_hand

    def _turn_sideways(self, kind, scene, socket):
        self._say(f"  plan a sideways insertion (clearance to perceived obstacles) and turn the hand")
        yaw = super()._turn_sideways(kind, scene, socket)
        log = self._log(kind)
        if yaw is not None:
            self._say(f"  fingers pointing {log.sideways_yaw_deg:.0f}°, planned clearance {log.sideways_clearance_mm:.1f} mm")
        return yaw

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
    base = replace(AssemblyConfig(), clearance=args.clearance, friction=args.friction, cable=args.cable)
    cfg = sample_layout(base, args.seed, orientation=args.orientation)
    sim = AssemblySim(build_assembly_model(cfg), cfg)
    sim.reset()
    if args.hold:
        run_managed(sim, make_hold_callback(sim))
        return

    try:
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    except RuntimeError as err:
        if "mjpython" in str(err):
            sys.exit("On macOS the assembly viewer needs mjpython:\n  uv run mjpython scripts/view.py " + " ".join(sys.argv[1:]))
        raise
    with viewer:
        viewer.cam.lookat[:] = [-0.04, 0.50, 0.23]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.9, 145, -30
        panels = None if args.no_camera else CameraPanels(viewer, sim.model)
        seed, tally = args.seed, []
        while True:
            # Same model and data every run (the viewer is attached to them); only the layout changes.
            cfg = sample_layout(base, seed, orientation=args.orientation)
            run = NarratedRun(cfg, seed=seed, noise=CONDITIONS[args.condition], sim=sim)
            pacer = ViewerPacer(viewer, sim, args.speed, panels=panels, run=run)
            run.robot.on_step.append(pacer)
            lying = [kind for kind, pose in (("round", cfg.round_pose), ("square", cfg.square_pose)) if pose.lying]
            print(f"\n=== Assembly seed {seed}: {args.condition} camera, {cfg.clearance * 1e3:.0f} mm clearance, friction {cfg.friction}, "
                  f"lying: {', '.join(lying) or 'none'} ===")
            try:
                result = run.run()
                run._play([], settle=args.pause)  # hold at the ready pose so the finished layout can be seen
            except ViewerClosed:
                print("Viewer closed.")
                if panels is not None:
                    panels.close()
                return
            finally:
                run.robot.camera.renderer.close()
            tally.append(result.both_seated_at_end)
            print("Result: " + "   ".join(f"{log.kind}: {'seated' if log.stayed_seated else 'FAILED ' + log.failure} "
                                          f"(robot says {'done' if log.verified else 'not done'})" for log in result.connectors))
            print(f"Both seated in {sum(tally)}/{len(tally)} runs so far.")
            if args.once:
                break
            seed += 1
        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            if panels is not None:
                panels.update(sim.data, run)
            viewer.sync()
            time.sleep(1 / 60)
        if panels is not None:
            panels.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=["insertion", "assembly"], default="insertion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--friction", type=float, default=SceneConfig().friction, help="sliding friction coefficient")
    parser.add_argument("--clearance", type=float, default=None, help="total clearance [m] (insertion default 10 mm, assembly 1 mm)")
    # assembly
    parser.add_argument("--condition", choices=list(CONDITIONS), default="realistic", help="assembly: camera noise model")
    parser.add_argument("--orientation", choices=["upright", "lying", "mixed"], default="mixed", help="assembly: connector orientations")
    parser.add_argument("--once", action="store_true", help="assembly: stop after one run (default: loop over seeds)")
    parser.add_argument("--pause", type=float, default=2.0, help="assembly: seconds to show each finished layout")
    parser.add_argument("--speed", type=float, default=1.0, help="assembly: playback speed relative to real time")
    parser.add_argument("--hold", action="store_true", help="assembly: just show the scene with the arm holding still")
    parser.add_argument("--cable", action="store_true", help="assembly: add the deformable cable at the table's left edge")
    parser.add_argument("--no-camera", action="store_true", help="assembly: hide the camera and perception overlays")
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
