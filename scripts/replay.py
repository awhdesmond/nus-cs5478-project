"""Replay a recorded trial: interactive viewer or MP4.

Kinematic replay: recorded joint angles are written back into the scene each frame (exact,
since the peg is welded to the arm). Trajectories come from `run_trials(record_dir=...)`.

    # MP4 (wide view + socket close-up, with time and contact force overlaid)
    uv run python scripts/replay.py results/easy_insert/trajectories/trial_0000_seed0.npz --video demo.mp4

    # Interactive viewer (macOS needs mjpython)
    uv run mjpython scripts/replay.py results/easy_insert/trajectories/trial_0000_seed0.npz
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from insertion.config import SceneConfig
from insertion.scene import build_model
from insertion.sim import Sim


def load(path: Path):
    traj = np.load(path)
    scene = SceneConfig(**json.loads(str(traj["config"]))["scene"])
    sim = Sim(build_model(scene))
    sim.reset()
    return sim, traj


def set_frame(sim: Sim, q: np.ndarray) -> None:
    sim.data.qpos[sim.qpos_adr] = q
    mujoco.mj_forward(sim.model, sim.data)


def frame_indices(times: np.ndarray, fps: float, speed: float) -> np.ndarray:
    """Recorded-step index for each output frame at `fps`, playing at `speed`× real time."""
    frame_times = np.arange(0.0, times[-1], speed / fps)
    return np.searchsorted(times, frame_times).clip(0, len(times) - 1)


def camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render_video(sim: Sim, traj, out: Path, fps: float, speed: float) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    height, width = 540, 720
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(sim.model, height, width)
    socket, _ = sim.site_pose("socket_entrance")
    views = [
        camera(socket + [0.05, -0.15, 0.20], 1.3, 120, -20),  # whole arm
        camera(socket + [0, 0, 0.005], 0.16, 135, -8),  # socket close-up, low so the wrist doesn't hide the entrance
    ]
    times, forces = traj["time"], np.linalg.norm(traj["contact_force"], axis=1)
    font = ImageFont.load_default(size=22)
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as writer:
        for i in frame_indices(times, fps, speed):
            set_frame(sim, traj["q"][i])
            tiles = []
            for cam in views:
                renderer.update_scene(sim.data, cam)
                tiles.append(renderer.render())
            image = Image.fromarray(np.hstack(tiles))
            ImageDraw.Draw(image).text(
                (14, 10), f"t = {times[i]:5.2f} s   contact force = {forces[i]:5.1f} N", fill=(255, 255, 255), font=font
            )
            writer.append_data(np.asarray(image))
    print(f"Wrote {out} ({times[-1]:.1f} s sim at {speed}x)")


def view_interactive(sim: Sim, traj, speed: float) -> None:
    import mujoco.viewer

    socket, _ = sim.site_pose("socket_entrance")
    times = traj["time"]
    idx = frame_indices(times, 60.0, speed)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = socket
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 135, -25
        while viewer.is_running():
            for i in idx:
                if not viewer.is_running():
                    break
                set_frame(sim, traj["q"][i])
                viewer.sync()
                time.sleep(1 / 60)
            time.sleep(1.0)  # pause, then loop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--video", type=Path, help="write an MP4 instead of opening the viewer")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    sim, traj = load(args.trajectory)
    if args.video:
        render_video(sim, traj, args.video, args.fps, args.speed)
        return
    try:
        view_interactive(sim, traj, args.speed)
    except RuntimeError as err:
        if "mjpython" in str(err):
            raise SystemExit(f"On macOS the interactive viewer needs mjpython:\n  uv run mjpython {' '.join(__import__('sys').argv)}")
        raise


if __name__ == "__main__":
    main()
