"""Replay a recorded trial: interactive viewer or MP4.

Kinematic replay: recorded joint angles are written back into the scene each frame (exact,
since the peg is welded to the arm). Trajectories come from `run_trials(record_dir=...)`.

    # MP4 (wide view + socket close-up + top-down tip path, with time and contact force overlaid)
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

from insertion.config import SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.harness import believed_entrance
from insertion.scene import build_model, hole_width
from insertion.sim import Sim


def load(path: Path):
    traj = np.load(path)
    raw = json.loads(str(traj["config"]))
    cfg = TrialConfig(
        seed=raw["seed"],
        scene=SceneConfig(**raw["scene"]),
        uncertainty=UncertaintyConfig(**raw["uncertainty"]),
        start=StartConfig(**raw.get("start", {})),
        task=TaskConfig(**raw["task"]),
    )
    sim = Sim(build_model(cfg.scene))
    sim.reset()
    return sim, traj, cfg


def trace_tile(size, cfg: TrialConfig, entrance, believed, tips, upto, font):
    """Top-down view of the tip path near the socket (ground truth, for visualisation only)."""
    from PIL import Image, ImageDraw

    half_view = 0.009  # [m] shown around the true hole centre
    img = Image.new("RGB", (size, size), (18, 22, 30))
    draw = ImageDraw.Draw(img)

    def px(xy):
        u = (xy[0] - entrance[0]) / half_view
        v = (xy[1] - entrance[1]) / half_view
        return (size / 2 * (1 + u), size / 2 * (1 - v))

    # Zones are for the peg CENTRE (what the trace shows), not the hole outline.
    h = hole_width(cfg.scene) / 2
    chamfer_zone = h - cfg.scene.peg_radius + cfg.scene.socket_chamfer  # peg rim reaches the chamfer
    direct_zone = h - cfg.scene.peg_radius  # peg drops straight in
    for half, style in ((chamfer_zone, dict(outline=(200, 140, 60), width=2)), (direct_zone, dict(fill=(60, 170, 90)))):
        a, b = px((entrance[0] - half, entrance[1] + half)), px((entrance[0] + half, entrance[1] - half))
        draw.rectangle([a, b], **style)

    near = np.abs(tips[: upto + 1, 2] - entrance[2]) < 0.006
    points = [px(p) for p in tips[: upto + 1][near]]
    if len(points) > 1:
        draw.line(points, fill=(240, 200, 80), width=2)
    bx, by = px(believed)
    draw.line([(bx - 7, by - 7), (bx + 7, by + 7)], fill=(230, 80, 80), width=3)
    draw.line([(bx - 7, by + 7), (bx + 7, by - 7)], fill=(230, 80, 80), width=3)
    tx, ty = px(tips[upto])
    draw.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], fill=(255, 255, 255))
    depth_mm = (entrance[2] - tips[upto, 2]) * 1e3
    draw.text((10, 8), f"top-down: peg centre path (±9 mm)   depth {depth_mm:+5.1f} mm", fill=(220, 220, 220), font=font)
    draw.text((10, size - 88), "green: centre here = drops straight in", fill=(60, 190, 100), font=font)
    draw.text((10, size - 60), "orange: rim reaches the chamfer", fill=(210, 150, 70), font=font)
    draw.text((10, size - 32), "red x: believed entrance (true hole = centre)", fill=(230, 100, 100), font=font)
    return np.asarray(img)


def set_frame(sim: Sim, q: np.ndarray) -> None:
    sim.data.qpos[sim.qpos_adr] = q
    mujoco.mj_forward(sim.model, sim.data)


def frame_indices(times: np.ndarray, fps: float, speed: float) -> np.ndarray:
    """Recorded-step index for each output frame at `fps`, playing at `speed`x real time."""
    frame_times = np.arange(0.0, times[-1], speed / fps)
    return np.searchsorted(times, frame_times).clip(0, len(times) - 1)


def camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render_video(sim: Sim, traj, cfg: TrialConfig, out: Path, fps: float, speed: float, title: str = "") -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    height, width = 540, 600
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(sim.model, height, width)
    socket, _ = sim.site_pose("socket_entrance")
    believed = believed_entrance(cfg, socket)
    views = [
        camera(socket + [0.05, -0.15, 0.20], 1.3, 120, -20),  # whole arm
        camera(socket + [0, 0, 0.005], 0.16, 135, -8),  # socket close-up, low so the wrist doesn't hide the entrance
    ]
    times, forces = traj["time"], np.linalg.norm(traj["contact_force"], axis=1)
    tips = traj["tip_pos"]
    font = ImageFont.load_default(size=20)
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as writer:
        for i in frame_indices(times, fps, speed):
            set_frame(sim, traj["q"][i])
            tiles = []
            for cam in views:
                renderer.update_scene(sim.data, cam)
                tiles.append(renderer.render())
            tiles.append(trace_tile(height, cfg, socket, believed, tips, i, font))
            image = Image.fromarray(np.hstack(tiles))
            draw = ImageDraw.Draw(image)
            label = f"{title}   " if title else ""
            draw.text((14, 10), f"{label}t = {times[i]:5.2f} s   contact force = {forces[i]:5.1f} N", fill=(255, 255, 255), font=font)
            if speed != 1.0:
                draw.text((14, 36), f"{speed:g}x speed", fill=(200, 200, 200), font=font)
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
    parser.add_argument("--title", default="", help="label drawn on the video")
    args = parser.parse_args()

    sim, traj, cfg = load(args.trajectory)
    if args.video:
        render_video(sim, traj, cfg, args.video, args.fps, args.speed, args.title)
        return
    try:
        view_interactive(sim, traj, args.speed)
    except RuntimeError as err:
        if "mjpython" in str(err):
            raise SystemExit(f"On macOS the interactive viewer needs mjpython:\n  uv run mjpython {' '.join(__import__('sys').argv)}")
        raise


if __name__ == "__main__":
    main()
