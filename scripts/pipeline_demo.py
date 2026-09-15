"""Demo video of the perception-driven two-connector assembly (M3 + M4).

    uv run python scripts/pipeline_demo.py                                   # realistic camera, 1 mm clearance
    uv run python scripts/pipeline_demo.py --seed 7 --condition ideal --clearance 0.003

Panels: angled 3D view | live overhead camera | what perception saw at the last capture (detected outlines
labelled round/square, matched sockets, and the chosen antipodal grasp's finger contact line).
"""

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from insertion.assembly_scene import build_assembly_model, sample_layout
from insertion.assembly_skills import AssemblyRun
from insertion.camera import DepthNoise
from insertion.config import RESULTS_DIR, AssemblyConfig

CONDITIONS = {
    "ideal": DepthNoise(),
    "realistic": DepthNoise(sigma=0.001, rgb_sigma=3.0, calib_translation=0.002, calib_rotation_deg=0.5),
}
COLOURS = {"round": (60, 220, 120), "square": (240, 90, 240)}


def perception_overlay(observation, scene, proposals, size) -> np.ndarray:
    """Camera image with the estimator's output drawn on it (projected with the estimator's own calibration)."""
    image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR).copy()
    calib = observation.calibration
    scale = observation.rgb.shape[1] / size[0]
    thickness = max(2, int(3 * scale))

    def px(points_xyz):
        return np.round(calib.project(points_xyz)).astype(np.int32)

    for order, kind in enumerate(("round", "square"), start=1):
        colour = COLOURS[kind][::-1]
        conn = scene.connector(kind)
        if conn is not None:
            outline = np.column_stack([conn.outline_xy, np.full(len(conn.outline_xy), conn.top[2])])
            cv2.polylines(image, [px(outline)], True, colour, thickness)
            u, v = px(conn.top)[0]
            cv2.putText(image, f"{order}. {kind}", (u + 30, v - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, thickness + 1)
        socket = scene.socket(kind)
        if socket is not None:
            u, v = px(socket.entrance)[0]
            cv2.drawMarker(image, (int(u), int(v)), colour, cv2.MARKER_CROSS, int(40 * scale / 2), thickness)
            cv2.putText(image, f"{kind} socket", (u - 110, v + 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, thickness + 1)
    for grasp in proposals:
        d = np.array([np.cos(grasp.closing_angle), np.sin(grasp.closing_angle), 0.0])
        top = grasp.tcp + [0, 0, 0.010]
        a, b = px(np.stack([top + d * (grasp.width / 2 + 0.004), top - d * (grasp.width / 2 + 0.004)]))
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (255, 220, 40), thickness + 1)
        for p in (a, b):
            cv2.circle(image, tuple(map(int, p)), thickness * 3, (255, 220, 40), -1)
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=list(CONDITIONS), default="realistic")
    parser.add_argument("--clearance", type=float, default=0.001)
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = replace(sample_layout(AssemblyConfig(), args.seed), clearance=args.clearance, friction=args.friction)
    result = AssemblyRun(cfg, seed=args.seed, noise=CONDITIONS[args.condition], record=True).run()
    for log in result.connectors:
        print(f"{log.kind}: identified={log.identified} grasped={log.grasped} seated={log.seated} stayed={log.stayed_seated} "
              f"in-hand error {log.in_hand_tip_error_mm:.2f} mm, insert {log.insert_time:.1f} s, searched={log.search_info.get('searched')}")
    print(f"both seated at end: {result.both_seated_at_end}")

    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    w, h = 640, 480
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, h, w)
    side = mujoco.MjvCamera()
    side.lookat[:] = [-0.04, 0.50, 0.23]
    side.distance, side.azimuth, side.elevation = 0.70, 145, -28
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    overlays = [(t, perception_overlay(obs, scene, proposals, (w, h))) for t, obs, scene, proposals in result.captures]

    out = args.out or RESULTS_DIR / "demos" / f"pipeline_{args.condition}_seed{args.seed}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([t for t, _ in result.trajectory])
    frame_times = np.arange(times[0], times[-1], args.speed / 30)
    summary = "   ".join(f"{log.kind}: {'seated' if log.stayed_seated else 'FAILED'}" for log in result.connectors)
    with imageio.get_writer(out, fps=30, macro_block_size=1) as writer:
        for t in frame_times:
            i = min(int(np.searchsorted(times, t)), len(times) - 1)
            data.qpos[:] = result.trajectory[i][1]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, side)
            left = renderer.render().copy()
            renderer.update_scene(data, camera="overhead")
            middle = renderer.render().copy()
            seen = [img for tc, img in overlays if tc <= t]
            right = seen[-1] if seen else np.zeros((h, w, 3), np.uint8)
            canvas = Image.new("RGB", (3 * w, h + 40), (20, 22, 28))
            canvas.paste(Image.fromarray(np.hstack([left, middle, right])), (0, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), f"Perception-driven assembly: {args.condition} camera, {cfg.clearance * 1e3:.0f} mm clearance, "
                                f"friction {cfg.friction}   t = {t:5.1f} s ({args.speed:g}x)", fill=(240, 240, 240), font=font)
            draw.text((12, 48), "3D view", fill=(255, 255, 255), font=small)
            draw.text((w + 12, 48), "overhead camera (live)", fill=(255, 255, 255), font=small)
            draw.text((2 * w + 12, 48), "perception at last capture: outlines, sockets, grasp", fill=(255, 255, 255), font=small)
            if t > frame_times[-1] - 3:
                draw.text((2 * w + 12, h + 10), summary, fill=(120, 240, 150), font=font)
            writer.append_data(np.asarray(canvas))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
