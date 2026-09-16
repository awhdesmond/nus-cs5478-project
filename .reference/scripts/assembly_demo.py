"""Render the hand-scripted ground-truth assembly (M1 scene check) to an MP4.

    uv run python scripts/assembly_demo.py                       # default layout
    uv run python scripts/assembly_demo.py --seed 3 --out results/demos/assembly_seed3.mp4

Left: angled view of the workspace. Right: the overhead camera image the perception stage will use.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from insertion.assembly_check import scripted_assembly
from insertion.assembly_scene import build_assembly_model, sample_layout
from insertion.config import RESULTS_DIR, AssemblyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="random connector layout (default: config layout)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "demos" / "assembly_scripted.mp4")
    parser.add_argument("--speed", type=float, default=2.0)
    args = parser.parse_args()

    cfg = AssemblyConfig() if args.seed is None else sample_layout(AssemblyConfig(), args.seed)
    result = scripted_assembly(cfg, record=True)
    print(*result.stages, sep="\n")
    print(f"both seated at end: {result.both_seated_at_end}")

    model = build_assembly_model(cfg)
    data = mujoco.MjData(model)
    h, w = 540, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, h, w)
    side = mujoco.MjvCamera()
    side.lookat[:] = [-0.03, 0.50, 0.22]
    side.distance, side.azimuth, side.elevation = 0.75, 150, -30
    font = ImageFont.load_default(size=22)

    snapshot_dt = 10 * model.opt.timestep
    stride = max(1, round(args.speed / (30 * snapshot_dt)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=30, macro_block_size=1) as writer:
        for i in range(0, len(result.trajectory), stride):
            data.qpos[:] = result.trajectory[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, side)
            left = renderer.render()
            renderer.update_scene(data, camera="overhead")
            right = renderer.render()
            frame = Image.fromarray(np.hstack([left, right]))
            draw = ImageDraw.Draw(frame)
            draw.text((12, 10), f"scripted ground-truth assembly   t = {i * snapshot_dt:5.1f} s   ({args.speed:g}x)", fill=(255, 255, 255), font=font)
            draw.text((w + 12, 10), "overhead camera", fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(frame))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
