"""Save one overhead RGB-D frame as images: python scripts/capture.py [--seed N] [--noise 0.002]"""

import argparse
from dataclasses import replace
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from sim.camera import OverheadCamera
from sim.config import SceneConfig
from sim.layout import sample_layout
from sim.scene import build_model
from sim.sim import Sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0, help="depth noise σ [m]")
    parser.add_argument("--out", type=Path, default=Path("results/capture"))
    args = parser.parse_args()

    cfg = replace(sample_layout(SceneConfig(), args.seed), depth_noise=args.noise)
    sim = Sim(build_model(cfg), cfg)
    sim.reset("ready")  # arm out of the camera's view
    with OverheadCamera(sim.model, cfg) as camera:
        obs = camera.capture(sim.data, np.random.default_rng(args.seed))

    # Depth as a picture: height above the table, 0 (black) to 60 mm (white), so pegs stand out from blocks.
    height = obs.points()[..., 2] - cfg.table_height
    grey = (np.clip(height / 0.06, 0, 1) * 255).astype(np.uint8)
    args.out.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out / f"rgb_{args.seed}.png", obs.rgb)
    iio.imwrite(args.out / f"height_{args.seed}.png", grey)
    print(f"saved {args.out}/rgb_{args.seed}.png and height_{args.seed}.png")


if __name__ == "__main__":
    main()
