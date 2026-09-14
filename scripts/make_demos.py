"""Record and render the demo videos into results/demos/.

    uv run python scripts/make_demos.py

Each demo is one benchmark trial (1 mm clearance, ±5 mm uncertainty, ±0.25 rad start noise),
so the same seed reproduces the same socket offset as in results/benchmark_*.
"""

import subprocess
import sys

from insertion.config import RESULTS_DIR, SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.controllers import CONTROLLERS
from insertion.harness import run_trial

# (file stem, controller, friction, seed, budget [s], playback speed, title)
DEMOS = [
    ("spiral_mu03_seed8", "spiral", 0.3, 8, 60.0, 1.0, "Spiral search, 1 mm clearance, mu=0.3"),
    ("impedance_mu03_seed8", "impedance", 0.3, 8, 20.0, 1.0, "Impedance only (no search), same trial"),
    ("spiral_mu08_seed1", "spiral", 0.8, 1, 60.0, 2.0, "Spiral search, 1 mm clearance, mu=0.8"),
]


def main():
    out = RESULTS_DIR / "demos"
    out.mkdir(parents=True, exist_ok=True)
    renders = []
    for stem, controller, friction, seed, budget, speed, title in DEMOS:
        cfg = TrialConfig(
            seed=seed,
            scene=SceneConfig(clearance=0.001, friction=friction),
            uncertainty=UncertaintyConfig(translation_max=0.005),
            start=StartConfig(joint_noise=0.25),
            task=TaskConfig(cycle_time_budget=budget),
        )
        result, traj = run_trial(cfg, CONTROLLERS[controller](), record=True)
        traj.save(out / f"{stem}.npz", cfg)
        print(f"{stem}: success={result.success} cycle={result.cycle_time:.1f}s {result.controller_info}")
        renders.append(subprocess.Popen([
            sys.executable, "scripts/replay.py", str(out / f"{stem}.npz"),
            "--video", str(out / f"{stem}.mp4"), "--speed", str(speed), "--title", title,
        ]))
    for proc in renders:
        proc.wait()


if __name__ == "__main__":
    main()
