# Vision-Guided Contact-Rich Peg Insertion with Compliant Control

CS5478 Intelligent Robots final project (Team 37). A simulated UR5e (MuJoCo) uses an overhead RGB-D
camera to find a round and a square peg and their ports, grasps each peg, and inserts it with a Cartesian
impedance controller, force/torque contact detection and a spiral search. A stiff position-control
baseline runs the same pipeline for comparison.

## Requirements

- Python 3.11+ (3.12 tested) and [uv](https://docs.astral.sh/uv/)
- `git` (to fetch the robot model)
- Tested on macOS; any OS supported by MuJoCo ≥ 3.3 should work

## Setup

```bash
./scripts/fetch_assets.sh   # UR5e from mujoco_menagerie at a pinned commit → third_party/
uv sync                     # create .venv and install dependencies from uv.lock
uv run pytest               # 100 tests, ~45 s
```

## Try it

```bash
uv run python scripts/view.py               # open the scene in MuJoCo's interactive viewer
uv run python scripts/capture.py --seed 0   # save one overhead RGB-D frame to results/capture/
```

## Reproduce the results

Run from the repository root. Every run is seeded, so the numbers in the paper are regenerated exactly.
Outputs go to `results/`.

| Command | Produces | Time |
| --- | --- | ---: |
| `uv run python scripts/scene_check.py` | Scene rest and keying checks | ~5 s |
| `uv run python scripts/perception_check.py` | Perception error over 50 layouts | ~20 s |
| `uv run python scripts/insertion_check.py` | Insertion module in isolation, 128 trials | ~2.5 min |
| `uv run python scripts/task_grid.py --seeds 20` | Full-task grid, clean perception, 320 tasks | ~3 min |
| `uv run python scripts/task_grid.py --seeds 20 --noise 2 --pose-error 0 1 2 3 --out results/robustness` | Robustness grid, 640 tasks (main result) | ~6 min |

`task_grid.py` runs trials in parallel (`--workers`, default 8).

## Code layout

| Package | Role |
| --- | --- |
| `src/sim/` | MuJoCo scene: torque-controlled UR5e, gripper, F/T sensor, pegs, ports, camera; seeded layouts |
| `src/perception/` | RGB-D image → kind, position and yaw of every peg and port |
| `src/motion/` | Kinematics, IK, trajectories, computed-torque and impedance controllers, wrist sensor |
| `src/grasp/` | Antipodal grasp planning from the perceived outline; pick and in-hand estimate |
| `src/insertion/` | Compliant insertion (impedance + contact + spiral search) and the stiff baseline |
| `src/harness/` | Trials, grids, pose-error injection, statistics (Wilson, McNemar); the only code that reads ground truth |
| `tests/` | Unit and integration tests |
| `scripts/` | Entry points listed above |
