# Vision-Guided Contact-Rich Connector Insertion with Compliant Control

## Setup

```bash
./scripts/fetch_assets.sh

uv sync
uv run pytest
```

## Run

```bash
uv run python scripts/view.py --seed 3 --orientation mixed          # the scene; arm holds `ready`
uv run mjpython scripts/view.py --policy survey --seed 3             # watch a policy live (macOS: mjpython)
uv run python scripts/perceive.py --seed 3 --camera realistic        # perception overlay → results/perceive/
uv run python scripts/scene_check.py                                 # rest stability + keying
uv run python scripts/reach_check.py --layouts 20                    # reach every task pose on the full dynamics
uv run python scripts/perception_check.py                            # perception error vs ground truth
uv run python scripts/evaluate.py --policy survey --layouts 5        # a policy over a seeded grid
```

Report, plan and proposal live in `docs/`; report figures in `docs/figures/` and demo videos in `docs/demos/`.
`.reference/` holds the previous attempt (two connectors in separate socket blocks).

## Layout

```
src/
  sim/          the simulated world
    config.py      scene parameters: parts, board and ports, layout areas, gripper, camera, F/T, physics
    ur5e.py        menagerie UR5e with torque motors
    holes.py       chamfered holes from box primitives (rectangular, 16-gon) + board filler boxes
    scene.py       MjSpec scene: arm, gripper, F/T sensor, table, board with one port per part, parts, camera, keyframes
    layout.py      seeded part layouts (upright / mixed / lying pegs) and board pose
    sim.py         stepping, gripper, attach-on-grasp, ground truth (seated, wrong port/end, contact forces)
    sensors.py     F/T tare + low-pass, contact detector
    camera.py      overhead RGB-D rendering, calibration, back-projection, noise
    spec.py        TaskSpec: what the robot is told (geometry, port offsets on the board), never where things are
  perception/   overhead RGB-D → parts, board pose, port poses
    estimates.py   estimate types + PoseEstimator interface
    shapes.py      footprint identification (round / square / card) and the one-of-each prior
    geometric.py   GeometricEstimator (classical CV)
    noise.py       independent per-object pose error (robustness condition)
  motion/       free-space motion
    ik.py          damped least-squares IK with joint limits and restarts
    trajectory.py  quintic joint moves, trapezoidal Cartesian lines
    joint_pd.py    joint PD + gravity compensation
    segments.py    joint / line segments and a segment runner
    planning.py    lift → via → line approach
  process/      ProcessConfig: one trial (seed, scene, camera condition, per-object error, success criteria)
  harness/      evaluation
    robot.py       Robot: the policy's only view of the world (no ground truth; blind kinematic model)
    episode.py     run one trial and score each part (seated, wrong port/end, force cap, cycle time, failure stage)
    grid.py        seeded grids, parallel, CSV appended per episode
    scoring.py     perception estimate vs ground truth
    stats.py       Wilson intervals, exact McNemar, success tables
    policies.py    IdlePolicy, SurveyPolicy (perceive → visit each part and port)
    scene_check.py, reach_check.py, perception_check.py
  skills/       grasping and insertion (weeks 3–4)
scripts/        the commands above, plus fetch_assets.sh
tests/
```
