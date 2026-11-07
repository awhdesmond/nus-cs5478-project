# Vision-Guided Contact-Rich Connector Insertion with Compliant Control

CS5478 Intelligent Robots project. See [docs/insertion-project-plan.md](docs/insertion-project-plan.md) (source of truth) and [docs/insertion-project-proposal.md](docs/insertion-project-proposal.md).

## Setup

```bash
./scripts/fetch_assets.sh          # UR5e model from mujoco_menagerie (pinned commit) → third_party/
uv sync                            # Python 3.12 venv + deps
uv run python scripts/smoke_test.py
uv run pytest
uv run python scripts/view.py --controller spiral --clearance 0.001 --uncertainty 0.005 --seed 1   # live physics; Ctrl+right-drag to shove
uv run mjpython scripts/view.py --scene assembly                      # watch the pipeline live, looping over layouts (upright + lying)
```

Report, plan and proposal live in `docs/`; report figures in `docs/figures/` and demo videos in `docs/demos/`.

## Layout

```
src/insertion/
  config.py              scene / motion / controller / sensor / task parameters (all tunables live here)
  scene.py               build the MuJoCo model (MjSpec): torque-controlled UR5e, welded peg, F/T sensor, socket
  sim.py                 stepping, state, raw F/T, ground-truth peg contacts
  sensors.py             F/T tare + low-pass filter (what controllers see)
  harness.py             controller interface, start perturbation, trial runner (parallel), CSV + trajectory logging
  contact_check.py       contact sanity scenarios and pass/fail thresholds
  reach_check.py         pre-insertion reach from perturbed starts
  compliance_check.py    impedance isolation checks: surface press, external push, tilted press
  assembly_scene.py      two-connector scene: gripper, sockets (round 16-gon + square), connectors (upright/lying, marked tip), camera, layouts
  assembly_sim.py        gripper commands, attach-on-grasp welds, ground-truth connector/socket state
  assembly_check.py      hand-scripted ground-truth assembly (M1 scene check)
  assembly_planning.py   perceived obstacles, arm clearance checks, sideways-insertion planning (lying connectors)
  camera.py              overhead RGB-D rendering, calibration, back-projection, noise model
  perception.py          PoseEstimator interface + GeometricEstimator (identify and localise connectors and sockets)
  perception_check.py    perception accuracy vs ground truth over random layouts
  grasping.py            antipodal grasp proposals from the perceived connector outline
  assembly_skills.py     perception-driven pick → insert → release skills (Robot interface without ground truth)
  controllers.py         controller registry (shared by evaluate.py and view.py)
  diagnose.py            spiral search failure diagnostics (phase log vs. ground-truth hole)
  stats.py               Wilson intervals, exact McNemar test
  control/
    ik.py                damped least-squares IK with joint limits and restarts
    trajectory.py        quintic joint trajectories, trapezoidal Cartesian lines
    joint_pd.py          joint PD + gravity compensation
    motion.py            joint / Cartesian-line segments and a segment runner
    scripted.py          collision-avoiding approach; straight-down insert (open loop)
    guarded.py           guarded-move + random-retry baseline
    impedance.py         Cartesian impedance law at torque level
    compliant.py         impedance insert (approach → impedance descent with leash); spiral search insert
scripts/
  evaluate.py            run a controller over a friction x clearance x uncertainty grid → trials.csv + summary.csv
  compare.py             paired comparison of two benchmarked controllers (CIs, McNemar, efficiency)
  diagnose_spiral.py     re-run failing spiral trials with diagnostics
  easy_insert.py         run the straight-down insert through the harness
  contact_sanity.py      contact health table; non-zero exit on failure
  compliance_check.py    impedance isolation tables → results/compliance_check/
  replay.py              replay a recorded trial in the viewer or render an MP4 (with top-down tip trace)
  make_demos.py          record + render the demo videos → results/demos/
  assembly_demo.py       render the scripted two-connector assembly (angled + overhead camera views)
  perception_check.py    perception accuracy table over layouts x noise levels → results/perception_check/
  assembly_benchmark.py  full pick-and-insert benchmark (layouts x clearance x friction x camera condition)
  pipeline_demo.py       demo video: 3D view | live overhead camera | perception overlay
  plots.py               report figures from the result CSVs → docs/figures/
  view.py                live interactive viewer (insertion: --controller …; --scene assembly: full pipeline, needs mjpython on macOS)
  smoke_test.py, fetch_assets.sh
docs/
  insertion-project-plan.md, insertion-project-proposal.md, report.md
tests/
results/                 trial CSVs, trajectories and plots (gitignored)
third_party/             fetched assets (gitignored)
```
