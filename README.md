# Wire Connector Insertion with Compliant Control

CS5478 Intelligent Robots project. See [docs/insertion-project-plan.md](docs/insertion-project-plan.md) (source of truth) and [docs/insertion-project-proposal.md](docs/insertion-project-proposal.md).

## Setup

```bash
./scripts/fetch_assets.sh          # UR5e model from mujoco_menagerie (pinned commit) → third_party/
uv sync                            # Python 3.12 venv + deps
uv run python scripts/smoke_test.py
uv run pytest
uv run python scripts/view.py --insert   # live physics: watch the insert, shove the arm with Ctrl+right-drag
```

## Layout

```
src/insertion/
  config.py              scene / controller / sensor / task parameters (all tunables live here)
  scene.py               build the MuJoCo model (MjSpec): torque-controlled UR5e, welded peg, F/T sensor, socket
  sim.py                 stepping, state, raw F/T, ground-truth peg contacts
  sensors.py             F/T tare + low-pass filter (what controllers see)
  harness.py             controller interface, trial runner, CSV + trajectory logging
  contact_check.py       contact sanity scenarios and pass/fail thresholds
  control/
    jacobian_step.py     Jacobian step / iterative IK (grows into DLS IK in week 2)
    joint_pd.py          joint PD + gravity compensation
    scripted.py          straight-down insert (week 1 easy mode)
scripts/
  easy_insert.py         run the straight-down insert through the harness
  contact_sanity.py      contact health table; non-zero exit on failure
  replay.py              replay a recorded trial in the viewer or render an MP4
  smoke_test.py, view.py, fetch_assets.sh
tests/
results/                 trial CSVs, trajectories and plots (gitignored)
third_party/             fetched assets (gitignored)
```
