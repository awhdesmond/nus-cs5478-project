# Wire Connector Insertion with Compliant Control

CS5478 Intelligent Robots project. See [docs/insertion-project-plan.md](docs/insertion-project-plan.md) (source of truth) and [docs/insertion-project-proposal.md](docs/insertion-project-proposal.md).

## Setup

```bash
./scripts/fetch_assets.sh          # UR5e model from mujoco_menagerie (pinned commit) → third_party/
uv sync                            # Python 3.12 venv + deps
uv run python scripts/smoke_test.py
uv run pytest
uv run python scripts/view.py --controller spiral --clearance 0.001 --uncertainty 0.005 --seed 1   # live physics; Ctrl+right-drag to shove
```

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
  controllers.py         controller registry (shared by evaluate.py and view.py)
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
  evaluate.py            run a controller over a clearance × uncertainty grid → trials.csv + summary.csv
  easy_insert.py         run the straight-down insert through the harness
  contact_sanity.py      contact health table; non-zero exit on failure
  compliance_check.py    impedance isolation tables → results/compliance_check/
  replay.py              replay a recorded trial in the viewer or render an MP4 (with top-down tip trace)
  make_demos.py          record + render the demo videos → results/demos/
  view.py                live interactive viewer (--controller straight|guarded|impedance|spiral, --friction)
  smoke_test.py, fetch_assets.sh
docs/
  insertion-project-plan.md, insertion-project-proposal.md, report.md
tests/
results/                 trial CSVs, trajectories and plots (gitignored)
third_party/             fetched assets (gitignored)
```
