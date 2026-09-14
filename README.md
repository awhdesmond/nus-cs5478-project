# Wire Connector Insertion with Compliant Control

CS5478 Intelligent Robots project. See [insertion-project-plan.md](insertion-project-plan.md) (source of truth) and [insertion-project-proposal.md](insertion-project-proposal.md).

## Setup

```bash
./scripts/fetch_assets.sh          # UR5e model from mujoco_menagerie (pinned commit) → third_party/
uv sync                            # Python 3.12 venv + deps
uv run python scripts/smoke_test.py
uv run pytest
uv run python scripts/view.py      # interactive viewer
```

## Layout

```
src/insertion/
  config.py              scene / uncertainty / task parameters (all tunables live here)
  scene.py               build the MuJoCo model (MjSpec) for a trial
  sim.py                 stepping, state, F/T and contact readout
  harness.py             N-trial runner + per-trial logging
  control/
    jacobian_step.py     week 1 Cartesian step → week 2 DLS IK
scripts/                 entry points (smoke test, viewer, easy insert)
tests/
results/                 trial CSVs and plots (gitignored)
third_party/             fetched assets (gitignored)
```

## Week 1 checklist

- [x] MuJoCo installed, stock UR5e loads and steps
- [ ] Swap position actuators → torque motors; joint PD + gravity comp holds home
- [ ] Socket (box geoms) + round peg welded to flange, 8–10 mm clearance
- [ ] Contact sanity check (≤4 hrs tuning)
- [ ] Wrist F/T sensors: tare + low-pass filter; contact forces readable
- [ ] Harness skeleton logging success / cycle time / peak force
- [ ] Jacobian step `Δq = J⁺·Δx`
- [ ] Hardcoded straight-down insert succeeds (`scripts/easy_insert.py`)
