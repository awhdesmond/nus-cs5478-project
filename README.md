# Vision-Guided Contact-Rich Connector Insertion with Compliant Control

## Setup

```bash
./scripts/fetch_assets.sh   # UR5e from mujoco_menagerie at a pinned commit → third_party/

uv sync
uv run pytest
```

Proposal, pitch and milestone slides live in `docs/pitch/`.
`.reference/` holds the first attempt (two connectors in separate socket blocks).

## Layout

```
src/
  sim/          the simulated world (ur5e.py: menagerie UR5e)
  perception/   overhead RGB-D → part, board and port poses
  motion/       IK and trajectories
  process/      one trial's configuration
  harness/      evaluation
scripts/        fetch_assets.sh
tests/
```
