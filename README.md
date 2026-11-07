# Vision-Guided Contact-Rich Connector Insertion with Compliant Control

## Setup

```bash
./scripts/fetch_assets.sh   # UR5e from mujoco_menagerie at a pinned commit → third_party/

uv sync
uv run pytest
uv run python scripts/view.py
uv run python scripts/capture.py --seed 0
```
