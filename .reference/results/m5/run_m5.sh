#!/bin/sh
# M5 grids, run back to back. Don't edit code until both finish: the second grid imports the code when it starts.
cd "$(dirname "$0")/../.."
uv run --quiet python scripts/assembly_benchmark.py --orientations upright mixed --conditions realistic \
    --clearances 0.001 --frictions 0.3 --debris 0.004 --out m5_debris --workers 6 > results/m5/debris.log 2>&1
uv run --quiet python scripts/assembly_benchmark.py --orientations upright mixed --object-errors 0 0.002 \
    --out m5 --workers 6 > results/m5/grid.log 2>&1
echo done > results/m5/DONE
