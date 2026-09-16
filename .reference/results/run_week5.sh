#!/usr/bin/env bash
# Week 5 compute: finish the benchmark with 10 ms force-cap scoring, then the Plan A grid.
set -u
cd "$(dirname "$0")/.."
run() { uv run --quiet python scripts/evaluate.py "$@" 2>&1 | grep -v VIRTUAL_ENV | tail -1; }
for c in impedance spiral; do
  run --controller $c --frictions 0.3 0.8 --name benchmark_$c
  echo "BENCHMARK DONE $c $(date +%H:%M:%S)"
done
for c in straight guarded impedance spiral; do
  run --controller $c --frictions 0.3 0.8 --clearances 0.003 0.002 0.001 --uncertainties 0.001 0.003 0.005 --trials 50 --name plan_a_$c
  echo "PLAN_A DONE $c $(date +%H:%M:%S)"
done
echo "ALL DONE $(date +%H:%M:%S)"
