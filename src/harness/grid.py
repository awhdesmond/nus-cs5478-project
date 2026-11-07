"""The full-task grid (proposal §4.2): clearance × friction × camera noise × seeds, both controllers on the
same seeds, then the summary tables the report needs."""

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from harness.stats import mcnemar_exact, wilson_interval
from harness.task import STAGES, run_task
from sim.config import SceneConfig

CONDITION = ["clearance", "friction", "noise", "pose_error"]


def _run(job):
    clearance, friction, noise, pose_error, seed, controller = job
    base = replace(SceneConfig(), clearance=clearance, friction=friction, depth_noise=noise)
    return run_task(base, seed, controller, pose_error)


def task_grid(seeds, clearances=(0.003, 0.001), frictions=(0.3, 0.8), noises=(0.0, 0.002), pose_errors=(0.0,),
              controllers=("compliant", "stiff"), workers: int = 1, progress=None) -> pd.DataFrame:
    """Every combination, one full-task trial each. workers > 1 runs trials in parallel processes (each trial
    builds its own simulation, so they're independent)."""
    jobs = list(itertools.product(clearances, frictions, noises, pose_errors, seeds, controllers))
    rows = []
    if workers == 1:
        for i, job in enumerate(jobs):
            rows.append(_run(job))
            if progress:
                progress(i + 1, len(jobs))
    else:
        with ProcessPoolExecutor(workers) as pool:
            for i, row in enumerate(pool.map(_run, jobs, chunksize=2)):
                rows.append(row)
                if progress:
                    progress(i + 1, len(jobs))
    return pd.DataFrame(rows)


def success_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per condition and controller: success rate with its Wilson 95 % interval, time and force."""
    def summary(group):
        n, k = len(group), int(group.success.sum())
        low, high = wilson_interval(k, n)
        forces = group.filter(like="_sustained_force").max(axis=1)
        peaks = group.filter(like="_peak_force").max(axis=1)
        return pd.Series(dict(trials=n, success=k / n, ci_low=low, ci_high=high,
                              time_median=group.loc[group.success, "task_time"].median(),
                              sustained_max=forces.max(), peak_max=peaks.max()))
    return results.groupby(CONDITION + ["controller"]).apply(summary, include_groups=False).round(3)


def comparison_table(results: pd.DataFrame, a: str = "compliant", b: str = "stiff") -> pd.DataFrame:
    """Per condition: the paired comparison of two controllers on identical seeds.

    a_only / b_only: seeds only that controller solved; McNemar p-value on those. Cycle time is compared only
    on seeds both solved (proposal: otherwise a controller that fails the hard cases looks faster).
    """
    rows = []
    for condition, group in results.groupby(CONDITION):
        pair = group.pivot_table(index="seed", columns="controller", values=["success", "task_time"], aggfunc="first")
        sa, sb = pair["success"][a].astype(bool), pair["success"][b].astype(bool)
        both = sa & sb
        time_diff = (pair["task_time"][a] - pair["task_time"][b])[both]
        rows.append(dict(zip(CONDITION, condition), seeds=len(pair), both=int(both.sum()),
                         **{f"{a}_only": int((sa & ~sb).sum()), f"{b}_only": int((sb & ~sa).sum())},
                         p_value=mcnemar_exact(int((sa & ~sb).sum()), int((sb & ~sa).sum())),
                         time_diff_median=time_diff.median() if len(time_diff) else np.nan))
    return pd.DataFrame(rows).set_index(CONDITION).round(4)


def failure_table(results: pd.DataFrame) -> pd.DataFrame:
    """How many trials failed at each stage (proposal's stage-wise breakdown), per controller."""
    failed = results[~results.success]
    table = failed.groupby("controller").failed_stage.value_counts().unstack(fill_value=0)
    return table.reindex(columns=list(STAGES), fill_value=0)


def perception_table(results: pd.DataFrame) -> pd.DataFrame:
    """Perception error against ground truth per camera-noise level: median / 95th percentile, mm and °."""
    cols = {"peg xy [mm]": ["round_peg_xy", "square_peg_xy"], "port xy [mm]": ["round_port_xy", "square_port_xy"],
            "square yaw [°]": ["square_peg_yaw", "square_port_yaw"]}
    one = results[results.controller == results.controller.iloc[0]]  # same perception for both controllers
    rows = {}
    for noise, group in one.groupby("noise"):
        row = {}
        for name, columns in cols.items():
            values = group[columns].to_numpy().ravel()
            values = values * (1e3 if "mm" in name else 180 / np.pi)
            row[f"{name} median"], row[f"{name} p95"] = np.nanmedian(values), np.nanpercentile(values, 95)
        rows[noise] = row
    return pd.DataFrame(rows).T.rename_axis("noise").round(3)
