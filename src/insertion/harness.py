"""Evaluation harness: run N trials per condition and log one row per trial.

Kept policy-agnostic: a controller is anything that maps observations to actions, so a
learned policy can later drop in behind the same interface.
"""

from dataclasses import dataclass


@dataclass
class TrialResult:
    seed: int
    success: bool
    reached_contact: bool
    cycle_time: float  # [s sim time]
    peak_force: float  # [N]
    final_depth: float  # [m] insertion depth reached
    failure_reason: str = ""


def run_trials(configs, controller_factory, out_csv):
    """TODO (week 1): for each TrialConfig, build model, run controller, write TrialResult rows + config to CSV."""
    raise NotImplementedError
