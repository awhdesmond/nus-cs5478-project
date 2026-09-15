import numpy as np

from insertion.config import SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.control.guarded import GuardedRetry
from insertion.harness import perturbed_start, run_trial, run_trials
from insertion.scene import build_model
from insertion.sim import Sim


def test_guarded_retry_inserts_after_being_blocked():
    cfg = TrialConfig(
        seed=0,
        scene=SceneConfig(clearance=0.003),
        uncertainty=UncertaintyConfig(translation_max=0.005),
        task=TaskConfig(cycle_time_budget=30.0),
    )
    result, _ = run_trial(cfg, GuardedRetry())
    assert result.success
    assert result.controller_info["contacts"] >= 1
    assert result.controller_info["attempts"] == result.controller_info["contacts"] + 1


def test_guarded_retry_first_attempt_succeeds_without_uncertainty():
    result, _ = run_trial(TrialConfig(task=TaskConfig(cycle_time_budget=15.0)), GuardedRetry())
    assert result.success and result.controller_info == {"attempts": 1, "contacts": 0}


def test_controller_info_is_logged_as_columns(tmp_path):
    configs = [TrialConfig(seed=s, task=TaskConfig(cycle_time_budget=15.0)) for s in range(2)]
    df = run_trials(configs, GuardedRetry, tmp_path / "trials.csv", workers=2)
    assert {"info.attempts", "info.contacts", "peak_force_10ms"} <= set(df.columns)


def test_perturbed_start_is_seeded_and_collision_free():
    cfg = TrialConfig(seed=5, start=StartConfig(joint_noise=1.0))
    starts = []
    for _ in range(2):
        sim = Sim(build_model(cfg.scene))
        sim.reset()
        perturbed_start(sim, cfg)
        assert sim.data.ncon == 0
        starts.append(sim.q())
    np.testing.assert_array_equal(*starts)
