import numpy as np
import pandas as pd

from insertion.config import SceneConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.control.scripted import StraightDownInsert
from insertion.harness import believed_entrance, run_trial, run_trials


def test_believed_entrance_is_seeded_and_bounded():
    cfg = TrialConfig(seed=7, uncertainty=UncertaintyConfig(translation_max=0.005))
    a = believed_entrance(cfg, np.zeros(3))
    b = believed_entrance(cfg, np.zeros(3))
    np.testing.assert_array_equal(a, b)
    assert np.all(np.abs(a[:2]) <= 0.005) and a[2] == 0.0


def test_zero_uncertainty_is_exact():
    cfg = TrialConfig(seed=3)
    np.testing.assert_array_equal(believed_entrance(cfg, np.ones(3)), np.ones(3))


def test_easy_mode_straight_down_insert_succeeds():
    result, _ = run_trial(TrialConfig(task=TaskConfig(cycle_time_budget=10.0)), StraightDownInsert())
    assert result.success and result.seated
    assert result.failure_reason == ""
    assert result.peak_force < TaskConfig().force_cap


def test_open_loop_insert_fails_under_uncertainty_at_tight_clearance():
    cfg = TrialConfig(
        seed=1,
        scene=SceneConfig(clearance=0.001),
        uncertainty=UncertaintyConfig(translation_max=0.005),
        task=TaskConfig(cycle_time_budget=8.0),
    )
    result, _ = run_trial(cfg, StraightDownInsert())
    assert not result.success
    assert result.reached_contact
    assert result.failure_reason == "timeout"


def test_run_trials_writes_csv_with_config_and_trajectories(tmp_path):
    configs = [TrialConfig(seed=s, task=TaskConfig(cycle_time_budget=10.0)) for s in range(2)]
    df = run_trials(configs, StraightDownInsert, tmp_path / "trials.csv", record_dir=tmp_path / "traj")
    on_disk = pd.read_csv(tmp_path / "trials.csv")
    assert len(on_disk) == 2
    assert {"success", "peak_force", "cfg.scene.clearance", "cfg.uncertainty.translation_max"} <= set(on_disk.columns)
    assert df.success.all()
    traj = np.load(next((tmp_path / "traj").glob("*.npz")))
    assert traj["q"].shape[1] == 6 and traj["action"].shape == traj["q"].shape
