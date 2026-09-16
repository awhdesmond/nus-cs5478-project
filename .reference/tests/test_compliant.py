from insertion.config import SceneConfig, StartConfig, TaskConfig, TrialConfig, UncertaintyConfig
from insertion.control.compliant import ImpedanceInsert
from insertion.control.scripted import StraightDownInsert
from insertion.harness import run_trial


def trial(seed, clearance, uncertainty, friction=0.3, budget=20.0):
    return TrialConfig(
        seed=seed,
        scene=SceneConfig(clearance=clearance, friction=friction),
        uncertainty=UncertaintyConfig(translation_max=uncertainty),
        start=StartConfig(joint_noise=0.25),
        task=TaskConfig(cycle_time_budget=budget),
    )


def test_compliance_lets_the_chamfer_guide_an_offset_peg_in():
    # Seed 1: ~1.8 mm offset at 1 mm clearance — beyond the gap, inside the chamfer's capture range.
    cfg = trial(seed=1, clearance=0.001, uncertainty=0.002)
    stiff, _ = run_trial(cfg, StraightDownInsert())
    compliant, _ = run_trial(cfg, ImpedanceInsert())
    assert not stiff.success
    assert compliant.success and compliant.peak_force < cfg.task.force_cap


def test_blocked_peg_presses_with_bounded_force():
    # ±5 mm offset lands outside the chamfer: the peg sits on the surface. The setpoint leash keeps the
    # push near K_z·max_lead instead of winding up.
    result, _ = run_trial(trial(seed=1, clearance=0.001, uncertainty=0.005, budget=15.0), ImpedanceInsert())
    assert not result.seated and result.failure_reason == "timeout"
    assert result.peak_force_10ms < 8.0
    assert result.ft_contact_time == result.ft_contact_time  # not nan: F/T saw the contact
    assert 0.0 <= result.ft_detect_latency < 0.05
