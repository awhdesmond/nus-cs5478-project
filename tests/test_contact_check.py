import pytest

from insertion.contact_check import run_all


@pytest.mark.parametrize("check", run_all(clearance=0.001, timesteps=(0.001,)), ids=lambda c: c.scenario)
def test_contact_scenario_is_healthy_at_1mm_clearance(check):
    assert check.passed, check
