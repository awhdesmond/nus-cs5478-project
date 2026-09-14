import pytest

from insertion.reach_check import reach_from_start


@pytest.mark.parametrize("seed", range(8))
def test_pre_insertion_reach_from_perturbed_start(seed):
    result = reach_from_start(seed, joint_noise=1.0)
    assert result.passed, result
