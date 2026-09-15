import pytest

from insertion.stats import mcnemar_exact, wilson


def test_wilson_interval_known_values():
    assert wilson(20, 20) == pytest.approx((0.8389, 1.0), abs=1e-3)
    assert wilson(10, 20) == pytest.approx((0.2993, 0.7007), abs=1e-3)
    lo, hi = wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.1611, abs=1e-3)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(9, 0) == pytest.approx(2 / 2**9)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 1) == pytest.approx(0.625)
