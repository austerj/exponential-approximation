import pytest

from expapprox.utils import float_range, int_range


def test_int_range():
    assert int_range(-3, 2, 1) == [-3, -2, -1, 0, 1, 2]
    assert int_range(-3, 2, 2) == [-3, -1, 1]
    assert int_range(1, 4, 3) == [1, 4]
    assert int_range(1, 9, 3) == [1, 4, 7]
    assert int_range(0, 20, 5) == [0, 5, 10, 15, 20]
    assert int_range(0, 21, 5) == [0, 5, 10, 15, 20]


def test_float_range():
    assert float_range(-0.5, 0.5, 0.25) == [
        pytest.approx(-0.5),
        pytest.approx(-0.25),
        pytest.approx(0.0),
        pytest.approx(0.25),
        pytest.approx(0.5),
    ]
    assert float_range(0.1, 0.35, 0.1) == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    ]
    assert float_range(0.1, 0.35, 0.05) == [
        pytest.approx(0.1),
        pytest.approx(0.15),
        pytest.approx(0.2),
        pytest.approx(0.25),
        pytest.approx(0.3),
        pytest.approx(0.35),
    ]
