import math

import pytest

from expapprox.approximator import ExponentialApproximator


class MockApproximator(ExponentialApproximator):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def test_ref():
    approximator = MockApproximator()

    # exp(0) == 1
    assert approximator.ref(0.0) == 1.0

    # exp(1) == e ~= 2.71828
    assert approximator.ref(1.0) == pytest.approx(math.e)

    for float_x in [-1.2, -0.2, 0.005, -0.231, -5.4, 0.12, 0.93, 8.2]:
        # exp(log(|x|)) == |x|
        assert approximator.ref(math.log(abs(float_x))) == pytest.approx(abs(float_x))
        # reference (mpmath) is close to built-in math exponential
        assert approximator.ref(float_x) == pytest.approx(math.exp(float_x))
