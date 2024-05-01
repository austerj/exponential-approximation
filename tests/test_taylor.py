import math

import pytest

from expapprox.approximators import TaylorApproximator
from expapprox.utils import float_range

DECIMALS = 10


def test_invalid_order():
    # order-0 fails
    with pytest.raises(ValueError):
        TaylorApproximator(DECIMALS, 0)
    # order-1 works
    TaylorApproximator(DECIMALS, 1)


def test_constants():
    # test that high-order approximator produces expected approximations of known constants
    approximator = TaylorApproximator(DECIMALS, 10)

    assert approximator(0.0) == pytest.approx(1.0)
    assert approximator(1.0) == pytest.approx(math.e)
    assert approximator(-1.0) == pytest.approx(1 / math.e)


def test_orders():
    xs = float_range(-2, 2, 0.1)

    # orders
    order_1 = TaylorApproximator(DECIMALS, 1)
    order_2 = TaylorApproximator(DECIMALS, 2)
    order_3 = TaylorApproximator(DECIMALS, 3)
    order_4 = TaylorApproximator(DECIMALS, 4)

    for x in xs:
        assert order_1(x) == pytest.approx(1 + x)
        assert order_2(x) == pytest.approx(1 + x + x**2 / 2)
        assert order_3(x) == pytest.approx(1 + x + x**2 / 2 + x**3 / 6)
        assert order_4(x) == pytest.approx(1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24)


def test_errors():
    # https://en.wikipedia.org/wiki/Taylor%27s_theorem#Example
    xs = float_range(-1, 1, 0.05)
    for order in range(1, 6):
        errs = TaylorApproximator(DECIMALS, order).benchmark(xs)
        err_bound = 4 / math.factorial(order + 1)
        assert all(err < err_bound for err in errs)
