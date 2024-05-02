import math

import pytest

from expapprox import errors
from expapprox.approximators import PadeApproximator
from expapprox.utils import float_range

DECIMALS = 10
CP_DECIMALS = 4


def test_invalid_order():
    # order-0 fails
    with pytest.raises(errors.ApproximatorError):
        PadeApproximator(DECIMALS, 0)
    # order-1 works
    PadeApproximator(DECIMALS, 1)


def test_critical_points():
    # test that evaluation after critical point fails (denominator <= 0 <=> even sum == odd sum)
    pa_1 = PadeApproximator(CP_DECIMALS, 1)
    cp_1 = pa_1.to_fixed(2)  # 2 - x == 0 <=> x == 2
    # evaluation at critical point fails
    with pytest.raises(errors.ApproximatorDomainError):
        pa_1.approx(cp_1)
    # evaluation below critical point is fine
    pa_1.approx(cp_1 - 1)

    pa_3 = PadeApproximator(CP_DECIMALS, 3)
    cp_3 = pa_3.to_fixed(4.6444)  # 120 - 60*x + 12*x**2 - x**3 == 0 <=> x ~= 4.6444
    # evaluation at critical point fails
    with pytest.raises(errors.ApproximatorDomainError):
        pa_3.approx(cp_3)
    # evaluation below critical point is fine
    pa_3.approx(cp_3 - 1)


def test_constants():
    # test that high-order approximator produces expected approximations of known constants
    approximator = PadeApproximator(DECIMALS, 10)

    assert approximator(0.0) == pytest.approx(1.0)
    assert approximator(1.0) == pytest.approx(math.e)
    assert approximator(-1.0) == pytest.approx(1 / math.e)


def test_orders():
    xs = float_range(-1.9, 1.9, 0.1)

    # orders
    order_1 = PadeApproximator(DECIMALS, 1)
    order_2 = PadeApproximator(DECIMALS, 2)
    order_3 = PadeApproximator(DECIMALS, 3)
    order_4 = PadeApproximator(DECIMALS, 4)

    for x in xs:
        assert order_1(x) == pytest.approx((2 + x) / (2 - x))
        assert order_2(x) == pytest.approx((12 + 6 * x + x**2) / (12 - 6 * x + x**2))
        assert order_3(x) == pytest.approx(
            (120 + 60 * x + 12 * x**2 + x**3) / (120 - 60 * x + 12 * x**2 - x**3)
        )
        assert order_4(x) == pytest.approx(
            (1680 + 840 * x + 180 * x**2 + 20 * x**3 + x**4)
            / (1680 - 840 * x + 180 * x**2 - 20 * x**3 + x**4)
        )


def test_errors():
    # test that relative errors are below the Taylor approximator bound for the same order
    # https://en.wikipedia.org/wiki/Pade%27s_theorem#Example
    xs = float_range(-1, 1, 0.05)
    for order in range(1, 6):
        errs = PadeApproximator(DECIMALS, order).benchmark(xs)
        err_bound = 4 / math.factorial(order + 1)
        assert all(err < err_bound for err in errs)
