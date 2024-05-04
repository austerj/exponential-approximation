import math

import mpmath
import pytest

from expapprox import errors
from expapprox.approximators import BitShiftPadeApproximator
from expapprox.approximators.pade import PadeApproximator
from expapprox.utils import float_range

DECIMALS = 10
CP_DECIMALS = 4


def test_invalid_order():
    # order-0 fails
    with pytest.raises(errors.ApproximatorError):
        BitShiftPadeApproximator(DECIMALS, 0)
    # order-1 works
    BitShiftPadeApproximator(DECIMALS, 1)


def test_log2_precision():
    # test that increasing precision of log(2) does not change its fixed-point representation
    for decimals in [2, 5, 10, 16]:
        approximator = BitShiftPadeApproximator(decimals, 2)
        for extra_decimals in [decimals, decimals + 2, decimals + 4, decimals + 8]:
            with mpmath.workdps(extra_decimals):
                assert approximator.log2 == approximator.to_fixed(mpmath.log(2))


def test_constants():
    # test that high-order approximator produces expected approximations of known constants
    approximator = BitShiftPadeApproximator(DECIMALS, 10)

    assert approximator(0.0) == pytest.approx(1.0)
    assert approximator(1.0) == pytest.approx(math.e)
    assert approximator(-1.0) == pytest.approx(1 / math.e)


def test_exact():
    # test that approximator gives exact results when log2 divides x without remainder
    approximator = BitShiftPadeApproximator(DECIMALS, 10)

    for i in [-9, -2, 0, 3, 8, 24]:
        assert approximator.approx(i * approximator.log2) == approximator.to_fixed(2**i)


def test_log2_range():
    # test that approximator matches with regular Pade approximator for -log(2) < x < log(2)
    bitshifted_approximator = BitShiftPadeApproximator(DECIMALS, 10)
    log2 = bitshifted_approximator.log2
    approximator = PadeApproximator(DECIMALS, 10)

    for i in range(2, 10):
        x = log2 // i
        assert bitshifted_approximator.approx(log2 // i) == approximator.approx(log2 // i)


def test_orders():
    xs = float_range(-2, 2, 0.1)

    # orders
    order_1 = BitShiftPadeApproximator(DECIMALS, 1)
    order_2 = BitShiftPadeApproximator(DECIMALS, 2)
    order_3 = BitShiftPadeApproximator(DECIMALS, 3)
    order_4 = BitShiftPadeApproximator(DECIMALS, 4)

    for x in xs:
        unsigned_q, unsigned_r = divmod(abs(order_1.to_fixed(x)), order_1.log2)
        q = math.copysign(unsigned_q, x)
        r = math.copysign(order_1.to_float(unsigned_r), x)
        assert order_1(x) == pytest.approx(2**q * (2 + r) / (2 - r))
        assert order_2(x) == pytest.approx(2**q * (12 + 6 * r + r**2) / (12 - 6 * r + r**2))
        assert order_3(x) == pytest.approx(
            2**q * (120 + 60 * r + 12 * r**2 + r**3) / (120 - 60 * r + 12 * r**2 - r**3)
        )
        assert order_4(x) == pytest.approx(
            2**q
            * (1680 + 840 * r + 180 * r**2 + 20 * r**3 + r**4)
            / (1680 - 840 * r + 180 * r**2 - 20 * r**3 + r**4)
        )


def test_errors():
    # test that relative errors are below the Taylor approximator bound for the same order
    # https://en.wikipedia.org/wiki/Pade%27s_theorem#Example
    xs = float_range(-1, 1, 0.05)
    for order in range(1, 6):
        errs = BitShiftPadeApproximator(DECIMALS, order).benchmark(xs)
        err_bound = 4 / math.factorial(order + 1)
        assert all(err < err_bound for err in errs)
