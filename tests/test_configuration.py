import mpmath
import pytest

from expapprox.approximators import BitShiftPadeApproximator
from expapprox.tracker import IntegerTracker
from expapprox.utils import float_range
from plots.configuration import *

XS = [*float_range(0, X_UPPER, 0.01), X_UPPER]
PADE_SUBORDER = BitShiftPadeApproximator(DECIMALS, ORDER - 1)
PADE_ORDER = BitShiftPadeApproximator(DECIMALS, ORDER)


def compute_value_error(approximator: BitShiftPadeApproximator, xs: list[float]):
    return [C * abs(approximator(x) - mpmath.exp(x)) for x in xs]


def test_dollar_errors():
    # below chosen order exceeds bound
    assert not all(y <= ALPHA for y in compute_value_error(PADE_SUBORDER, XS))
    # chosen order stays below bound
    assert all(y <= ALPHA for y in compute_value_error(PADE_ORDER, XS))


def test_relative_errors():
    # below chosen order exceeds bound
    assert not all(y <= RELATIVE_ERROR_BOUND for y in PADE_SUBORDER.benchmark(XS))
    # chosen order stays below bound
    assert all(y <= RELATIVE_ERROR_BOUND for y in PADE_ORDER.benchmark(XS))


def test_bits():
    # intermediary values take up less than 128 bits
    assert PADE_ORDER.max_bits(XS) <= 128

    # product takes up less than 128 bits
    with IntegerTracker() as tracker:
        deposit = tracker.int(BAR_C)
        deposit * PADE_ORDER.approx(PADE_ORDER.to_fixed(X_UPPER))  # type: ignore
        assert tracker.bits <= 128


def test_continuous_to_annual():
    # fixed-point continuous compounding corresponds to annual rate
    fp_approx = PADE_ORDER.approx(PADE_ORDER.to_fixed(RATE_CONTINUOUS * GAMMA))
    assert PADE_ORDER.to_float(fp_approx) == pytest.approx(1 + RATE_ANNUAL)
