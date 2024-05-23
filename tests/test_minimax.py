import math

from expapprox.approximators import PadeApproximator, TaylorApproximator
from expapprox.approximators.minimax import MinimaxPolynomialApproximator, MinimaxRationalApproximator
from expapprox.utils import float_range

DECIMALS = 10
XS = float_range(-math.log(2) / 2, math.log(2) / 2, 0.05)


def test_polynomial():
    # test that minimax polynomial has max relative errors strictly less than Taylor of same order
    for order in range(1, 5):
        minimax_rel_err = MinimaxPolynomialApproximator(order).benchmark(XS)
        taylor_rel_err = TaylorApproximator(DECIMALS, order).benchmark(XS)
        assert max(minimax_rel_err) < max(taylor_rel_err)


def test_rational():
    # test that minimax rational has max relative errors strictly less than Pade of same order
    for order in range(1, 5):
        minimax_rel_err = MinimaxRationalApproximator(order).benchmark(XS)
        pade_rel_err = PadeApproximator(DECIMALS, order).benchmark(XS)
        assert max(minimax_rel_err) < max(pade_rel_err)
