import mpmath

from expapprox import errors
from expapprox.approximator import ExponentialApproximator

# minimax polynomial coefficients computed via MATLAB chebfun package (see coeff.m)
MINIMAX_POLY = {
    1: (
        "1.0201394465967895",
        "1.0302293604985662",
    ),
    2: (
        "0.5062836705685021",
        "1.0150876009849295",
        "0.9998487929242696",
    ),
    3: (
        "0.1681733019572182",
        "0.5050232905802932",
        "0.9999396041503092",
        "0.9999244965509334",
    ),
    4: (
        "0.0419594398629578",
        "0.1679215798287936",
        "0.4999836570457601",
        "0.9999622811704657",
        "1.0000001510806225",
    ),
    5: (
        "0.0083811120378278",
        "0.0419175264834640",
        "0.1666632564455734",
        "0.4999886914730604",
        "1.0000000647031471",
        "1.0000000754895673",
    ),
    6: (
        "0.0013956055937947",
        "0.0083751285786969",
        "0.0416660835945164",
        "0.1666641548191586",
        "0.5000000168413123",
        "1.0000000377260430",
        "0.9999999999190666",
    ),
}


class MinimaxPolynomialApproximator(ExponentialApproximator):
    """Order-N minimax polynomial approximator of the exponential function on [-log(2)/2, log(2)/2]."""

    __slots__ = ("order", "coefficients")

    def __call__(self, x: int | float) -> float:
        return mpmath.polyval(self.coefficients, x)

    def __init__(self, order: int):
        if order not in MINIMAX_POLY:
            raise errors.ApproximatorError("Invalid order {order}; no minimax coefficients available")
        self.order = order
        self.coefficients = [mpmath.mpf(x) for x in MINIMAX_POLY[order]]

    def _fields(self):
        return [*super()._fields(), f"order={self.order}"]
