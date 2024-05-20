import math

from expapprox import errors
from expapprox.approximator import ExponentialApproximator


class TaylorApproximator(ExponentialApproximator):
    """Order-N Taylor fixed-point approximator of the exponential function."""

    __slots__ = ("order", "factorial")

    def __init__(self, decimals: int, order: int):
        super().__init__(decimals)
        if order < 1:
            raise errors.ApproximatorError("Invalid order {order}; must be 1 or greater")
        self.order = order
        self.factorial = math.factorial(order)

    def _fields(self):
        return [*super()._fields(), f"order={self.order}"]

    def approx(self, x: int) -> int:
        # initialize accumulator to 1 + x (in fixed-point representation)
        accumulator = x.__class__(self.identity)
        accumulator += x
        # accumulate power terms
        x_pow = x
        for i in range(2, self.order + 1):
            # compute next power and rescale
            x_pow *= x
            x_pow //= self.identity
            accumulator *= i  # correction for final factorial division
            accumulator += x_pow
        return accumulator // self.factorial
