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
        # precompute powers and initialize accumulator to 1 (in fixed-point representation)
        x_pows = self.powers(x)
        accumulator = x.__class__(self.identity)
        for i, x_pow in enumerate(x_pows):
            # correction for final factorial division
            if i > 0:
                accumulator *= i + 1
            accumulator += x_pow
        return accumulator // self.factorial

    def powers(self, x: int) -> list[int]:
        """Compute powers x^1, x^2, ..., x^order as fixed-point numbers."""
        # compute x^2, x^3, ..., x^order
        x_pow = x
        x_pows = [(x_pow := (x_pow * x) // self.identity) for _ in range(1, self.order)]
        # prepend x^1
        x_pows.insert(0, x)
        return x_pows
